import json
import os
import random
import re
import sys
import time

import fire
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from peft import PeftModel
from sklearn.metrics import r2_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

from prompter import Prompter


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except Exception:
    pass


def parse_float_list(value):
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def extract_number_from_text(text):
    if text is None:
        return None

    boxed_match = re.search(r"\$\\boxed\{([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\}\$", text)
    if boxed_match:
        return boxed_match.group(1)

    boxed_match2 = re.search(r"\\boxed\{([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\}", text)
    if boxed_match2:
        return boxed_match2.group(1)

    answer_match = re.search(
        r"answer\s+is[:\s]+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
        text,
        re.IGNORECASE,
    )
    if answer_match:
        return answer_match.group(1)

    float_match = re.search(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", text)
    if float_match:
        return float_match.group(0)

    int_match = re.search(r"[-+]?\d+(?:[eE][-+]?\d+)?", text)
    if int_match:
        return int_match.group(0)

    return None


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def summarize_samples(values):
    arr = np.asarray(values, dtype=float)
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return {
            "pred_mean": np.nan,
            "pred_median": np.nan,
            "pred_std": np.nan,
            "pred_mad": np.nan,
            "pred_iqr": np.nan,
            "pred_min": np.nan,
            "pred_max": np.nan,
            "num_valid": 0,
            "invalid_rate": 1.0,
        }

    median = float(np.median(valid))
    return {
        "pred_mean": float(np.mean(valid)),
        "pred_median": median,
        "pred_std": float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
        "pred_mad": float(np.median(np.abs(valid - median))),
        "pred_iqr": float(np.percentile(valid, 75) - np.percentile(valid, 25)),
        "pred_min": float(np.min(valid)),
        "pred_max": float(np.max(valid)),
        "num_valid": int(len(valid)),
        "invalid_rate": float(1.0 - len(valid) / len(arr)),
    }


def split_calibration_test(n_rows, calibration_fraction, seed):
    indices = list(range(n_rows))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_cal = max(1, int(round(n_rows * calibration_fraction)))
    n_cal = min(n_cal, max(1, n_rows - 1))
    return set(indices[:n_cal]), set(indices[n_cal:])


def conformal_metrics(df, point_col, uncertainty_col, confidence_levels, calibration_fraction, seed, min_uncertainty):
    valid_df = df[np.isfinite(df["true_value"]) & np.isfinite(df[point_col])].copy()
    if len(valid_df) < 3:
        return df, []

    valid_df = valid_df.reset_index().rename(columns={"index": "_original_index"})
    cal_set, test_set = split_calibration_test(len(valid_df), calibration_fraction, seed)
    valid_df["split"] = ["calibration" if i in cal_set else "test" for i in range(len(valid_df))]

    residual = np.abs(valid_df["true_value"].to_numpy() - valid_df[point_col].to_numpy())
    raw_unc = valid_df[uncertainty_col].to_numpy()
    scaled_unc = np.where(np.isfinite(raw_unc), raw_unc, 0.0)
    scaled_unc = np.maximum(scaled_unc, min_uncertainty)
    scores = residual / scaled_unc
    valid_df["absolute_error"] = residual
    valid_df["scaled_uncertainty"] = scaled_unc
    valid_df["conformal_score"] = scores

    cal_scores = valid_df.loc[valid_df["split"] == "calibration", "conformal_score"].to_numpy()
    metric_rows = []

    for level in confidence_levels:
        alpha = 1.0 - float(level)
        q = float(np.quantile(cal_scores, 1.0 - alpha, method="higher"))
        lower_col = f"lower_{int(level * 100)}"
        upper_col = f"upper_{int(level * 100)}"
        width_col = f"width_{int(level * 100)}"
        covered_col = f"covered_{int(level * 100)}"

        valid_df[lower_col] = valid_df[point_col] - q * valid_df["scaled_uncertainty"]
        valid_df[upper_col] = valid_df[point_col] + q * valid_df["scaled_uncertainty"]
        valid_df[width_col] = valid_df[upper_col] - valid_df[lower_col]
        valid_df[covered_col] = (
            (valid_df["true_value"] >= valid_df[lower_col])
            & (valid_df["true_value"] <= valid_df[upper_col])
        )

        test_part = valid_df[valid_df["split"] == "test"]
        metric_rows.append(
            {
                "confidence_level": level,
                "qhat": q,
                "test_coverage": float(test_part[covered_col].mean()) if len(test_part) else np.nan,
                "test_mean_interval_width": float(test_part[width_col].mean()) if len(test_part) else np.nan,
                "calibration_size": int((valid_df["split"] == "calibration").sum()),
                "test_size": int((valid_df["split"] == "test").sum()),
            }
        )

    merged = df.copy()
    for _, row in valid_df.iterrows():
        original_index = int(row["_original_index"])
        for col in valid_df.columns:
            if col not in {"_original_index"}:
                merged.loc[original_index, col] = row[col]

    return merged, metric_rows


def rank_metrics(df, point_col, uncertainty_col):
    valid = df[np.isfinite(df["true_value"]) & np.isfinite(df[point_col]) & np.isfinite(df[uncertainty_col])].copy()
    if len(valid) < 3:
        return {
            "spearman_uncertainty_abs_error": np.nan,
            "pearson_uncertainty_abs_error": np.nan,
            "mae_all": np.nan,
            "rmse_all": np.nan,
            "r2_all": np.nan,
            "mae_keep_90pct_least_uncertain": np.nan,
            "mae_keep_80pct_least_uncertain": np.nan,
        }

    valid["absolute_error"] = np.abs(valid["true_value"] - valid[point_col])
    mae = float(valid["absolute_error"].mean())
    rmse = float(np.sqrt(np.mean((valid["true_value"] - valid[point_col]) ** 2)))
    r2 = float(r2_score(valid["true_value"], valid[point_col])) if len(valid) > 1 else np.nan
    spearman = float(valid[[uncertainty_col, "absolute_error"]].corr(method="spearman").iloc[0, 1])
    pearson = float(valid[[uncertainty_col, "absolute_error"]].corr(method="pearson").iloc[0, 1])

    ordered = valid.sort_values(uncertainty_col, ascending=True)
    keep_90 = ordered.iloc[: max(1, int(round(0.90 * len(ordered))))]
    keep_80 = ordered.iloc[: max(1, int(round(0.80 * len(ordered))))]

    return {
        "spearman_uncertainty_abs_error": spearman,
        "pearson_uncertainty_abs_error": pearson,
        "mae_all": mae,
        "rmse_all": rmse,
        "r2_all": r2,
        "mae_keep_90pct_least_uncertain": float(keep_90["absolute_error"].mean()),
        "mae_keep_80pct_least_uncertain": float(keep_80["absolute_error"].mean()),
    }


def main(
    load_8bit: bool = True,
    base_model: str = "./models/Qwen2.5-7B-Instruct",
    lora_weights: str = "./models/Qwen_100000",
    prompt_template: str = "",
    path: str = "./data",
    shot: int = 8,
    batch_size: int = 4,
    sampling_n: int = 20,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    max_new_tokens: int = 32,
    calibration_fraction: float = 0.2,
    confidence_levels: str = "0.9,0.95",
    uncertainty_col: str = "pred_std",
    point_col: str = "pred_median",
    min_uncertainty: float = 1e-6,
    seed: int = 42,
    model_name: str = "qwen_100000",
    output_dir: str = "./downstream_uq_sampling_conformal",
    limit_per_dataset: int = 0,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    confidence_levels = parse_float_list(confidence_levels)

    print(f"Running sampling UQ + conformal calibration for {shot}-shot")
    print(f"Dataset path: {path}")
    print(f"sampling_n={sampling_n}, temperature={temperature}, top_p={top_p}, top_k={top_k}")
    print(f"point_col={point_col}, uncertainty_col={uncertainty_col}")
    print("=" * 60)

    prompter = Prompter(prompt_template)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None or tokenizer.pad_token == "!":
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading model...")
    start_time = time.time()
    if device == "cuda":
        torch.cuda.empty_cache()
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=load_8bit,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map={"": 0},
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)
        model.to(device)

    model_type = "base" if not lora_weights or not lora_weights.strip() else "finetuned"
    if lora_weights and lora_weights.strip():
        print(f"Loading LoRA weights from {lora_weights}")
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={"": 0} if device == "cuda" else None,
        )
    else:
        print("No LoRA weights provided; using base model.")

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if not load_8bit and device == "cuda":
        model.half()
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    print(f"Model loaded in {time.time() - start_time:.2f}s")

    generation_config = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_beams=1,
        num_return_sequences=sampling_n,
    )

    shot_dir = os.path.join(path, f"{shot}-shot")
    if not os.path.exists(shot_dir):
        raise FileNotFoundError(f"Shot directory not found: {shot_dir}")

    file_list = sorted([f for f in os.listdir(shot_dir) if f.endswith(".json")])
    print(f"Found {len(file_list)} datasets in {shot_dir}")

    base_save_dir = os.path.join(output_dir, model_type, model_name, f"{shot}-shot")
    detail_dir = os.path.join(base_save_dir, "detailed_results")
    metrics_dir = os.path.join(base_save_dir, "metrics")
    os.makedirs(detail_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    all_metric_rows = []

    for file_idx, file_name in enumerate(file_list):
        dataset_name = file_name.rsplit(".", 1)[0]
        data_file = os.path.join(shot_dir, file_name)
        print(f"\n[{file_idx + 1}/{len(file_list)}] Processing {dataset_name}")
        raw_dataset = load_dataset("json", data_files=data_file)
        val_data = raw_dataset["train"]
        if limit_per_dataset and limit_per_dataset > 0:
            val_data = val_data.select(range(min(limit_per_dataset, len(val_data))))
        print(f"Samples: {len(val_data)}")

        detailed_results = []

        for start in tqdm(range(0, len(val_data), batch_size), desc=f"Sampling {dataset_name}"):
            end = min(start + batch_size, len(val_data))
            prompts = []
            rows = []

            for idx in range(start, end):
                row = val_data[idx]
                modified_instruction = (
                    row["instruction"]
                    + " Just give me a single numerical value without any explanation or additional text."
                )
                prompt = prompter.generate_prompt(modified_instruction, row["input"])
                prompts.append(prompt)
                rows.append(row)

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(device)

            with torch.no_grad():
                generated = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens,
                )

            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

            for local_idx, row in enumerate(rows):
                sample_texts = []
                sample_values = []
                for sample_idx in range(sampling_n):
                    flat_idx = local_idx * sampling_n + sample_idx
                    response = prompter.get_response(decoded[flat_idx])
                    sample_texts.append(response)
                    sample_values.append(safe_float(extract_number_from_text(response)))

                summary = summarize_samples(sample_values)
                true_value = safe_float(row["output"])
                pred_value = summary[point_col]

                detailed_results.append(
                    {
                        "index": start + local_idx,
                        "instruction": row["instruction"],
                        "input": row["input"],
                        "true_value": true_value,
                        "pred_value": pred_value,
                        "absolute_error": abs(pred_value - true_value)
                        if np.isfinite(pred_value) and np.isfinite(true_value)
                        else np.nan,
                        "relative_error": abs(pred_value - true_value) / abs(true_value) * 100
                        if np.isfinite(pred_value) and np.isfinite(true_value) and true_value != 0
                        else np.nan,
                        "pred_samples": json.dumps(sample_values, ensure_ascii=False),
                        "raw_responses": json.dumps(sample_texts, ensure_ascii=False),
                        **summary,
                    }
                )

        detail_df = pd.DataFrame(detailed_results)
        calibrated_df, conformal_rows = conformal_metrics(
            detail_df,
            point_col=point_col,
            uncertainty_col=uncertainty_col,
            confidence_levels=confidence_levels,
            calibration_fraction=calibration_fraction,
            seed=seed,
            min_uncertainty=min_uncertainty,
        )
        rank_row = rank_metrics(calibrated_df, point_col=point_col, uncertainty_col=uncertainty_col)

        detail_path = os.path.join(detail_dir, f"{dataset_name}_sampling_uq.csv")
        calibrated_df.to_csv(detail_path, index=False)
        print(f"Saved detailed UQ results to {detail_path}")

        dataset_metric_rows = []
        for row in conformal_rows:
            metric_row = {
                "shot": shot,
                "dataset": dataset_name,
                "sampling_n": sampling_n,
                "temperature": temperature,
                "top_p": top_p,
                "point_col": point_col,
                "uncertainty_col": uncertainty_col,
                **rank_row,
                **row,
            }
            dataset_metric_rows.append(metric_row)
            all_metric_rows.append(metric_row)

        metric_df = pd.DataFrame(dataset_metric_rows)
        metric_path = os.path.join(metrics_dir, f"{dataset_name}_metrics.csv")
        metric_df.to_csv(metric_path, index=False)
        print(f"Saved metrics to {metric_path}")

    summary_df = pd.DataFrame(all_metric_rows)
    summary_path = os.path.join(base_save_dir, "summary_sampling_uq_conformal.csv")
    summary_df.to_csv(summary_path, index=False)
    print("\nAll done.")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    fire.Fire(main)
