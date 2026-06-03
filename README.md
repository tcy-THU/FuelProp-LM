# FuelProp-LM Downstream Inference Package

This directory contains the scripts and processed downstream prompt datasets needed to reproduce downstream inference for FuelProp-LM. It does not include the base model weights or the fine-tuned LoRA adapter weights. Please download both from Hugging Face before running inference.

## Contents

```text
.
|-- downstream_single.py                 # deterministic single-generation downstream inference
|-- downstream_sampling_uq_conformal.py  # sampling-based uncertainty estimation
|-- evaluate_downstream_metrics.py       # MAE/RMSE/R2 evaluation from detailed CSV outputs
|-- prompter.py                          # prompt construction and response extraction helper
|-- templates/alpaca.json                # Alpaca prompt template used by prompter.py
|-- requirements.txt                     # minimal dependencies for inference and evaluation
`-- data/                                # processed downstream prompts, organized by shot setting
    |-- 0-shot/
    |-- 1-shot/
    |-- 2-shot/
    |-- 3-shot/
    |-- 4-shot/
    |-- 6-shot/
    `-- 8-shot/
```

The released downstream data include the fuel-property tasks used in the manuscript after removing RON, MON, and VP prompt files.

## Model Files

The inference scripts use relative paths by default. From this directory, place the base model and the fine-tuned LoRA adapter under:

```text
./models/Qwen2.5-7B-Instruct
./models/Qwen_100000
```

Download the base model from Hugging Face:

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir ./models/Qwen2.5-7B-Instruct
```

Download the fine-tuned FuelProp-LM LoRA adapter from Hugging Face:

```bash
huggingface-cli download tcy0512/FuelProp-LM \
  --local-dir ./models/Qwen_100000
```

After downloading, the directory layout should be:

```text
models/
|-- Qwen2.5-7B-Instruct/
|   |-- config.json
|   |-- tokenizer_config.json
|   |-- model-*.safetensors
|   `-- ...
`-- Qwen_100000/
    |-- adapter_config.json
    `-- adapter_model.safetensors
```

If `adapter_config.json` contains a local `base_model_name_or_path`, it can be changed to `Qwen/Qwen2.5-7B-Instruct`, or to the relative local directory `./models/Qwen2.5-7B-Instruct`.

## Environment

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

A CUDA GPU is recommended. The original scripts use 8-bit loading via `bitsandbytes` by default.

## Run Deterministic Downstream Inference

From this directory:

```bash
python downstream_single.py \
  --path ./data \
  --shots='[0,1,4,8]' \
  --base_model ./models/Qwen2.5-7B-Instruct \
  --lora_weights ./models/Qwen_100000 \
  --model_name qwen_100000 \
  --batch_size 64
```

The script writes detailed prediction CSV files under:

```text
./downstream_test0520/finetuned/qwen_100000/detailed_results/<shot>-shot/
```

## Compute MAE, RMSE, and R2

After inference, run:

```bash
python evaluate_downstream_metrics.py \
  --result_root ./downstream_test0520/finetuned/qwen_100000 \
  --shots 0 1 4 8
```

## Run Sampling-Based Uncertainty Inference

To reproduce the sampling-based uncertainty analysis, run 20 sampled generations per molecule:

```bash
python downstream_sampling_uq_conformal.py \
  --path ./data \
  --shot 8 \
  --sampling_n 20 \
  --temperature 0.7 \
  --top_p 0.9 \
  --top_k 50 \
  --base_model ./models/Qwen2.5-7B-Instruct \
  --lora_weights ./models/Qwen_100000 \
  --model_name qwen_100000 \
  --output_dir ./downstream_uq_sampling_conformal
```

The script saves per-sample prediction distributions and uncertainty metrics under:

```text
./downstream_uq_sampling_conformal/finetuned/qwen_100000/8-shot/
```

## Notes

- The downstream datasets in `data/` are processed prompt JSON files. They already contain the instruction, input prompt, and target output used for downstream inference.
- The number of shots corresponds to the number of in-context examples included in each prompt.
- This package is intended for downstream inference reproduction. It does not include scripts or data for reproducing LoRA fine-tuning from scratch.
