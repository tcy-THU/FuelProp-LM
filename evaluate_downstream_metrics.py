#!/usr/bin/env python
import argparse
import math
from pathlib import Path

import pandas as pd


def r2_score_manual(y_true, y_pred):
    mean_true = y_true.mean()
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - mean_true) ** 2).sum()
    if ss_tot == 0:
        return float("nan")
    return 1 - ss_res / ss_tot


def task_name_from_file(path, shot):
    suffix = f"_prompts_{shot}_shot_detailed"
    stem = path.stem
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    suffix = f"_{shot}_shot_detailed"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem.replace("_detailed", "")


def compute_metrics(csv_path, shot):
    df = pd.read_csv(csv_path)
    required = {"true_value", "pred_value"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

    values = df[["true_value", "pred_value"]].apply(pd.to_numeric, errors="coerce")
    before = len(values)
    values = values.dropna()
    dropped = before - len(values)

    y_true = values["true_value"]
    y_pred = values["pred_value"]
    err = y_pred - y_true

    mae = err.abs().mean()
    rmse = math.sqrt((err ** 2).mean())
    r2 = r2_score_manual(y_true, y_pred)

    return {
        "shot": shot,
        "task": task_name_from_file(csv_path, shot),
        "n": len(values),
        "dropped_non_numeric": dropped,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "source_file": str(csv_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MAE, RMSE, and R2 for downstream detailed prediction CSV files."
    )
    parser.add_argument(
        "--result_root",
        default="/root/autodl-fs/MGPT/downstream_test0520/finetuned/qwen_100000",
        help="Directory containing detailed_results/<shot>-shot/*_detailed.csv.",
    )
    parser.add_argument(
        "--shots",
        nargs="+",
        type=int,
        default=[4, 8],
        help="Shot values to evaluate, e.g. --shots 4 8.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to <result_root>/metrics_mae_rmse_r2.csv.",
    )
    parser.add_argument(
        "--pivot_output",
        default=None,
        help="Optional wide-format CSV path. Defaults to <result_root>/metrics_mae_rmse_r2_pivot.csv.",
    )
    args = parser.parse_args()

    result_root = Path(args.result_root)
    detail_root = result_root / "detailed_results"
    if not detail_root.exists():
        raise FileNotFoundError(f"Detailed result directory not found: {detail_root}")

    rows = []
    for shot in args.shots:
        shot_dir = detail_root / f"{shot}-shot"
        if not shot_dir.exists():
            print(f"[WARN] Missing shot directory, skipped: {shot_dir}")
            continue

        files = sorted(shot_dir.glob("*_detailed.csv"))
        if not files:
            print(f"[WARN] No detailed CSV files found in: {shot_dir}")
            continue

        for csv_path in files:
            rows.append(compute_metrics(csv_path, shot))

    if not rows:
        raise RuntimeError(f"No metrics computed under: {detail_root}")

    metrics = pd.DataFrame(rows).sort_values(["task", "shot"]).reset_index(drop=True)

    output = Path(args.output) if args.output else result_root / "metrics_mae_rmse_r2.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output, index=False)

    pivot_output = (
        Path(args.pivot_output)
        if args.pivot_output
        else result_root / "metrics_mae_rmse_r2_pivot.csv"
    )
    pivot = metrics.pivot(index="task", columns="shot", values=["MAE", "RMSE", "R2", "n"])
    pivot.columns = [f"{metric}_{shot}shot" for metric, shot in pivot.columns]
    pivot = pivot.reset_index()
    pivot.to_csv(pivot_output, index=False)

    print(f"Computed metrics for {len(metrics)} task-shot files.")
    print(f"Long-format metrics saved to: {output}")
    print(f"Wide-format metrics saved to: {pivot_output}")
    print()
    print(metrics[["shot", "task", "n", "MAE", "RMSE", "R2"]].to_string(index=False))


if __name__ == "__main__":
    main()
