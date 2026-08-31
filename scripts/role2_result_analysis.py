"""담당 2 결과를 데이터셋별로 재분석하고 이상 결과를 표시한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)


def infer_task(labels: pd.Series) -> str:
    values = set(pd.to_numeric(labels, errors="coerce").dropna().unique())
    return "classification" if len(values) <= 2 and values <= {0, 1} else "regression"


def bh_adjust(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=float)
    adjusted = np.full(len(array), np.nan)
    valid = np.flatnonzero(np.isfinite(array))
    if not len(valid):
        return adjusted.tolist()
    order = valid[np.argsort(array[valid])]
    ranked = array[order] * len(valid) / np.arange(1, len(valid) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.clip(ranked, 0, 1)
    return adjusted.tolist()


def prediction_metrics(y: np.ndarray, prediction: np.ndarray, task: str) -> dict:
    if task == "classification":
        return {
            "roc_auc": roc_auc_score(y, prediction),
            "pr_auc": average_precision_score(y, prediction),
            "accuracy": accuracy_score(y, prediction >= 0.5),
        }
    return {
        "rmse": root_mean_squared_error(y, prediction),
        "mae": mean_absolute_error(y, prediction),
        "r2": r2_score(y, prediction),
        "nrmse_std": root_mean_squared_error(y, prediction) / max(np.std(y), 1e-12),
    }


def bootstrap_ci(y: np.ndarray, prediction: np.ndarray, task: str, repeats: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        index = rng.integers(0, len(y), len(y))
        if task == "classification" and len(np.unique(y[index])) < 2:
            continue
        metric = (
            roc_auc_score(y[index], prediction[index])
            if task == "classification"
            else mean_absolute_error(y[index], prediction[index])
        )
        values.append(metric)
    name = "roc_auc" if task == "classification" else "mae"
    return {
        "bootstrap_metric": name,
        "bootstrap_low": float(np.quantile(values, 0.025)),
        "bootstrap_high": float(np.quantile(values, 0.975)),
        "bootstrap_valid_repeats": len(values),
    }


def parse_set_contains(value: object, label: int) -> bool:
    text = str(value).replace(" ", "")
    return str(label) in text.strip("[]").split(",")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()
    metric_rows, conformal_rows, correlation_rows, missing = [], [], [], []
    prediction_columns = {
        "RandomForest": "pred_rf",
        "XGBoost": "pred_xgb",
        "Fingerprint primary": "pred_fp_primary",
        "ChemBERTa regular": "pred_chemberta_regular",
        "ChemBERTa augmented": "pred_chemberta_augmented",
    }
    for split_path in sorted(args.processed_dir.glob("*/splits.csv")):
        dataset = split_path.parent.name
        source = pd.read_csv(split_path, low_memory=False)
        signals_path = args.output_dir / dataset / "role2_signals.csv"
        if not signals_path.exists():
            missing.append(dataset)
            continue
        signals = pd.read_csv(signals_path, low_memory=False)
        frame = source.merge(signals, on="row_uid", how="left", validate="one_to_one", suffixes=("", "_signal"))
        task = infer_task(frame["Y_final"])
        test = frame[frame["split"].eq("test")].copy()
        y = pd.to_numeric(test["Y_final"]).to_numpy(int if task == "classification" else float)
        for model_name, column in prediction_columns.items():
            if column not in test:
                continue
            pred = pd.to_numeric(test[column]).to_numpy(float)
            row = {"dataset": dataset, "task_type": task, "model": model_name, "test_rows": len(test)}
            row.update(prediction_metrics(y, pred, task))
            row.update(bootstrap_ci(y, pred, task, args.bootstrap_repeats, 42))
            metric_rows.append(row)
        chem = pd.to_numeric(test.get("pred_chemberta_augmented"), errors="coerce").to_numpy(float)
        if task == "regression" and "conformal_lower" in test:
            lower = pd.to_numeric(test["conformal_lower"]).to_numpy(float)
            upper = pd.to_numeric(test["conformal_upper"]).to_numpy(float)
            conformal_rows.append({
                "dataset": dataset, "task_type": task, "test_rows": len(test),
                "coverage": float(np.mean((y >= lower) & (y <= upper))),
                "mean_width": float(np.mean(upper - lower)),
                "median_width": float(np.median(upper - lower)),
            })
        elif task == "classification" and "aps_prediction_set" in test:
            covered = [parse_set_contains(value, int(label)) for value, label in zip(test["aps_prediction_set"], y)]
            size = pd.to_numeric(test["aps_set_size"]).to_numpy(float)
            conformal_rows.append({
                "dataset": dataset, "task_type": task, "test_rows": len(test),
                "coverage": float(np.mean(covered)),
                "mean_set_size": float(np.mean(size)),
                "fraction_set_size_2": float(np.mean(size == 2)),
            })
        error = np.abs(y - chem)
        for signal in ["ad_knn_tanimoto_top5_mean", "ad_local_density_fraction_s040", "model_disagreement_abs"]:
            if signal not in test:
                continue
            values = pd.to_numeric(test[signal], errors="coerce").to_numpy(float)
            valid = np.isfinite(values) & np.isfinite(error)
            rho, pvalue = spearmanr(values[valid], error[valid])
            correlation_rows.append({
                "dataset": dataset, "task_type": task, "signal": signal,
                "spearman_rho_with_abs_error": rho, "p_value": pvalue,
            })
    metrics = pd.DataFrame(metric_rows)
    conformal = pd.DataFrame(conformal_rows)
    correlations = pd.DataFrame(correlation_rows)
    if len(correlations):
        correlations["p_value_bh"] = bh_adjust(correlations["p_value"].tolist())
        correlations["significant_bh_005"] = correlations["p_value_bh"] < 0.05
    args.report_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.report_dir / "dataset_model_metrics_detailed.csv", index=False)
    conformal.to_csv(args.report_dir / "conformal_detailed.csv", index=False)
    correlations.to_csv(args.report_dir / "reliability_correlations_bh.csv", index=False)
    xgb_issues = metrics[(metrics.get("model") == "XGBoost") & (metrics.get("task_type") == "regression")].sort_values("r2")
    xgb_issues.to_csv(args.report_dir / "xgboost_regression_diagnostics.csv", index=False)
    summary = {
        "analyzed_datasets": int(metrics["dataset"].nunique()) if len(metrics) else 0,
        "missing_datasets": missing,
        "negative_xgboost_r2_datasets": (
            xgb_issues.loc[xgb_issues["r2"] < 0, "dataset"].tolist() if len(xgb_issues) else []
        ),
        "target_coverage": 0.9,
        "datasets_below_coverage": (
            conformal.loc[conformal["coverage"] < 0.9, "dataset"].tolist() if len(conformal) else []
        ),
    }
    (args.report_dir / "followup_analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
