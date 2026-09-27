"""물성 하나의 Morgan RF/XGBoost 기준 모델을 학습한다."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
    roc_auc_score,
)

from data_validation import validate_split_file
from fingerprint_models import (
    MorganConfig,
    make_model,
    make_morgan_matrix,
    predict_values,
    save_fingerprint_cache,
)


SEEDS = [42, 43, 44, 45, 46]
MODEL_NAMES = ["rf", "xgb"]


def save_fitted_model(model, model_name: str, path: Path) -> None:
    """학습된 지문 모델을 모델 종류에 맞는 형식으로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if model_name == "xgb":
        model.save_model(path)
    elif model_name == "rf":
        joblib.dump(model, path, compress=3)
    else:
        raise ValueError(f"지원하지 않는 모델: {model_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/role2_local/outputs")
    )
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("data/role2_local/artifacts")
    )
    return parser.parse_args()


def metrics(y_true, prediction, task_type: str) -> dict[str, float]:
    if task_type == "classification":
        return {
            "roc_auc": float(roc_auc_score(y_true, prediction)),
            "pr_auc": float(average_precision_score(y_true, prediction)),
            "brier": float(brier_score_loss(y_true, prediction)),
            "log_loss": float(log_loss(y_true, prediction, labels=[0, 1])),
        }
    return {
        "rmse": float(root_mean_squared_error(y_true, prediction)),
        "mae": float(mean_absolute_error(y_true, prediction)),
        "r2": float(r2_score(y_true, prediction)),
    }


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    split_path = args.processed_dir / args.dataset / "splits.csv"
    validation = validate_split_file(split_path, check_smiles=True)
    if not validation.valid:
        raise ValueError(f"Dataset validation failed: {validation.errors}")
    frame = pd.read_csv(split_path, low_memory=False)
    task_type = validation.task_type
    config = MorganConfig()
    matrix = make_morgan_matrix(frame["parent_smiles"], config)
    if matrix.shape != (len(frame), config.n_bits):
        raise ValueError("Fingerprint matrix shape mismatch")
    if frame["row_uid"].duplicated().any():
        raise ValueError("row_uid must be unique before caching")
    cache_path = args.artifact_dir / args.dataset / "morgan_r2_2048_chiral.npz"
    save_fingerprint_cache(cache_path, frame["row_uid"], matrix, config)

    train_mask = frame["split"].eq("train").to_numpy()
    y = frame["Y_final"].astype(int if task_type == "classification" else float).to_numpy()
    folds = sorted(frame.loc[train_mask, "cv_fold"].unique().tolist())
    cv_results = {}
    for model_name in MODEL_NAMES:
        oof = np.full(len(frame), np.nan, dtype=float)
        for fold in folds:
            validation_mask = train_mask & frame["cv_fold"].eq(fold).to_numpy()
            fit_mask = train_mask & ~validation_mask
            if not validation_mask.any():
                raise ValueError(f"Fold {fold} is empty")
            if task_type == "classification" and len(np.unique(y[validation_mask])) < 2:
                raise ValueError(f"Fold {fold} cannot support classification metrics")
            model = make_model(model_name, task_type, SEEDS[0])
            model.fit(matrix[fit_mask], y[fit_mask])
            oof[validation_mask] = predict_values(
                model, matrix[validation_mask], task_type
            )
        if np.isnan(oof[train_mask]).any():
            raise ValueError(f"Incomplete train OOF predictions for {model_name}")
        cv_results[model_name] = metrics(y[train_mask], oof[train_mask], task_type)

    if task_type == "classification":
        primary_model = max(MODEL_NAMES, key=lambda name: cv_results[name]["roc_auc"])
    else:
        primary_model = min(MODEL_NAMES, key=lambda name: cv_results[name]["rmse"])
    predictions = {}
    saved_models: list[str] = []
    model_dir = args.artifact_dir / args.dataset / "models"
    for model_name in MODEL_NAMES:
        per_seed = []
        for seed in SEEDS:
            model = make_model(model_name, task_type, seed)
            model.fit(matrix[train_mask], y[train_mask])
            suffix = ".json" if model_name == "xgb" else ".joblib"
            model_path = model_dir / f"{model_name}_seed_{seed}{suffix}"
            save_fitted_model(model, model_name, model_path)
            saved_models.append(str(model_path))
            per_seed.append(predict_values(model, matrix, task_type))
        stack = np.vstack(per_seed)
        predictions[model_name] = {
            "mean": stack.mean(axis=0),
            "std": stack.std(axis=0, ddof=0),
        }

    output = frame[["row_uid", "dataset", "task_type", "split", "Y_final"]].copy()
    for model_name in MODEL_NAMES:
        output[f"pred_{model_name}"] = predictions[model_name]["mean"]
        output[f"std_{model_name}"] = predictions[model_name]["std"]
    output["pred_fp_primary"] = output[f"pred_{primary_model}"]
    output["std_fp_primary"] = output[f"std_{primary_model}"]
    output["fp_primary_model"] = primary_model

    split_metrics = {}
    for split_name in ["calib", "meta", "test"]:
        mask = frame["split"].eq(split_name).to_numpy()
        split_metrics[split_name] = {
            name: metrics(y[mask], predictions[name]["mean"][mask], task_type)
            for name in MODEL_NAMES
        }

    out_dir = args.output_dir / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = out_dir / "fingerprint_predictions.csv"
    metrics_path = out_dir / "fingerprint_metrics.json"
    output.to_csv(prediction_path, index=False)
    metrics_payload = {
        "dataset": args.dataset,
        "task_type": task_type,
        "rows": len(frame),
        "train_rows": int(train_mask.sum()),
        "seeds": SEEDS,
        "morgan": {
            "radius": config.radius,
            "n_bits": config.n_bits,
            "include_chirality": config.include_chirality,
        },
        "train_only_cv": cv_results,
        "primary_model": primary_model,
        "holdout_diagnostics": split_metrics,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "prediction_path": str(prediction_path),
        "fingerprint_cache": str(cache_path),
        "saved_models": saved_models,
    }
    metrics_path.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
