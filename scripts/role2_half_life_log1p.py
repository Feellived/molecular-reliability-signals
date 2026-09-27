"""half_life_obach 회귀를 원본 라벨과 log1p 라벨로 비교한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_validation import validate_split_file
from fingerprint_models import MorganConfig, make_model, make_morgan_matrix, predict_values
from run_fingerprint_dataset import metrics, save_fitted_model


DATASET = "half_life_obach"
MODELS = ["rf", "xgb"]
SEEDS = [42, 43, 44, 45, 46]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_path = args.processed_dir / DATASET / "splits.csv"
    validation = validate_split_file(split_path, check_smiles=True)
    if not validation.valid or validation.task_type != "regression":
        raise ValueError(f"회귀 데이터 검증 실패: {validation.errors}")

    frame = pd.read_csv(split_path, low_memory=False)
    y = pd.to_numeric(frame["Y_final"], errors="raise").to_numpy(float)
    if np.any(y < 0):
        raise ValueError("log1p 비교에는 음수가 아닌 라벨이 필요합니다")
    matrix = make_morgan_matrix(frame["parent_smiles"], MorganConfig())
    train = frame["split"].eq("train").to_numpy()
    test = frame["split"].eq("test").to_numpy()
    if not train.any() or not test.any():
        raise ValueError("train 또는 test 행이 없습니다")

    output = frame[["row_uid", "split", "Y_final"]].copy()
    metric_rows = []
    saved_models = []
    model_dir = args.artifact_dir / "models"
    for model_name in MODELS:
        for target_mode in ["original", "log1p"]:
            fit_target = np.log1p(y) if target_mode == "log1p" else y
            per_seed = []
            for seed in SEEDS:
                model = make_model(model_name, "regression", seed)
                model.fit(matrix[train], fit_target[train])
                prediction = predict_values(model, matrix, "regression")
                if target_mode == "log1p":
                    prediction = np.expm1(prediction)
                per_seed.append(prediction)
                suffix = ".json" if model_name == "xgb" else ".joblib"
                model_path = model_dir / f"{model_name}_{target_mode}_seed_{seed}{suffix}"
                save_fitted_model(model, model_name, model_path)
                saved_models.append(str(model_path))

            mean_prediction = np.vstack(per_seed).mean(axis=0)
            column = f"pred_{model_name}_{target_mode}"
            output[column] = mean_prediction
            row = {
                "model": model_name,
                "target_mode": target_mode,
                **metrics(y[test], mean_prediction[test], "regression"),
                "prediction_min": float(mean_prediction[test].min()),
                "prediction_max": float(mean_prediction[test].max()),
            }
            metric_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "predictions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(args.output_dir / "metrics.csv", index=False)
    metadata = {
        "dataset": DATASET,
        "seeds": SEEDS,
        "models": MODELS,
        "train_rows": int(train.sum()),
        "test_rows": int(test.sum()),
        "log_transform": "log1p for fit; expm1 for prediction",
        "outlier_removal": False,
        "prediction_clipping": False,
        "saved_models": saved_models,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
