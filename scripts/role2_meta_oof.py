"""meta 200건 미만 데이터에 두 종류의 cv-fold OOF 결과를 만든다.

1) base OOF: meta fold f를 예측할 때 train fold f를 제외한 지문 모델 사용
2) meta-level OOF: meta 신호로 base model의 오답/절대오차를 fold 밖에서 예측
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from fingerprint_models import MorganConfig, make_model, make_morgan_matrix, predict_values


SEEDS = [42, 43, 44, 45, 46]
MODELS = ["rf", "xgb"]
HANDOFF_COLUMNS = ["row_uid", "cv_fold", "pred_oof_rf", "pred_oof_xgb"]
FORBIDDEN_FEATURES = {
    "Y_final", "split", "dataset", "task_type", "aps_true_pvalue",
    "aps_calibrated_margin", "conformal_true_score", "row_uid",
}


def infer_task(labels: pd.Series) -> str:
    values = set(pd.to_numeric(labels, errors="coerce").dropna().unique())
    return "classification" if len(values) <= 2 and values <= {0, 1} else "regression"


def base_meta_oof(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    task = infer_task(frame["Y_final"])
    matrix = make_morgan_matrix(frame["parent_smiles"], MorganConfig())
    y = pd.to_numeric(frame["Y_final"]).to_numpy(int if task == "classification" else float)
    train = frame["split"].eq("train").to_numpy()
    meta = frame["split"].eq("meta").to_numpy()
    folds = [int(value) for value in sorted(frame.loc[meta, "cv_fold"].astype(int).unique())]
    result = frame.loc[meta, ["row_uid", "cv_fold", "Y_final"]].copy().reset_index(drop=True)
    meta_indices = np.flatnonzero(meta)
    predictions: dict[str, np.ndarray] = {
        name: np.full(meta.sum(), np.nan, dtype=float) for name in MODELS
    }
    stds = {name: np.full(meta.sum(), np.nan, dtype=float) for name in MODELS}
    for fold in folds:
        fit = train & ~frame["cv_fold"].eq(fold).to_numpy()
        target_global = np.flatnonzero(meta & frame["cv_fold"].eq(fold).to_numpy())
        target_local = np.flatnonzero(np.isin(meta_indices, target_global))
        for name in MODELS:
            per_seed = []
            for seed in SEEDS:
                model = make_model(name, task, seed)
                model.fit(matrix[fit], y[fit])
                per_seed.append(predict_values(model, matrix[target_global], task))
            stack = np.vstack(per_seed)
            predictions[name][target_local] = stack.mean(axis=0)
            stds[name][target_local] = stack.std(axis=0, ddof=0)
    for name in MODELS:
        if np.isnan(predictions[name]).any():
            raise ValueError(f"meta OOF prediction incomplete: {name}")
        result[f"pred_{name}_meta_fold_excluded"] = predictions[name]
        result[f"std_{name}_meta_fold_excluded"] = stds[name]
    metadata = {
        "method": "fit train excluding matching cv_fold; predict meta cv_fold",
        "task_type": task,
        "folds": folds,
        "seeds": SEEDS,
        "models": MODELS,
        "rows": len(result),
    }
    return result, metadata


def make_handoff_oof(base: pd.DataFrame) -> pd.DataFrame:
    """담당 4 전달용 OOF 파일을 정답 열 없이 고정된 형식으로 만든다."""
    renamed = base.rename(
        columns={
            "pred_rf_meta_fold_excluded": "pred_oof_rf",
            "pred_xgb_meta_fold_excluded": "pred_oof_xgb",
        }
    )
    missing = sorted(set(HANDOFF_COLUMNS) - set(renamed.columns))
    if missing:
        raise ValueError(f"OOF 전달 파일에 필요한 열이 없습니다: {missing}")
    result = renamed[HANDOFF_COLUMNS].copy()
    result["cv_fold"] = pd.to_numeric(result["cv_fold"], errors="raise").astype(int)
    if result["row_uid"].duplicated().any():
        raise ValueError("OOF 전달 파일의 row_uid가 중복됩니다")
    if result[["pred_oof_rf", "pred_oof_xgb"]].isna().any().any():
        raise ValueError("OOF 전달 파일의 예측값에 결측이 있습니다")
    return result


def meta_level_oof(frame: pd.DataFrame, signals: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    task = infer_task(frame["Y_final"])
    merged = frame[["row_uid", "split", "cv_fold", "Y_final"]].merge(
        signals, on="row_uid", how="left", validate="one_to_one", suffixes=("", "_signal")
    )
    meta = merged[merged["split"].eq("meta")].copy().reset_index(drop=True)
    prediction_column = (
        "pred_fp_primary" if "pred_fp_primary" in meta else
        "pred_chemberta_augmented"
    )
    base_prediction = pd.to_numeric(meta[prediction_column]).to_numpy(float)
    truth = pd.to_numeric(meta["Y_final"]).to_numpy(float)
    if task == "classification":
        target = ((base_prediction >= 0.5).astype(int) != truth.astype(int)).astype(int)
    else:
        target = np.abs(base_prediction - truth)
    numeric = meta.select_dtypes(include=[np.number]).columns
    features = [
        column for column in numeric
        if column not in FORBIDDEN_FEATURES
        and column not in {"cv_fold", "Y_final_signal"}
        and not column.startswith("conformal_true")
        and not column.startswith("aps_true")
    ]
    if not features:
        raise ValueError("meta-level CV에 사용할 안전한 숫자 신호가 없습니다")
    prediction = np.full(len(meta), np.nan, dtype=float)
    folds = [int(value) for value in sorted(meta["cv_fold"].astype(int).unique())]
    for fold in folds:
        valid = meta["cv_fold"].eq(fold).to_numpy()
        fit = ~valid
        if task == "classification":
            if len(np.unique(target[fit])) < 2:
                prediction[valid] = float(np.mean(target[fit]))
                continue
            estimator = RandomForestClassifier(
                n_estimators=500, min_samples_leaf=5, class_weight="balanced",
                random_state=42, n_jobs=1,
            )
        else:
            estimator = RandomForestRegressor(
                n_estimators=500, min_samples_leaf=5, random_state=42, n_jobs=1,
            )
        model = make_pipeline(SimpleImputer(strategy="median"), estimator)
        model.fit(meta.loc[fit, features], target[fit])
        if task == "classification":
            prediction[valid] = model.predict_proba(meta.loc[valid, features])[:, 1]
        else:
            prediction[valid] = model.predict(meta.loc[valid, features])
    result = meta[["row_uid", "cv_fold", "Y_final"]].copy()
    result["meta_reliability_target"] = target
    result["meta_reliability_oof_prediction"] = prediction
    metadata = {
        "method": "meta cv_fold out-of-fold RandomForest reliability baseline",
        "task_type": task,
        "base_prediction": prediction_column,
        "target": "incorrect" if task == "classification" else "absolute_error",
        "features": features,
        "folds": folds,
        "rows": len(result),
    }
    return result, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--meta-output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    datasets = []
    validation_rows = []
    for split_path in sorted(args.processed_dir.glob("*/splits.csv")):
        frame = pd.read_csv(split_path, low_memory=False)
        if int(frame["split"].eq("meta").sum()) >= args.limit:
            continue
        dataset = split_path.parent.name
        target = args.meta_output_dir / dataset
        target.mkdir(parents=True, exist_ok=True)
        base_path = target / "base_model_meta_oof.csv"
        if base_path.exists():
            base = pd.read_csv(base_path, low_memory=False)
            base_info = {
                "status": "existing_file_reused",
                "rows": len(base),
            }
        else:
            base, base_info = base_meta_oof(frame)
            base.to_csv(base_path, index=False)
        handoff = make_handoff_oof(base)
        handoff.to_csv(target / "oof_predictions.csv", index=False)
        expected = frame.loc[frame["split"].eq("meta"), ["row_uid", "cv_fold"]].copy()
        expected["cv_fold"] = expected["cv_fold"].astype(int)
        row_uids_match = set(handoff["row_uid"].astype(str)) == set(
            expected["row_uid"].astype(str)
        )
        folds_valid = set(handoff["cv_fold"]).issubset({0, 1, 2, 3, 4})
        valid = (
            len(handoff) == len(expected)
            and handoff["row_uid"].is_unique
            and row_uids_match
            and folds_valid
            and not handoff[["pred_oof_rf", "pred_oof_xgb"]].isna().any().any()
        )
        validation_rows.append(
            {
                "dataset": dataset,
                "expected_meta_rows": len(expected),
                "oof_rows": len(handoff),
                "row_uid_unique": bool(handoff["row_uid"].is_unique),
                "row_uids_match": row_uids_match,
                "folds_valid": folds_valid,
                "missing_predictions": int(
                    handoff[["pred_oof_rf", "pred_oof_xgb"]].isna().sum().sum()
                ),
                "valid": valid,
            }
        )
        signals_path = args.output_dir / dataset / "role2_signals.csv"
        meta_info = {"status": "skipped", "reason": "role2_signals.csv missing"}
        if signals_path.exists():
            meta, meta_info = meta_level_oof(frame, pd.read_csv(signals_path, low_memory=False))
            meta.to_csv(target / "meta_level_reliability_oof.csv", index=False)
        (target / "metadata.json").write_text(
            json.dumps({"base_oof": base_info, "meta_level_oof": meta_info}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        datasets.append(dataset)
        print(f"[DONE] {dataset}")
    validation = pd.DataFrame(validation_rows)
    validation_path = args.meta_output_dir / "oof_validation_summary.csv"
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation.to_csv(validation_path, index=False)
    if not validation.empty and not validation["valid"].all():
        failed = validation.loc[~validation["valid"], "dataset"].tolist()
        raise ValueError(f"OOF 검증 실패: {failed}")
    print(f"meta {args.limit}건 미만 처리: {len(datasets)}개")
    print(f"검증 요약: {validation_path}")


if __name__ == "__main__":
    main()
