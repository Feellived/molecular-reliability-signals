#!/usr/bin/env python
"""변형 분자에 지문 모델 예측값을 붙인다 (연구계획서 5.8절 전단계).

담당2가 적합한 모델 객체는 저장되지 않았으므로 같은 설정으로 다시 적합한다.
설정은 담당2의 fingerprint_models.py와 run_fingerprint_dataset.py를 그대로 따른다.

  Morgan   radius 2, 2048비트, 카이랄리티 반영
  RF       n_estimators 500, max_features sqrt, 분류는 class_weight balanced
  XGBoost  n_estimators 300, max_depth 6, lr 0.05, subsample 0.8
  시드     42~46의 예측 평균을 pred, 표준편차를 std로 쓴다
  학습     train 분할만 사용

재적합 모델은 담당2의 원본 예측을 정확히 복원하지 못한다. scikit-learn과
XGBoost 버전이 달라 같은 시드에서도 난수 소비 순서가 달라지기 때문이다.
dili 기준 pred_rf 최대차가 0.054였다.

그래서 원본 분자와 변형을 모두 이 스크립트의 모델로 다시 채점한다. 분산은
같은 모델 안에서만 계산되므로 담당2 원본과의 차이가 신호를 오염시키지 않는다.
담당2 원본과의 차이는 reproduction_check.csv에 기록해 추적 가능하게 남긴다.

대표 모델(fp_primary)은 담당2가 train 전용 교차검증으로 고른 것을 그대로
따른다. 여기서 다시 고르면 기존 신호 파일과 대표 모델이 어긋날 수 있다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_repairs import apply_known_repairs  # noqa: E402
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

RDLogger.DisableLog("rdApp.*")

SEEDS = [42, 43, 44, 45, 46]
MODEL_NAMES = ["rf", "xgb"]
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
MORGAN_CHIRALITY = True

PRED_COLUMNS = [
    "pred_rf",
    "std_rf",
    "pred_xgb",
    "std_xgb",
    "pred_fp_primary",
    "std_fp_primary",
]


def make_morgan_matrix(smiles) -> np.ndarray:
    values = list(smiles)
    matrix = np.zeros((len(values), MORGAN_BITS), dtype=np.uint8)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS, fpSize=MORGAN_BITS, includeChirality=MORGAN_CHIRALITY
    )
    for index, value in enumerate(values):
        mol = Chem.MolFromSmiles(str(value))
        if mol is None:
            raise ValueError(f"Invalid SMILES at position {index}: {value!r}")
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), matrix[index])
    return matrix


def make_model(model_name: str, task_type: str, seed: int):
    if model_name == "rf":
        common = dict(
            n_estimators=500,
            random_state=seed,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=1,
        )
        if task_type == "classification":
            return RandomForestClassifier(class_weight="balanced", **common)
        return RandomForestRegressor(**common)
    common = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
    )
    if task_type == "classification":
        return XGBClassifier(eval_metric="logloss", **common)
    return XGBRegressor(objective="reg:squarederror", **common)


def predict_values(model, matrix: np.ndarray, task_type: str) -> np.ndarray:
    if task_type == "classification":
        return model.predict_proba(matrix)[:, 1]
    return model.predict(matrix)


def _assemble(per_model: dict, primary: str, index) -> pd.DataFrame:
    frame = pd.DataFrame(index=index)
    for name in MODEL_NAMES:
        frame[f"pred_{name}"] = per_model[name]["mean"]
        frame[f"std_{name}"] = per_model[name]["std"]
    frame["pred_fp_primary"] = frame[f"pred_{primary}"]
    frame["std_fp_primary"] = frame[f"std_{primary}"]
    return frame


def process_dataset(
    dataset: str,
    processed_dir: Path,
    variants_dir: Path,
    reference_dir: Path,
    out_dir: Path,
) -> dict:
    started = time.perf_counter()

    splits = pd.read_csv(processed_dir / dataset / "splits.csv", low_memory=False)
    splits, repaired = apply_known_repairs(splits)
    if repaired:
        print(f"    알려진 SMILES 교정 적용: {', '.join(repaired)}", flush=True)
    variants = pd.read_csv(variants_dir / dataset / "variants.csv")
    task_type = splits["task_type"].iloc[0]

    metrics_path = reference_dir / dataset / "fingerprint_metrics.json"
    primary = json.loads(metrics_path.read_text(encoding="utf-8"))["primary_model"]

    origin_matrix = make_morgan_matrix(splits["parent_smiles"])
    variant_matrix = make_morgan_matrix(variants["variant_smiles"])

    train_mask = splits["split"].eq("train").to_numpy()
    labels = splits["Y_final"].astype(
        int if task_type == "classification" else float
    ).to_numpy()

    origin_pred: dict = {}
    variant_pred: dict = {}
    for name in MODEL_NAMES:
        origin_stack, variant_stack = [], []
        for seed in SEEDS:
            model = make_model(name, task_type, seed)
            model.fit(origin_matrix[train_mask], labels[train_mask])
            origin_stack.append(predict_values(model, origin_matrix, task_type))
            variant_stack.append(predict_values(model, variant_matrix, task_type))
        origin_stack = np.vstack(origin_stack)
        variant_stack = np.vstack(variant_stack)
        origin_pred[name] = {
            "mean": origin_stack.mean(axis=0),
            "std": origin_stack.std(axis=0, ddof=0),
        }
        variant_pred[name] = {
            "mean": variant_stack.mean(axis=0),
            "std": variant_stack.std(axis=0, ddof=0),
        }

    origin_out = _assemble(origin_pred, primary, splits.index)
    origin_out.insert(0, "row_uid", splits["row_uid"])
    origin_out.insert(1, "dataset", dataset)
    origin_out.insert(2, "split", splits["split"])
    origin_out["fp_primary_model"] = primary

    variant_out = _assemble(variant_pred, primary, variants.index)
    variant_out.insert(0, "variant_uid", variants["variant_uid"])
    variant_out.insert(1, "parent_row_uid", variants["parent_row_uid"])
    variant_out.insert(2, "dataset", dataset)
    variant_out.insert(3, "axis", variants["axis"])
    variant_out.insert(4, "split", variants["split"])
    variant_out["fp_primary_model"] = primary

    dataset_out = out_dir / dataset
    dataset_out.mkdir(parents=True, exist_ok=True)
    origin_out.to_csv(dataset_out / "origin_predictions_refit.csv", index=False)
    variant_out.to_csv(dataset_out / "variant_predictions_fp.csv", index=False)

    reference = (
        pd.read_csv(reference_dir / dataset / "fingerprint_predictions.csv")
        .set_index("row_uid")
        .reindex(splits["row_uid"])
    )
    drift = {
        f"max_abs_diff_{column}": float(
            np.abs(origin_out[column].to_numpy() - reference[column].to_numpy()).max()
        )
        for column in PRED_COLUMNS
    }

    return {
        "dataset": dataset,
        "task_type": task_type,
        "primary_model": primary,
        "n_origin": len(splits),
        "n_variant": len(variants),
        "elapsed_sec": round(time.perf_counter() - started, 1),
        **drift,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="변형 분자 지문 모델 채점")
    parser.add_argument("--processed-dir", required=True, help="담당1 분할 산출물")
    parser.add_argument("--variants-dir", required=True, help="변형 테이블 최상위")
    parser.add_argument("--reference-dir", required=True, help="담당2 outputs 최상위")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    variants_dir = Path(args.variants_dir)
    reference_dir = Path(args.reference_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets or sorted(
        path.name for path in variants_dir.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    )

    summaries = []
    for index, dataset in enumerate(datasets, 1):
        target = out_dir / dataset / "variant_predictions_fp.csv"
        if args.resume and target.exists():
            print(f"[{index}/{len(datasets)}] {dataset}: 이미 있음, 건너뜀", flush=True)
            continue
        summary = process_dataset(
            dataset, processed_dir, variants_dir, reference_dir, out_dir
        )
        summaries.append(summary)
        print(
            f"[{index}/{len(datasets)}] {dataset}: "
            f"원본 {summary['n_origin']:,} + 변형 {summary['n_variant']:,} 채점 "
            f"({summary['elapsed_sec']}초, 대표 {summary['primary_model']}, "
            f"원본 대비 최대차 {summary['max_abs_diff_pred_fp_primary']:.3f})",
            flush=True,
        )

    if not summaries:
        print("새로 채점한 물성이 없다.")
        return 0

    summary_dir = out_dir / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(summaries)
    frame.to_csv(summary_dir / "reproduction_check.csv", index=False)

    print()
    print(
        f"완료: {len(frame)}종, 변형 {int(frame['n_variant'].sum()):,}건 채점"
    )
    print(
        "담당2 원본 예측과의 최대차 "
        f"평균 {frame['max_abs_diff_pred_fp_primary'].mean():.4f} / "
        f"최대 {frame['max_abs_diff_pred_fp_primary'].max():.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
