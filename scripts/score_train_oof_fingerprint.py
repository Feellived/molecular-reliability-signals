#!/usr/bin/env python
"""train 분할에 폴드 외 지문 예측을 붙인다 (연구계획서 5.7절).

결합 규칙은 meta 분할에서 학습하는데, meta가 47개뿐인 물성이 있어 규칙이
잡음을 학습한다. train 분할을 규칙 학습에 함께 쓰면 표본이 여덟 배로 늘지만,
모델이 train 분자를 이미 학습했으므로 그 예측은 외운 값이라 쓸 수 없다.

폴드 외 예측이 이를 해결한다. train을 cv_fold로 다섯 조각으로 나누고, 조각
k에 속한 분자는 나머지 네 조각으로만 학습한 모델로 예측한다. 그러면 그 분자
입장에서는 처음 보는 모델이므로 예측이 정직해진다.

담당1의 분할에서 골격군이 fold 경계를 넘지 않는 것을 22종 전부에서 확인했다.
따라서 폴드 간 골격 누출이 없다.

변형도 같은 규칙을 따른다. 변형은 원본 행의 cv_fold를 상속하므로, 원본이
조각 k에 있으면 그 변형도 조각 k를 뺀 모델로 예측한다.

모델 설정은 담당2의 본학습과 동일하다. 시드 42~46의 평균을 예측으로, 표준편차를
불확실성으로 쓴다. 대표 모델도 담당2가 고른 것을 그대로 따른다.
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
from score_variants_fingerprint import (  # noqa: E402
    MODEL_NAMES,
    SEEDS,
    make_model,
    make_morgan_matrix,
    predict_values,
)


def process_dataset(
    dataset: str, processed_dir: Path, variants_dir: Path,
    reference_dir: Path, out_dir: Path,
) -> dict:
    started = time.perf_counter()

    splits = pd.read_csv(processed_dir / dataset / "splits.csv", low_memory=False)
    splits, _ = apply_known_repairs(splits)
    variants = pd.read_csv(variants_dir / dataset / "variants.csv")
    task_type = splits["task_type"].iloc[0]

    primary = json.loads(
        (reference_dir / dataset / "fingerprint_metrics.json").read_text(encoding="utf-8")
    )["primary_model"]

    origin_matrix = make_morgan_matrix(splits["parent_smiles"])
    variant_matrix = make_morgan_matrix(variants["variant_smiles"])
    labels = splits["Y_final"].astype(
        int if task_type == "classification" else float
    ).to_numpy()

    is_train = splits["split"].eq("train").to_numpy()
    fold = splits["cv_fold"].to_numpy()
    variant_fold = variants["cv_fold"].to_numpy()

    origin_out = {name: {"mean": np.full(len(splits), np.nan),
                         "std": np.full(len(splits), np.nan)} for name in MODEL_NAMES}
    variant_out = {name: {"mean": np.full(len(variants), np.nan),
                          "std": np.full(len(variants), np.nan)} for name in MODEL_NAMES}

    for k in sorted({int(f) for f in fold if f >= 0}):
        fit_rows = is_train & (fold != k)
        origin_rows = is_train & (fold == k)
        variant_rows = variant_fold == k
        if not origin_rows.any():
            continue
        for name in MODEL_NAMES:
            origin_stack, variant_stack = [], []
            for seed in SEEDS:
                model = make_model(name, task_type, seed)
                model.fit(origin_matrix[fit_rows], labels[fit_rows])
                origin_stack.append(
                    predict_values(model, origin_matrix[origin_rows], task_type)
                )
                if variant_rows.any():
                    variant_stack.append(
                        predict_values(model, variant_matrix[variant_rows], task_type)
                    )
            origin_stack = np.vstack(origin_stack)
            origin_out[name]["mean"][origin_rows] = origin_stack.mean(axis=0)
            origin_out[name]["std"][origin_rows] = origin_stack.std(axis=0, ddof=0)
            if variant_stack:
                variant_stack = np.vstack(variant_stack)
                variant_out[name]["mean"][variant_rows] = variant_stack.mean(axis=0)
                variant_out[name]["std"][variant_rows] = variant_stack.std(axis=0, ddof=0)

    def assemble(container, index, extra):
        frame = pd.DataFrame(extra)
        for name in MODEL_NAMES:
            frame[f"pred_{name}"] = container[name]["mean"]
            frame[f"std_{name}"] = container[name]["std"]
        frame["pred_fp_primary"] = frame[f"pred_{primary}"]
        frame["std_fp_primary"] = frame[f"std_{primary}"]
        frame["fp_primary_model"] = primary
        return frame.loc[index].reset_index(drop=True)

    origin_frame = assemble(
        origin_out, np.flatnonzero(is_train),
        {"row_uid": splits["row_uid"], "dataset": dataset,
         "split": splits["split"], "cv_fold": fold},
    )
    variant_frame = assemble(
        variant_out, np.flatnonzero(~np.isnan(variant_out[MODEL_NAMES[0]]["mean"])),
        {"variant_uid": variants["variant_uid"],
         "parent_row_uid": variants["parent_row_uid"], "dataset": dataset,
         "axis": variants["axis"], "split": variants["split"], "cv_fold": variant_fold},
    )

    dataset_out = out_dir / dataset
    dataset_out.mkdir(parents=True, exist_ok=True)
    origin_frame.to_csv(dataset_out / "origin_predictions_oof.csv", index=False)
    variant_frame.to_csv(dataset_out / "variant_predictions_fp_oof.csv", index=False)

    return {
        "dataset": dataset,
        "task_type": task_type,
        "primary_model": primary,
        "n_train": int(is_train.sum()),
        "n_origin_scored": len(origin_frame),
        "n_variant_scored": len(variant_frame),
        "elapsed_sec": round(time.perf_counter() - started, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="train 폴드 외 지문 채점")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--variants-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for index, dataset in enumerate(args.datasets, 1):
        if args.resume and (out_dir / dataset / "variant_predictions_fp_oof.csv").exists():
            print(f"[{index}/{len(args.datasets)}] {dataset}: 이미 있음", flush=True)
            continue
        record = process_dataset(
            dataset, Path(args.processed_dir), Path(args.variants_dir),
            Path(args.reference_dir), out_dir,
        )
        records.append(record)
        print(
            f"[{index}/{len(args.datasets)}] {dataset}: train {record['n_train']:,} "
            f"→ 원본 {record['n_origin_scored']:,} + 변형 {record['n_variant_scored']:,} "
            f"({record['elapsed_sec']}초)",
            flush=True,
        )

    if records:
        summary_dir = out_dir / "_summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(records)
        frame.to_csv(summary_dir / "train_oof_summary.csv", index=False)
        print()
        print(
            f"완료: {len(frame)}종, 원본 {int(frame['n_origin_scored'].sum()):,} + "
            f"변형 {int(frame['n_variant_scored'].sum()):,}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
