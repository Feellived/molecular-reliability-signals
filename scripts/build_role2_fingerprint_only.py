#!/usr/bin/env python
"""새 물성의 담당2 산출물 중 지문 모델 쪽만 만든다 (사전 기록 2 재현용).

담당2의 run_fingerprint_dataset.py와 role2_signals.py(적용가능도메인 부분)를 옮겼다.
설정은 그대로다.

  모델      RF, XGBoost 각 5시드(42~46), Morgan r=2, 2048비트, 카이랄리티 반영
  대표 모델  train 안의 골격 교차적합에서 분류는 ROC AUC, 회귀는 RMSE로 고른다
  적용가능도메인  train과의 타니모토 상위 5개 평균, 유사도 0.40 이상인 train 수

ChemBERTa는 학습하지 않는다. B축 효과가 지문 모델에서 나온다는 22종 결과에 따라
재현 실험을 지문 모델만으로 설계했기 때문이다(사전 기록 2).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import DataStructs
from sklearn.metrics import roc_auc_score, root_mean_squared_error

from score_variants_fingerprint import MODEL_NAMES, SEEDS, make_model, make_morgan_matrix, predict_values

K_NEIGHBORS = 5
DENSITY_THRESHOLD = 0.40


def to_bitvects(matrix: np.ndarray):
    out = []
    for row in matrix:
        bv = DataStructs.ExplicitBitVect(len(row))
        bv.SetBitsFromList(np.flatnonzero(row).tolist())
        out.append(bv)
    return out


def applicability_domain(matrix: np.ndarray, is_train: np.ndarray) -> pd.DataFrame:
    vectors = to_bitvects(matrix)
    train_idx = np.flatnonzero(is_train)
    train_pos = {int(g): p for p, g in enumerate(train_idx)}
    train_vectors = [vectors[i] for i in train_idx]
    k = min(K_NEIGHBORS, max(1, len(train_vectors) - 1))
    knn, density = [], []
    for i, query in enumerate(vectors):
        sim = np.asarray(DataStructs.BulkTanimotoSimilarity(query, train_vectors))
        if i in train_pos:
            sim[train_pos[i]] = -1
        knn.append(float(np.partition(sim, len(sim) - k)[-k:].mean()))
        density.append(int((sim >= DENSITY_THRESHOLD).sum()))
    return pd.DataFrame({"ad_knn_tanimoto_top5_mean": knn,
                         "ad_local_density_count_s040": density,
                         "ad_local_density_fraction_s040": np.asarray(density) / len(train_vectors)})


def main() -> int:
    parser = argparse.ArgumentParser(description="새 물성 지문 모델·적용가능도메인")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(Path(args.processed_dir) / args.dataset / "splits.csv", low_memory=False)
    task = frame["task_type"].iloc[0]
    matrix = make_morgan_matrix(frame["parent_smiles"])
    is_train = frame["split"].eq("train").to_numpy()
    y = frame["Y_final"].astype(int if task == "classification" else float).to_numpy()

    cv = {}
    for name in MODEL_NAMES:
        oof = np.full(len(frame), np.nan)
        for fold in sorted(frame.loc[is_train, "cv_fold"].unique()):
            val = is_train & frame["cv_fold"].eq(fold).to_numpy()
            model = make_model(name, task, SEEDS[0])
            model.fit(matrix[is_train & ~val], y[is_train & ~val])
            oof[val] = predict_values(model, matrix[val], task)
        cv[name] = (float(roc_auc_score(y[is_train], oof[is_train])) if task == "classification"
                    else float(root_mean_squared_error(y[is_train], oof[is_train])))
    primary = (max if task == "classification" else min)(MODEL_NAMES, key=lambda n: cv[n])

    out = frame[["row_uid", "dataset", "task_type", "split", "Y_final"]].copy()
    for name in MODEL_NAMES:
        stack = np.vstack([predict_values(make_model(name, task, s).fit(matrix[is_train], y[is_train]), matrix, task)
                           for s in SEEDS])
        out[f"pred_{name}"], out[f"std_{name}"] = stack.mean(0), stack.std(0, ddof=0)
    out["pred_fp_primary"], out["std_fp_primary"] = out[f"pred_{primary}"], out[f"std_{primary}"]
    out["fp_primary_model"] = primary
    signals = pd.concat([out, applicability_domain(matrix, is_train)], axis=1)

    target = Path(args.out_dir) / args.dataset
    target.mkdir(parents=True, exist_ok=True)
    out.to_csv(target / "fingerprint_predictions.csv", index=False)
    signals.to_csv(target / "role2_signals.csv", index=False)
    (target / "fingerprint_metrics.json").write_text(json.dumps(
        {"dataset": args.dataset, "task_type": task, "primary_model": primary, "train_only_cv": cv,
         "seeds": SEEDS, "chemberta": "not trained (fingerprint-only replication)"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.dataset}: 대표 {primary}  교차검증 {cv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
