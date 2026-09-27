#!/usr/bin/env python
"""오차 탐지 AUPRC 기반 1차 지표 산출 (연구계획서 6.3·6.5절).

6.3절은 오차 탐지 AUPRC를 1차 지표로 지정한다. 큰 오차 사례가 소수이므로
클래스 불균형에 강건한 지표가 적절하기 때문이다. AUROC는 보조로 함께 본다.

6.5절은 p-값 중심의 서술을 지양하고 물성별 효과 크기와 신뢰구간을 보고하도록
정한다. 신뢰구간은 골격 군집 단위 부트스트랩으로 산출한다. 같은 골격에 속한
분자들은 독립이 아니므로 분자 단위로 재표집하면 구간이 실제보다 좁아진다.

오차 라벨 정의

  분류   실제 오분류. 양성 확률이 0.5를 기준으로 정답과 반대편에 있는 경우다.
  회귀   물성 안에서 절대 오차 상위 20퍼센트.

분류에서 오분류를 쓰는 이유는 그것이 현장에서 실제로 문제가 되는 사건이기
때문이다. 회귀는 자연스러운 경계가 없어 상위 20퍼센트를 쓴다. 담당1의 라벨
잡음 표에 반복 측정이 거의 없어 잡음 기반 임계값은 적용할 수 없었다.

신호 결합 규칙은 제거 실험과 동일하게 meta에서 학습하고 test에서 평가한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score

RIDGE_ALPHA = 1.0
REGRESSION_ERROR_QUANTILE = 0.80
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260829

BASE_FEATURES = [
    "base__ad_knn__pct",
    "base__ad_density__pct",
    "base__disagreement__pct",
    "base__conformal_cb__pct",
    "base__conformal_fp__pct",
]
B_FEATURES = [
    "axis__fp_primary__B_combined__pct",
    "axis__cb_augmented__B_combined__pct",
]
A_FEATURES = ["axis__cb_augmented__A__pct"]

CONFIGURATIONS = {
    "기준": BASE_FEATURES,
    "기준+B": BASE_FEATURES + B_FEATURES,
    "기준+A": BASE_FEATURES + A_FEATURES,
    "기준+A+B": BASE_FEATURES + A_FEATURES + B_FEATURES,
}


def error_labels(frame: pd.DataFrame) -> np.ndarray:
    """큰 오차 사건을 1로 표시한다."""
    task_type = frame["task_type"].iloc[0]
    truth = pd.to_numeric(frame["Y_final"]).to_numpy(dtype=float)
    if task_type == "classification":
        predicted = (frame["pred_fp_primary"].to_numpy(dtype=float) >= 0.5).astype(int)
        return (predicted != truth.astype(int)).astype(int)
    error = frame["abs_error_fp"].to_numpy(dtype=float)
    return (error >= np.quantile(error, REGRESSION_ERROR_QUANTILE)).astype(int)


def _fit_scores(frame: pd.DataFrame, features: list[str]) -> np.ndarray | None:
    meta = frame[frame["split"].eq("meta")]
    test = frame[frame["split"].eq("test")]
    usable = [f for f in features if f in frame.columns and frame[f].nunique() > 1]
    if not usable:
        return None
    error_meta = meta["abs_error_fp"].to_numpy(dtype=float)
    target = rankdata(error_meta) / len(error_meta)
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(meta[usable].to_numpy(dtype=float), target)
    return model.predict(test[usable].to_numpy(dtype=float))


def _cluster_bootstrap_delta(
    labels: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    clusters: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """골격 군집을 단위로 재표집해 AUPRC 차이의 분포를 만든다."""
    unique = np.unique(clusters)
    index_by_cluster = {c: np.flatnonzero(clusters == c) for c in unique}
    deltas = []
    for _ in range(N_BOOTSTRAP):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([index_by_cluster[c] for c in drawn])
        y = labels[index]
        if y.min() == y.max():
            continue
        deltas.append(
            average_precision_score(y, score_b[index])
            - average_precision_score(y, score_a[index])
        )
    return np.asarray(deltas)


def evaluate_dataset(frame: pd.DataFrame, scaffolds: pd.DataFrame) -> dict:
    test = frame[frame["split"].eq("test")].merge(scaffolds, on="row_uid", how="left")
    labels = error_labels(frame[frame["split"].eq("test")])
    clusters = test["scaffold_group"].fillna("__none__").to_numpy()

    record = {
        "dataset": frame["dataset"].iloc[0],
        "task_type": frame["task_type"].iloc[0],
        "n_test": len(test),
        "n_error": int(labels.sum()),
        "error_rate": float(labels.mean()),
        "n_scaffold_group": int(len(np.unique(clusters))),
    }
    if labels.min() == labels.max():
        return record

    scores = {}
    for name, features in CONFIGURATIONS.items():
        score = _fit_scores(frame, features)
        if score is None:
            continue
        scores[name] = score
        record[f"auprc__{name}"] = float(average_precision_score(labels, score))
        record[f"auroc__{name}"] = float(roc_auc_score(labels, score))

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for name in ("기준+B", "기준+A", "기준+A+B"):
        if name not in scores or "기준" not in scores:
            continue
        deltas = _cluster_bootstrap_delta(
            labels, scores["기준"], scores[name], clusters, rng
        )
        if len(deltas) < 100:
            continue
        record[f"delta_auprc__{name}"] = float(
            record[f"auprc__{name}"] - record["auprc__기준"]
        )
        record[f"ci_low__{name}"] = float(np.percentile(deltas, 2.5))
        record[f"ci_high__{name}"] = float(np.percentile(deltas, 97.5))
        record[f"p_positive__{name}"] = float((deltas > 0).mean())
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="오차 탐지 AUPRC 1차 지표")
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    evaluation_dir = Path(args.evaluation_dir)
    splits_dir = Path(args.splits_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = sorted(
        path.name
        for path in evaluation_dir.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    )

    records = []
    for index, dataset in enumerate(datasets, 1):
        frame = pd.read_csv(evaluation_dir / dataset / "evaluation_signals.csv")
        scaffolds = pd.read_csv(
            splits_dir / dataset / "splits.csv", low_memory=False,
            usecols=["row_uid", "scaffold_group"],
        )
        record = evaluate_dataset(frame, scaffolds)
        records.append(record)
        delta = record.get("delta_auprc__기준+B", float("nan"))
        low = record.get("ci_low__기준+B", float("nan"))
        high = record.get("ci_high__기준+B", float("nan"))
        print(
            f"[{index}/{len(datasets)}] {dataset}: 기준 AUPRC "
            f"{record.get('auprc__기준', float('nan')):.3f} → +B {delta:+.4f} "
            f"[{low:+.3f}, {high:+.3f}]",
            flush=True,
        )

    table = pd.DataFrame(records)
    table.to_csv(out_dir / "error_detection_by_dataset.csv", index=False)

    print()
    print("=== 오차 탐지 AUPRC (1차 지표, test) ===")
    for name in CONFIGURATIONS:
        column = table.get(f"auprc__{name}")
        if column is None:
            continue
        print(f"  {name:10s} 평균 {column.mean():.4f}   중앙 {column.median():.4f}")

    print()
    print("=== 기준 대비 효과 크기와 95퍼센트 신뢰구간 (골격 군집 부트스트랩) ===")
    for name in ("기준+B", "기준+A", "기준+A+B"):
        delta = table[f"delta_auprc__{name}"].dropna()
        low = table[f"ci_low__{name}"]
        high = table[f"ci_high__{name}"]
        positive = int(((low > 0)).sum())
        negative = int(((high < 0)).sum())
        print(
            f"  {name:8s} 평균 {delta.mean():+.4f}  개선 {int((delta > 0).sum())}/{len(delta)}종  "
            f"구간이 0 초과 {positive}종, 0 미만 {negative}종"
        )

    summary = {
        "primary_metric": "error detection AUPRC (계획서 6.3절)",
        "error_label": {
            "classification": "실제 오분류",
            "regression": f"절대 오차 상위 {int((1 - REGRESSION_ERROR_QUANTILE) * 100)}퍼센트",
        },
        "interval": f"골격 군집 부트스트랩 {N_BOOTSTRAP}회, 95퍼센트",
        "mean_auprc": {
            name: float(table[f"auprc__{name}"].mean())
            for name in CONFIGURATIONS
            if f"auprc__{name}" in table
        },
    }
    (out_dir / "error_detection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
