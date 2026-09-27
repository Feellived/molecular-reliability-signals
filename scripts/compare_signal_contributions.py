#!/usr/bin/env python
"""모든 신호를 같은 잣대로 비교한다 (연구계획서 6.5절).

제안 신호의 기여를 판정할 때, 기존 신호가 같은 잣대를 통과하는지 함께 보지
않으면 판정 기준이 타당한지 알 수 없다. 이 스크립트는 기준 모형에서 신호를
하나씩 제거해 성능이 얼마나 나빠지는지 재고, 제안 축을 추가했을 때의 개선과
같은 표에 놓는다.

값이 양수면 그 신호를 뺐을 때 성능이 나빠졌다는 뜻, 곧 그 신호가 기여하고
있다는 뜻이다. 제안 축은 반대로 추가했을 때의 개선을 잰다.

6.5절은 p-값 중심 서술을 지양하고 효과 크기와 물성별 방향을 보고하도록
정한다. p는 참고로만 병기한다. 검정 단위가 물성 22개뿐이라 어지간한 효과가
아니면 유의가 나오지 않으며, 이는 신호가 무력하다는 뜻이 아니다.

컨포멀은 본래 오차 탐지 성능이 아니라 커버리지 보장으로 정당화되는 방법이다.
이 표에서의 낮은 기여가 그 방법의 유효성을 부정하지 않는다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score

REGRESSION_ERROR_QUANTILE = 0.80
RIDGE_ALPHA = 1.0

BASE = {
    "base__ad_knn__pct": "적용가능도메인 이웃 유사도",
    "base__ad_density__pct": "적용가능도메인 국소 밀도",
    "base__disagreement__pct": "모델 불일치",
    "base__conformal_cb__pct": "컨포멀 (ChemBERTa)",
    "base__conformal_fp__pct": "컨포멀 (지문)",
}
PROPOSED = {
    "표현 불안정성 A": ["axis__cb_augmented__A__pct"],
    "입력 상태 민감성 B (조건부)": [
        "cond_B__fp_primary__std__pct",
        "cond_B__cb_augmented__std__pct",
    ],
}


def normalized_aurc(risk: np.ndarray, error: np.ndarray) -> float:
    def aurc(score):
        ordered = error[np.argsort(score, kind="stable")]
        return float(np.mean(np.cumsum(ordered) / np.arange(1, len(ordered) + 1)))

    oracle, random = aurc(error), float(np.mean(error))
    return np.nan if random - oracle < 1e-12 else (aurc(risk) - oracle) / (random - oracle)


def evaluate(frame: pd.DataFrame, features: list[str]) -> tuple[float, float]:
    meta, test = frame[frame["split"].eq("meta")], frame[frame["split"].eq("test")]
    usable = [f for f in features if f in frame.columns and frame[f].nunique() > 1]
    if not usable:
        return np.nan, np.nan
    error = test["abs_error_fp"].to_numpy(dtype=float)
    truth = pd.to_numeric(test["Y_final"])
    if frame["task_type"].iloc[0] == "classification":
        labels = (
            (test["pred_fp_primary"].to_numpy(dtype=float) >= 0.5).astype(int)
            != truth.astype(int)
        ).astype(int)
    else:
        labels = (error >= np.quantile(error, REGRESSION_ERROR_QUANTILE)).astype(int)
    target = rankdata(meta["abs_error_fp"]) / len(meta)
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(meta[usable].to_numpy(dtype=float), target)
    score = model.predict(test[usable].to_numpy(dtype=float))
    auprc = (
        float(average_precision_score(labels, score))
        if labels.min() != labels.max() else np.nan
    )
    return auprc, normalized_aurc(score, error)


def main() -> int:
    parser = argparse.ArgumentParser(description="신호 기여도 동일 잣대 비교")
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    evaluation_dir = Path(args.evaluation_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = sorted(
        p.name for p in evaluation_dir.iterdir()
        if p.is_dir() and not p.name.startswith("_")
    )
    base_features = list(BASE)

    results: dict[str, list] = {}
    for dataset in datasets:
        frame = pd.read_csv(evaluation_dir / dataset / "evaluation_signals.csv")
        results.setdefault("__full__", []).append(evaluate(frame, base_features))
        for column in base_features:
            results.setdefault(column, []).append(
                evaluate(frame, [f for f in base_features if f != column])
            )
        for label, extra in PROPOSED.items():
            results.setdefault(label, []).append(evaluate(frame, base_features + extra))
        results.setdefault("A + B (조건부) 동시", []).append(
            evaluate(frame, base_features + [f for e in PROPOSED.values() for f in e])
        )

    full = np.array(results.pop("__full__"), dtype=float)
    rows = []
    for key, values in results.items():
        arr = np.array(values, dtype=float)
        if key in BASE:
            # 제거했을 때의 손실. 양수면 그 신호가 기여하고 있다는 뜻이다.
            delta = full - arr
            delta[:, 1] = -delta[:, 1]  # AURC는 낮을수록 좋으므로 방향을 맞춘다
            kind, name = "기존 기준선", BASE[key]
        else:
            delta = arr - full
            delta[:, 1] = -delta[:, 1]
            kind, name = "제안 축", key
        rows.append(
            {
                "구분": kind,
                "신호": name,
                "AUPRC_기여": float(np.nanmean(delta[:, 0])),
                "AUPRC_개선물성": int(np.nansum(delta[:, 0] > 0)),
                "AUPRC_p": float(wilcoxon(delta[~np.isnan(delta[:, 0]), 0])[1]),
                "AURC_기여": float(np.nanmean(delta[:, 1])),
                "AURC_개선물성": int(np.nansum(delta[:, 1] > 0)),
                "AURC_p": float(wilcoxon(delta[~np.isnan(delta[:, 1]), 1])[1]),
            }
        )

    table = pd.DataFrame(rows).sort_values(["구분", "AUPRC_기여"], ascending=[True, False])
    table.to_csv(out_dir / "signal_contribution_comparison.csv", index=False)

    print("=== 신호 기여도 동일 잣대 비교 (기준선은 제거 손실, 제안 축은 추가 이득) ===")
    print(f"{'구분':11s}{'신호':28s}{'AUPRC':>9s}{'물성':>7s}{'p':>7s}{'AURC':>9s}{'물성':>7s}{'p':>7s}")
    for _, r in table.iterrows():
        print(
            f"{r['구분']:11s}{r['신호']:28s}{r['AUPRC_기여']:+9.4f}{r['AUPRC_개선물성']:5d}/22"
            f"{r['AUPRC_p']:7.3f}{r['AURC_기여']:+9.4f}{r['AURC_개선물성']:5d}/22{r['AURC_p']:7.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
