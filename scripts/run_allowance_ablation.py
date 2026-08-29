#!/usr/bin/env python
"""변형 허용성을 존중한 제거 실험과 축별 분해 (연구계획서 5.2절 재정렬).

계획서 5.2절은 물성별로 허용되는 변형 축을 사전에 표로 확정한 뒤 실험을
시작하도록 정한다. 그러나 실제 축 판정(07_axis_decision.csv)은 표본 수 기준
게이트만 적용해, 허용성 표에서 주의로 분류된 물성에서도 변형을 생성했다.

  B1 호변이성질체   허용 21종, 주의 1종 → 22종 전부 생성 (1종 불일치)
  B1 양성자화       허용 15종, 주의 7종 → 22종 전부 생성 (7종 불일치)
  B3 입체 표기      허용 22종 전부 → 19종 생성 (3종은 표본 부족, 정당한 제외)

불일치는 사실상 B1 양성자화에 몰려 있다. 이 스크립트는 물성마다 허용 판정을
받은 축만 신호로 넣는 구성을 만들어 현행 구성과 비교한다. 축별 기여도 함께 본다.

이 분석의 성격. 축 제한 자체는 5.2절이 지시한 바이므로 사후에 만들어낸 규칙이
아니다. 다만 두 절의 불일치를 전체 결과가 음성인 것을 확인한 뒤에 발견했으므로,
사전 지정 분석을 대체하지 않고 병기한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score

RIDGE_ALPHA = 1.0
REGRESSION_ERROR_QUANTILE = 0.80
MODELS = ("fp_primary", "cb_augmented")
B_AXES = ("B1_tautomer", "B1_protonation", "B3_stereo")

BASE_FEATURES = [
    "base__ad_knn__pct",
    "base__ad_density__pct",
    "base__disagreement__pct",
    "base__conformal_cb__pct",
    "base__conformal_fp__pct",
]


def axis_features(axes: tuple[str, ...]) -> list[str]:
    return [f"axis__{m}__{a}__pct" for m in MODELS for a in axes]


def error_labels(frame: pd.DataFrame) -> np.ndarray:
    task_type = frame["task_type"].iloc[0]
    truth = pd.to_numeric(frame["Y_final"]).to_numpy(dtype=float)
    if task_type == "classification":
        predicted = (frame["pred_fp_primary"].to_numpy(dtype=float) >= 0.5).astype(int)
        return (predicted != truth.astype(int)).astype(int)
    error = frame["abs_error_fp"].to_numpy(dtype=float)
    return (error >= np.quantile(error, REGRESSION_ERROR_QUANTILE)).astype(int)


def normalized_aurc(risk: np.ndarray, error: np.ndarray) -> float:
    def aurc(score):
        ordered = error[np.argsort(score, kind="stable")]
        return float(np.mean(np.cumsum(ordered) / np.arange(1, len(ordered) + 1)))

    oracle, random = aurc(error), float(np.mean(error))
    if random - oracle < 1e-12:
        return float("nan")
    return (aurc(risk) - oracle) / (random - oracle)


def score_config(frame: pd.DataFrame, features: list[str]) -> np.ndarray | None:
    meta = frame[frame["split"].eq("meta")]
    test = frame[frame["split"].eq("test")]
    usable = [f for f in features if f in frame.columns and frame[f].nunique() > 1]
    if not usable:
        return None
    target = rankdata(meta["abs_error_fp"].to_numpy(dtype=float)) / len(meta)
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(meta[usable].to_numpy(dtype=float), target)
    return model.predict(test[usable].to_numpy(dtype=float))


def main() -> int:
    parser = argparse.ArgumentParser(description="허용성 존중 제거 실험")
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--reports-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    evaluation_dir = Path(args.evaluation_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    allowance = pd.read_csv(
        Path(args.reports_dir) / "06_transformation_allowance_revised.csv"
    ).set_index("dataset")
    decision = pd.read_csv(
        Path(args.reports_dir) / "07_axis_decision.csv"
    ).set_index("dataset")

    datasets = sorted(
        p.name for p in evaluation_dir.iterdir()
        if p.is_dir() and not p.name.startswith("_")
    )

    records = []
    for dataset in datasets:
        frame = pd.read_csv(evaluation_dir / dataset / "evaluation_signals.csv")
        test = frame[frame["split"].eq("test")]
        error = test["abs_error_fp"].to_numpy(dtype=float)
        labels = error_labels(test)

        allowed = tuple(
            axis for axis in B_AXES
            if allowance.at[dataset, axis] == "허용"
            and decision.at[dataset, f"use_{axis}"] == "사용"
        )
        generated = tuple(
            axis for axis in B_AXES if decision.at[dataset, f"use_{axis}"] == "사용"
        )

        configs = {
            "기준": BASE_FEATURES,
            "기준+B(생성 전체)": BASE_FEATURES + axis_features(generated),
            "기준+B(허용축만)": BASE_FEATURES + axis_features(allowed),
            "기준+B1": BASE_FEATURES + axis_features(
                tuple(a for a in allowed if a.startswith("B1"))
            ),
            "기준+B3": BASE_FEATURES + axis_features(
                tuple(a for a in allowed if a == "B3_stereo")
            ),
        }
        record = {
            "dataset": dataset,
            "task_type": frame["task_type"].iloc[0],
            "n_allowed_axes": len(allowed),
            "allowed_axes": "+".join(a.replace("B1_", "").replace("B3_", "") for a in allowed),
            "excluded_axes": "+".join(set(generated) - set(allowed)) or "없음",
        }
        for name, features in configs.items():
            score = score_config(frame, features)
            if score is None:
                record[f"aurc__{name}"] = np.nan
                record[f"auprc__{name}"] = np.nan
                continue
            record[f"aurc__{name}"] = normalized_aurc(score, error)
            record[f"auprc__{name}"] = (
                float(average_precision_score(labels, score))
                if labels.min() != labels.max() else np.nan
            )
        records.append(record)

    table = pd.DataFrame(records)
    table.to_csv(out_dir / "allowance_ablation_by_dataset.csv", index=False)

    print("=== 축 허용성 반영 결과 ===")
    print(f"  허용축 3개 물성 {(table.n_allowed_axes == 3).sum()}종, "
          f"2개 {(table.n_allowed_axes == 2).sum()}종, "
          f"1개 {(table.n_allowed_axes == 1).sum()}종")
    print()
    for metric, better in [("aurc", "낮을수록 좋음"), ("auprc", "높을수록 좋음")]:
        print(f"=== {metric.upper()} ({better}) ===")
        for name in ["기준", "기준+B(생성 전체)", "기준+B(허용축만)", "기준+B1", "기준+B3"]:
            column = table[f"{metric}__{name}"].dropna()
            print(f"  {name:18s} 평균 {column.mean():.4f}   중앙 {column.median():.4f}")
        base = table[f"{metric}__기준"]
        for name in ["기준+B(생성 전체)", "기준+B(허용축만)", "기준+B1", "기준+B3"]:
            delta = (table[f"{metric}__{name}"] - base).dropna()
            sign = -1 if metric == "aurc" else 1
            improved = int((delta * sign > 0).sum())
            p = wilcoxon(delta)[1] if len(delta) >= 6 else np.nan
            print(f"    {name:18s} 변화 {delta.mean():+.4f}  개선 {improved}/{len(delta)}종  p={p:.3f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
