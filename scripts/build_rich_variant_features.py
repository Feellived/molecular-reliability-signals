#!/usr/bin/env python
"""변형 예측에서 더 풍부한 통계를 뽑는다 (연구계획서 5.8절 확장).

지금까지는 변형 예측의 표준편차 하나만 신호로 썼다. 28만 건을 만들어놓고
흩어짐만 쓰는 것은 정보를 크게 버리는 셈이다. 같은 표본에서 다음을 추가로
뽑는다.

  std        흩어짐. 기존 신호
  max_dev    원본에서 가장 멀리 간 변형까지의 거리. 평균이 작아도 하나가
             크게 튀면 그 예측은 위험하다. 흩어짐이 놓치는 꼬리를 잡는다
  shift      변형 예측 평균에서 원본 예측을 뺀 값. 방향이 있는 이동으로,
             원본 표기가 체계적으로 치우쳐 있는지를 나타낸다
  rel_std    std를 그 모델의 예측 폭으로 나눈 값. 잘 맞는 모델일수록 예측이
             넓게 퍼지므로 원값만 보면 좋은 모델이 불안정해 보인다
  flip       분류에서 변형이 예측 라벨을 뒤집은 비율. 확률이 0.5 경계를
             넘나든다는 뜻이므로 연속적인 흩어짐보다 직접적인 위험 신호다

축별 적용 조건은 build_conditional_signals.py와 동일하게 담당1의 허용성
표를 따른다. 조건이 성립하는 축의 변형만 하나의 표본으로 합친다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = {"fp_primary": "pred_fp_primary", "cb_augmented": "pred_chemberta_augmented"}
DATASET_LEVEL_AXES = ("B1_tautomer", "B1_protonation")
MOLECULE_LEVEL_AXES = ("B3_stereo",)


def allowed_axes(dataset, allowance, decision) -> tuple[str, ...]:
    axes = [
        a for a in DATASET_LEVEL_AXES
        if allowance.at[dataset, a] == "허용" and decision.at[dataset, f"use_{a}"] == "사용"
    ]
    axes += [a for a in MOLECULE_LEVEL_AXES if decision.at[dataset, f"use_{a}"] == "사용"]
    return tuple(axes)


def statistics(values: np.ndarray, parent: float, scale: float, task: str) -> dict:
    sample = np.append(values, parent)
    std = float(np.std(sample, ddof=0)) if len(sample) > 1 else 0.0
    deviation = np.abs(values - parent) if len(values) else np.array([0.0])
    out = {
        "std": std,
        "max_dev": float(deviation.max()),
        "shift": float(values.mean() - parent) if len(values) else 0.0,
        "rel_std": float(std / scale) if scale > 1e-9 else 0.0,
    }
    if task == "classification":
        out["flip"] = (
            float(((values >= 0.5) != (parent >= 0.5)).mean()) if len(values) else 0.0
        )
    else:
        out["flip"] = 0.0
    return out


def build(dataset, scores_dir: Path, axes, task: str) -> pd.DataFrame:
    fp = pd.read_csv(scores_dir / "fingerprint" / dataset / "variant_predictions_fp.csv")
    cb = pd.read_csv(
        scores_dir / "chemberta" / dataset / "variant_predictions_chemberta.csv"
    ).drop(columns=["dataset", "axis", "split", "parent_row_uid"], errors="ignore")
    variants = fp.merge(cb, on="variant_uid")
    origin = pd.read_csv(
        scores_dir / "fingerprint" / dataset / "origin_predictions_refit.csv"
    ).merge(
        pd.read_csv(
            scores_dir / "chemberta" / dataset / "origin_predictions_chemberta.csv"
        ).drop(columns=["dataset", "split"], errors="ignore"),
        on="row_uid",
    ).set_index("row_uid")

    scales = {k: float(origin[c].std()) for k, c in MODELS.items()}
    stat_names = ("std", "max_dev", "shift", "rel_std", "flip")
    columns = ["row_uid"] + [f"rich__{m}__{g}__{s}"
                             for m in MODELS for g in ("A", "B") for s in stat_names]
    if not axes:
        frame = pd.DataFrame(columns=columns)
        frame["row_uid"] = origin.index
        return frame.fillna(0.0)

    groups = {"A": ("A",), "B": axes}
    rows = []
    for row_uid, group in variants.groupby("parent_row_uid"):
        record = {"row_uid": row_uid}
        for model_key, column in MODELS.items():
            parent = float(origin.at[row_uid, column])
            for group_name, group_axes in groups.items():
                values = group.loc[
                    group["axis"].isin(group_axes), column
                ].to_numpy(dtype=float)
                stats = statistics(values, parent, scales[model_key], task)
                for name, value in stats.items():
                    record[f"rich__{model_key}__{group_name}__{name}"] = value
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="변형 예측 확장 통계 산출")
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--reports-dir", required=True)
    parser.add_argument("--evaluation-dir", required=True)
    args = parser.parse_args()

    scores_dir = Path(args.scores_dir)
    evaluation_dir = Path(args.evaluation_dir)
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
    for dataset in datasets:
        target = evaluation_dir / dataset / "evaluation_signals.csv"
        frame = pd.read_csv(target)
        task = frame["task_type"].iloc[0]
        rich = build(dataset, scores_dir, allowed_axes(dataset, allowance, decision), task)

        frame = frame.drop(
            columns=[c for c in frame.columns if c.startswith("rich__")], errors="ignore"
        ).merge(rich, on="row_uid", how="left")
        for column in [c for c in frame.columns if c.startswith("rich__")]:
            frame[column] = frame[column].fillna(0.0)
            frame[f"{column}__pct"] = frame[column].rank(pct=True, method="average")
        frame.to_csv(target, index=False)
        print(f"  {dataset:32s} 확장 통계 {len([c for c in rich.columns if c != 'row_uid'])}개", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
