#!/usr/bin/env python
"""축별 적용 조건을 반영한 신호 재산출 (연구계획서 5.2절 재정렬).

축마다 화학적 성격이 다르므로 적용 조건도 달라야 한다. 지금까지는 세 B축을
조건 구분 없이 하나로 뭉쳐 신호를 만들었고, 그 결과 조건이 성립하지 않는
물성의 변형이 잡음으로 섞여 들어갔다.

축별 조건과 그 근거

  A 표기            화학적 조건이 없다. 분자가 완전히 동일하고 표기만 다르다.
                    SMILES를 직접 읽는 모델에서만 0이 아닌 값이 나온다.
                    적용 층위는 없음. 22종 전부.

  B1 호변이성질체   수소 위치가 용액에서 평형을 이룬다는 전제가 필요하다.
                    고체 상태나 조건이 이질적인 통합 자료에서는 어느 형태가
                    측정된 것인지 확정할 수 없다. 물성 단위 조건.

  B1 양성자화       이온화 상태는 pH에 따라 결정되므로 측정 pH가 정의돼야
                    변형이 같은 약의 다른 상태로 해석된다. 여러 출처를 합쳐
                    pH가 보존되지 않은 자료에서는 잡음이 된다. 물성 단위 조건.

  B3 입체 표기      입체중심이 있는 분자만 입체 정보를 잃을 수 있다. 물성이
                    아니라 분자의 성질이므로 물성 단위로 걸러낼 대상이 아니다.
                    입체가 없는 분자는 변형 수가 0이라 자동으로 배제된다.

물성 단위 조건은 담당1의 06_transformation_allowance_revised.csv에서 읽는다.
허용으로 판정된 축만 해당 물성에서 사용한다.

변수 개수를 늘리지 않는 것이 중요하다. 축별로 변수를 따로 만들면 기준 5개에
축 6개가 붙어 meta가 작은 물성에서 규칙이 과적합된다. 그래서 허용된 축의
변형을 하나의 표본으로 합쳐 모델당 하나의 분산 신호를 만든다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = {"fp_primary": "pred_fp_primary", "cb_augmented": "pred_chemberta_augmented"}
DATASET_LEVEL_AXES = ("B1_tautomer", "B1_protonation")
MOLECULE_LEVEL_AXES = ("B3_stereo",)
ZERO_TOLERANCE = 1e-12


def allowed_axes(dataset: str, allowance: pd.DataFrame, decision: pd.DataFrame) -> tuple[str, ...]:
    """이 물성에서 조건이 성립하는 B축 목록."""
    axes = []
    for axis in DATASET_LEVEL_AXES:
        if (
            allowance.at[dataset, axis] == "허용"
            and decision.at[dataset, f"use_{axis}"] == "사용"
        ):
            axes.append(axis)
    # 분자 단위 조건인 축은 물성 단위로 거르지 않는다. 생성만 되었으면 포함한다.
    for axis in MOLECULE_LEVEL_AXES:
        if decision.at[dataset, f"use_{axis}"] == "사용":
            axes.append(axis)
    return tuple(axes)


def build(dataset: str, variants_dir: Path, scores_dir: Path, axes: tuple[str, ...]) -> pd.DataFrame:
    fp = pd.read_csv(scores_dir / "fingerprint" / dataset / "variant_predictions_fp.csv")
    cb = pd.read_csv(
        scores_dir / "chemberta" / dataset / "variant_predictions_chemberta.csv"
    ).drop(columns=["dataset", "axis", "split", "parent_row_uid"], errors="ignore")
    variants = fp.merge(cb, on="variant_uid")

    origin_fp = pd.read_csv(
        scores_dir / "fingerprint" / dataset / "origin_predictions_refit.csv"
    ).set_index("row_uid")
    origin_cb = pd.read_csv(
        scores_dir / "chemberta" / dataset / "origin_predictions_chemberta.csv"
    ).set_index("row_uid")

    columns = ["row_uid", "n_conditional_variants"] + [
        f"cond_B__{key}__std" for key in MODELS
    ]
    if not axes:
        # 조건이 성립하는 축이 하나도 없는 물성이 있다. 신호를 0으로 둔다.
        return pd.DataFrame(columns=columns)

    selected = variants[variants["axis"].isin(axes)]
    rows = []
    for row_uid, group in selected.groupby("parent_row_uid"):
        record = {"row_uid": row_uid, "n_conditional_variants": len(group)}
        for key, column in MODELS.items():
            parent = float(
                origin_fp.at[row_uid, column]
                if key == "fp_primary"
                else origin_cb.at[row_uid, column]
            )
            sample = np.append(group[column].to_numpy(dtype=float), parent)
            std = float(np.std(sample, ddof=0)) if len(sample) > 1 else 0.0
            record[f"cond_B__{key}__std"] = 0.0 if std < ZERO_TOLERANCE else std
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="축별 조건 반영 신호 산출")
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--variants-dir", required=True)
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

    manifest = []
    for dataset in datasets:
        axes = allowed_axes(dataset, allowance, decision)
        signals = build(dataset, Path(args.variants_dir), scores_dir, axes)

        target = evaluation_dir / dataset / "evaluation_signals.csv"
        frame = pd.read_csv(target)
        frame = frame.drop(
            columns=[c for c in frame.columns if c.startswith("cond_B__")], errors="ignore"
        ).merge(signals, on="row_uid", how="left")
        for column in [c for c in frame.columns if c.startswith("cond_B__")]:
            frame[column] = frame[column].fillna(0.0)
            frame[f"{column}__pct"] = frame[column].rank(pct=True, method="average")
        frame.to_csv(target, index=False)

        manifest.append(
            {
                "dataset": dataset,
                "적용축": "+".join(a.replace("B1_", "").replace("B3_", "") for a in axes),
                "제외축": "+".join(
                    a.replace("B1_", "")
                    for a in DATASET_LEVEL_AXES
                    if a not in axes and decision.at[dataset, f"use_{a}"] == "사용"
                ) or "없음",
                "n_axes": len(axes),
                "평균_변형수": float(signals["n_conditional_variants"].mean()),
            }
        )
        print(
            f"  {dataset:32s} 적용 {manifest[-1]['적용축']:34s} 제외 {manifest[-1]['제외축']}",
            flush=True,
        )

    table = pd.DataFrame(manifest)
    out = scores_dir / "conditional_signals"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "axis_condition_manifest.csv", index=False)
    (out / "axis_conditions.json").write_text(
        json.dumps(
            {
                "A": {"조건": "없음. 분자가 동일하고 표기만 다르다", "층위": "없음", "적용": "22/22"},
                "B1_tautomer": {"조건": "용액 상태 평형", "층위": "물성",
                                "적용": f"{int((table.적용축.str.contains('tautomer')).sum())}/22"},
                "B1_protonation": {"조건": "측정 pH가 정의됨", "층위": "물성",
                                   "적용": f"{int((table.적용축.str.contains('protonation')).sum())}/22"},
                "B3_stereo": {"조건": "분자가 입체중심을 가짐", "층위": "분자",
                              "적용": f"{int((table.적용축.str.contains('stereo')).sum())}/22 물성, 분자의 약 33퍼센트"},
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print(table.groupby("적용축").size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
