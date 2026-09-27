#!/usr/bin/env python
"""판정 등급의 경계를 실제 오차율로 정한다.

데모는 축 백분위가 0.85를 넘으면 주의, 0.65를 넘으면 보통으로 판정해왔다.
두 값에 근거가 없었다. 화면 한가운데에 놓이는 문장이 임의의 숫자에 기대고
있으면 곤란하다.

여기서는 시험 분할에서 실제로 크게 틀린 비율을 백분위 구간별로 재고, 그
비율이 전체 대비 몇 배인지로 경계를 고른다. 그러면 판정에 근거가 생긴다.
주의 구간의 분자는 실제로 몇 배 더 자주 틀렸다고 말할 수 있게 된다.

미달의 정의는 오차 탐지 라벨과 같다. 회귀는 상위 20퍼센트 오차, 분류는
오분류다. 백분위는 데모와 같은 방식으로 보정 분할 분포에 고정해 매긴다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

AXIS_COLUMNS = ("axis__cb_augmented__A", "cond_B__fp_primary__std")
REFERENCE_SPLIT = "meta"
# 등급 경계 후보. 아래쪽 꼬리가 위쪽보다 또렷해 안정 구간을 넉넉히 잡는다.
CANDIDATES = ((0.30, 0.80), (0.35, 0.85), (0.40, 0.85), (0.35, 0.90), (0.40, 0.90))


def calibration_percentile(values, reference) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=float))
    if len(reference) == 0:
        return np.zeros(len(values))
    return np.searchsorted(reference, np.asarray(values, float), side="right") / len(reference)


def shortfall(frame: pd.DataFrame) -> np.ndarray:
    error = frame["abs_error_fp"].to_numpy(float)
    if frame["task_type"].iloc[0] == "classification":
        predicted = (frame["pred_fp_primary"].to_numpy(float) >= 0.5).astype(int)
        return (predicted != pd.to_numeric(frame["Y_final"]).astype(int)).to_numpy().astype(int)
    return (error >= np.quantile(error, 0.8)).astype(int)


def main() -> int:
    parser = argparse.ArgumentParser(description="판정 경계 보정")
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    evaluation_dir = Path(args.evaluation_dir)
    rows = []
    for path in sorted(evaluation_dir.iterdir()):
        if not path.is_dir() or path.name.startswith("_"):
            continue
        frame = pd.read_csv(path / "evaluation_signals.csv")
        meta = frame["split"].eq(REFERENCE_SPLIT).to_numpy()
        test = frame["split"].eq("test").to_numpy()

        # 축마다 보정 분포에 고정해 백분위를 매기고, 그중 가장 높은 값을 쓴다.
        # 데모의 판정이 어느 한 축이라도 높으면 주의로 가는 구조라 그와 맞춘다.
        per_axis = []
        for column in AXIS_COLUMNS:
            if column not in frame or frame.loc[meta, column].nunique() <= 1:
                continue
            values = frame[column].to_numpy(float)
            per_axis.append(calibration_percentile(values, values[meta]))
        if not per_axis:
            continue
        highest = np.max(np.column_stack(per_axis), axis=1)

        subset = frame[test]
        rows.append(pd.DataFrame({
            "dataset": path.name,
            "percentile": highest[test],
            "shortfall": shortfall(subset),
        }))

    pooled = pd.concat(rows, ignore_index=True)
    base = pooled["shortfall"].mean()

    edges = np.arange(0, 1.0001, 0.1)
    pooled["bin"] = pd.cut(pooled["percentile"], edges, include_lowest=True)
    table = pooled.groupby("bin", observed=True).agg(
        분자=("shortfall", "size"), 미달률=("shortfall", "mean"))
    table["배수"] = table["미달률"] / base

    print(f"물성 {pooled.dataset.nunique()}종, 시험 분자 {len(pooled)}개, "
          f"전체 미달률 {base:.4f}\n")
    print("축 백분위 구간별 실제 미달률")
    print(table.round(4).to_string())

    # 세 등급으로 갈라 각 구간의 실제 미달률을 잰다. 경계를 고르는 기준은
    # 등급 사이의 배수 차이가 크면서 어느 등급도 지나치게 비지 않는 것이다.
    def evaluate(low: float, high: float) -> dict:
        bands = {"안정": pooled["percentile"] < low,
                 "보통": (pooled["percentile"] >= low) & (pooled["percentile"] < high),
                 "주의": pooled["percentile"] >= high}
        record = {"low": low, "high": high, "bands": {}}
        for name, mask in bands.items():
            rate = float(pooled.loc[mask, "shortfall"].mean()) if mask.any() else float("nan")
            record["bands"][name] = {"n": int(mask.sum()), "share": float(mask.mean()),
                                     "shortfall": rate, "lift": rate / base}
        record["spread"] = (record["bands"]["주의"]["lift"]
                            / record["bands"]["안정"]["lift"])
        return record

    options = [evaluate(low, high) for low, high in CANDIDATES]
    print("\n등급 경계 후보")
    print(f"{'경계':14s}{'안정':>22s}{'보통':>22s}{'주의':>22s}{'주의/안정':>10s}")
    for option in options:
        cells = "".join(
            f"{option['bands'][n]['lift']:8.2f}배 {option['bands'][n]['share'] * 100:5.1f}%"
            for n in ("안정", "보통", "주의"))
        print(f"  {option['low']:.2f} / {option['high']:.2f}  {cells}{option['spread']:10.2f}")

    best = max(options, key=lambda o: o["spread"])
    chosen = {"MODERATE": best["low"], "HIGH": best["high"], "bands": best["bands"],
              "spread": best["spread"]}
    print(f"\n고른 경계  보통 {best['low']:.2f} 이상, 주의 {best['high']:.2f} 이상")
    for name in ("안정", "보통", "주의"):
        band = best["bands"][name]
        print(f"  {name}  분자 {band['n']:5d} ({band['share'] * 100:4.1f}%)  "
              f"미달률 {band['shortfall']:.4f}  전체의 {band['lift']:.2f}배")
    print(f"  주의 구간은 안정 구간보다 {best['spread']:.2f}배 자주 크게 틀린다")
    curve = options

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "base_shortfall_rate": float(base),
        "n_molecules": int(len(pooled)),
        "n_datasets": int(pooled.dataset.nunique()),
        "candidates": curve,
        "chosen": chosen,
        "definition": "미달은 오차 탐지 라벨과 같다. 회귀는 상위 20퍼센트 오차, 분류는 오분류.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
