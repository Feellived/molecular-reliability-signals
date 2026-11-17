#!/usr/bin/env python
"""변형 예측에서 A축·B축 분산 신호를 만든다 (연구계획서 4.2·5.8절).

피드백 반영: 기존 표준편차(std) 대신 중위수 절대 편차(MAD)를 계산하도록 수정됨.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, median_abs_deviation

AXES = ("A", "B1_tautomer", "B1_protonation", "B3_stereo")
B_AXES = ("B1_tautomer", "B1_protonation", "B3_stereo")

MODELS = {
    "fp_primary": "pred_fp_primary",
    "cb_regular": "pred_chemberta_regular",
    "cb_augmented": "pred_chemberta_augmented",
}
TARGET_SPLITS = ("meta", "test")

# 부동소수점 잔차를 0으로 본다. 
ZERO_TOLERANCE = 1e-12

def _spread(values: np.ndarray) -> tuple[float, float]:
    """표준편차 대신 MAD와 범위(range)를 반환하도록 수정"""
    if len(values) < 2:
        return 0.0, 0.0
    
    # 수정: np.std 대신 median_abs_deviation 사용
    mad = float(median_abs_deviation(values, scale='normal'))
    span = float(values.max() - values.min())
    return (
        0.0 if mad < ZERO_TOLERANCE else mad,
        0.0 if span < ZERO_TOLERANCE else span,
    )

def build_signals(dataset: str, splits_dir: Path, scores_dir: Path) -> pd.DataFrame:
    splits = pd.read_csv(splits_dir / dataset / "splits.csv", low_memory=False)
    splits = splits[splits["split"].isin(TARGET_SPLITS)].reset_index(drop=True)

    fp_dir = scores_dir / "fingerprint" / dataset
    cb_dir = scores_dir / "chemberta" / dataset
    origin = (
        pd.read_csv(fp_dir / "origin_predictions_refit.csv")
        .merge(
            pd.read_csv(cb_dir / "origin_predictions_chemberta.csv").drop(
                columns=["dataset", "split"]
            ),
            on="row_uid",
        )
        .set_index("row_uid")
    )
    variants = pd.read_csv(fp_dir / "variant_predictions_fp.csv").merge(
        pd.read_csv(cb_dir / "variant_predictions_chemberta.csv").drop(
            columns=["dataset", "axis", "split", "parent_row_uid"]
        ),
        on="variant_uid",
    )

    rows = []
    grouped = {key: frame for key, frame in variants.groupby("parent_row_uid")}
    for row_uid in splits["row_uid"]:
        record: dict = {"row_uid": row_uid}
        group = grouped.get(row_uid)
        for model_key, column in MODELS.items():
            parent_value = float(origin.at[row_uid, column])
            b_pool = [parent_value]
            for axis in AXES:
                if group is None:
                    axis_values = np.array([], dtype=float)
                else:
                    axis_values = group.loc[group["axis"] == axis, column].to_numpy(
                        dtype=float
                    )
                sample = np.append(axis_values, parent_value)
                mad, span = _spread(sample)
                # 열 이름을 std에서 mad로 모두 수정
                record[f"{model_key}__{axis}__mad"] = mad
                record[f"{model_key}__{axis}__range"] = span
                record[f"{model_key}__{axis}__n"] = int(len(axis_values))
                if axis in B_AXES:
                    b_pool.extend(axis_values.tolist())
            mad, span = _spread(np.asarray(b_pool, dtype=float))
            # 열 이름을 std에서 mad로 수정
            record[f"{model_key}__B_combined__mad"] = mad
            record[f"{model_key}__B_combined__range"] = span
            record[f"{model_key}__point"] = parent_value
        rows.append(record)

    signals = pd.DataFrame(rows)
    frame = splits[["row_uid", "dataset", "task_type", "split", "cv_fold", "Y_final"]].merge(
        signals, on="row_uid"
    )

    task_type = frame["task_type"].iloc[0]
    truth = pd.to_numeric(frame["Y_final"]).to_numpy(dtype=float)
    for model_key in MODELS:
        frame[f"{model_key}__abs_error"] = np.abs(
            truth - frame[f"{model_key}__point"].to_numpy(dtype=float)
        )
    frame["task_type"] = task_type

    # 백분위 정규화 대상 열도 __mad로 수정
    for column in [c for c in frame.columns if c.endswith("__mad")]:
        frame[f"{column}_pct"] = frame[column].rank(pct=True, method="average")
    return frame

def summarize(frame: pd.DataFrame) -> list[dict]:
    subset = frame[frame["split"].eq("test")]
    out = []
    for model_key in MODELS:
        error = subset[f"{model_key}__abs_error"].to_numpy(dtype=float)
        for axis in (*AXES, "B_combined"):
            # std 대신 mad 열 참조
            signal = subset[f"{model_key}__{axis}__mad"].to_numpy(dtype=float)
            if np.allclose(signal, signal[0]) or np.allclose(error, error[0]):
                rho = np.nan
            else:
                rho = float(spearmanr(signal, error).statistic)
            out.append(
                {
                    "dataset": subset["dataset"].iloc[0],
                    "task_type": subset["task_type"].iloc[0],
                    "model": model_key,
                    "axis": axis,
                    "n_test": len(subset),
                    "signal_mean": float(signal.mean()),
                    "signal_nonzero_rate": float((signal > 0).mean()),
                    "spearman_vs_abs_error": rho,
                }
            )
    return out

def main() -> int:
    parser = argparse.ArgumentParser(description="A축·B축 분산 신호 산출 (MAD 적용)")
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    scores_dir = Path(args.scores_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets or sorted(
        path.name
        for path in (scores_dir / "fingerprint").iterdir()
        if path.is_dir() and not path.name.startswith("_")
    )

    all_summaries = []
    for index, dataset in enumerate(datasets, 1):
        frame = build_signals(dataset, splits_dir, scores_dir)
        dataset_out = out_dir / dataset
        dataset_out.mkdir(parents=True, exist_ok=True)
        frame.to_csv(dataset_out / "ab_signals.csv", index=False)
        all_summaries.extend(summarize(frame))
        print(f"[{index}/{len(datasets)}] {dataset}: {len(frame):,}행", flush=True)

    summary = pd.DataFrame(all_summaries)
    summary_dir = out_dir / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_dir / "ab_signal_summary.csv", index=False)

    print()
    print("=== 축·모델별 신호 크기(MAD)와 오차 상관 (test, 22종 중앙값) ===")
    pivot = (
        summary.groupby(["model", "axis"])
        .agg(
            신호평균=("signal_mean", "median"),
            비영비율=("signal_nonzero_rate", "median"),
            상관중앙=("spearman_vs_abs_error", "median"),
            상관양수=("spearman_vs_abs_error", lambda s: int((s > 0).sum())),
        )
        .reset_index()
    )
    print(pivot.to_string(index=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
