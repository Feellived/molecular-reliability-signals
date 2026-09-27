#!/usr/bin/env python
"""변형 예측에서 A축·B축 분산 신호를 만든다 (연구계획서 4.2·5.8절).

분자 하나마다 축마다 예측이 얼마나 흩어지는지를 재는 것이 전부다.
흩어짐의 기준점은 원본 분자의 예측이므로 원본을 항상 표본에 포함한다.

  A                표기만 바꾼 등가 SMILES에서의 흩어짐
  B1_tautomer      호변이성질체 상태에서의 흩어짐
  B1_protonation   양성자화 상태에서의 흩어짐
  B3_stereo        입체 표기 유무에서의 흩어짐
  B_combined       세 B축 변형을 한 표본으로 합친 흩어짐

모델 계열은 셋이다. 지문 대표 모델, ChemBERTa 정규, ChemBERTa 증강.
Morgan 지문은 SMILES 표기 순서와 무관하므로 지문 모델의 A축 분산은
구조적으로 0이다. 그대로 계산해 값이 실제로 0인지 확인한다.

물성마다 라벨 단위가 다르므로 원 분산값은 물성 간 비교가 되지 않는다.
그래서 각 신호를 물성 안에서 백분위로 정규화한 열을 함께 낸다(5.8절).

오차는 회귀가 |실측 − 예측|, 분류가 |실측 − 양성확률|이다. 분류에서
0/1 오분류 대신 확률 거리를 쓰는 이유는 신호와의 상관을 연속량으로
보기 위해서다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

AXES = ("A", "B1_tautomer", "B1_protonation", "B3_stereo")
B_AXES = ("B1_tautomer", "B1_protonation", "B3_stereo")

MODELS = {
    "fp_primary": "pred_fp_primary",
    "cb_regular": "pred_chemberta_regular",
    "cb_augmented": "pred_chemberta_augmented",
}
TARGET_SPLITS = ("meta", "test")


# 부동소수점 잔차를 0으로 본다. 지문 모델의 A축은 구조적으로 0인데
# 계산 과정에서 1e-17 수준의 값이 남아 비영으로 집계되는 것을 막는다.
ZERO_TOLERANCE = 1e-12


def _spread(values: np.ndarray) -> tuple[float, float]:
    """표준편차와 범위. 표본이 하나뿐이면 흩어짐은 0이다."""
    if len(values) < 2:
        return 0.0, 0.0
    std = float(np.std(values, ddof=0))
    span = float(values.max() - values.min())
    return (
        0.0 if std < ZERO_TOLERANCE else std,
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
                std, span = _spread(sample)
                record[f"{model_key}__{axis}__std"] = std
                record[f"{model_key}__{axis}__range"] = span
                record[f"{model_key}__{axis}__n"] = int(len(axis_values))
                if axis in B_AXES:
                    b_pool.extend(axis_values.tolist())
            std, span = _spread(np.asarray(b_pool, dtype=float))
            record[f"{model_key}__B_combined__std"] = std
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

    # 물성 안에서 백분위 정규화한다. 물성마다 라벨 단위가 달라 원값은 비교되지 않는다.
    for column in [c for c in frame.columns if c.endswith("__std")]:
        frame[f"{column}_pct"] = frame[column].rank(pct=True, method="average")
    return frame


def summarize(frame: pd.DataFrame) -> list[dict]:
    """신호와 실제 오차의 Spearman 상관. test 분할에서만 본다."""
    subset = frame[frame["split"].eq("test")]
    out = []
    for model_key in MODELS:
        error = subset[f"{model_key}__abs_error"].to_numpy(dtype=float)
        for axis in (*AXES, "B_combined"):
            signal = subset[f"{model_key}__{axis}__std"].to_numpy(dtype=float)
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
    parser = argparse.ArgumentParser(description="A축·B축 분산 신호 산출")
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--scores-dir", required=True, help="fingerprint·chemberta 상위")
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
    print("=== 축·모델별 신호 크기와 오차 상관 (test, 22종 중앙값) ===")
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
