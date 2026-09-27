#!/usr/bin/env python
"""기준선 신호와 A·B 신호를 하나의 평가용 표로 합친다 (연구계획서 5.8절).

제거 실험은 기준 모형과 그 위에 축을 얹은 모형을 비교한다. 그러려면 두 무리의
신호가 같은 행 위에 정렬되어 있어야 한다. 이 스크립트가 그 표를 만든다.

기준 모형은 다섯 신호로 구성한다.

  ad_knn            적용가능도메인 최근접 이웃 유사도 (담당2, 지문 공간)
  ad_density        적용가능도메인 국소 밀도 (담당2, 지문 공간)
  disagreement      지문 모델과 ChemBERTa의 예측 불일치 (담당2)
  conformal_cb      ChemBERTa 기반 적응형 컨포멀 (담당2)
  conformal_fp      지문 기반 적응형 컨포멀 (담당4 추가, 5.6절)

conformal_fp를 추가한 이유는 나머지 컨포멀이 점추정과 척도를 모두 ChemBERTa에서
가져와 기준 모형 전체가 언어 모델 하나에 걸려 있었기 때문이다. 적용가능도메인을
언어 모델에서 분리한 5.6절의 취지에 맞추고, 분류에서 더 강한 모델 위에
기준선이 서도록 한다.

신호의 방향은 모두 "값이 클수록 위험"으로 통일한다. 적용가능도메인 두 신호는
값이 클수록 안전하므로 부호를 뒤집는다. 뒤집지 않으면 결합 단계에서 부호가
섞여 해석이 불가능해진다.

물성마다 라벨 단위와 신호의 척도가 다르므로 모든 신호를 물성 안에서 백분위로
정규화한 열을 함께 낸다. 제거 실험은 이 정규화 열을 쓴다.

대상은 meta와 test 분할이다. A·B 신호가 그 두 분할에만 존재한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TARGET_SPLITS = ("meta", "test")

# 값이 클수록 위험한 방향으로 통일한다. 부호가 -1인 신호는 뒤집는다.
BASELINE_SIGNALS = {
    "ad_knn": -1,
    "ad_density": -1,
    "disagreement": +1,
    "conformal_cb": +1,
    "conformal_fp": +1,
}

AXIS_SIGNALS = [
    "A",
    "B1_tautomer",
    "B1_protonation",
    "B3_stereo",
    "B_combined",
]
AXIS_MODELS = ["fp_primary", "cb_augmented"]


def build(dataset: str, jiye_dir: Path, scores_dir: Path) -> pd.DataFrame:
    signals = pd.read_csv(scores_dir / "signals" / dataset / "ab_signals.csv")
    signals = signals[signals["split"].isin(TARGET_SPLITS)].copy()

    role2 = pd.read_csv(jiye_dir / dataset / "role2_signals.csv")
    fp_conf = pd.read_csv(
        scores_dir / "fp_conformal" / dataset / "fp_conformal.csv"
    ).drop(columns=["dataset", "split"], errors="ignore")

    frame = signals.merge(role2, on="row_uid", suffixes=("", "_r2")).merge(
        fp_conf, on="row_uid", suffixes=("", "_fp")
    )
    task_type = frame["task_type"].iloc[0]

    out = frame[
        ["row_uid", "dataset", "task_type", "split", "cv_fold", "Y_final"]
    ].copy()

    # 감사 대상 예측과 그 오차. 지문 대표 모델을 주 대상으로 삼는다.
    truth = pd.to_numeric(frame["Y_final"]).to_numpy(dtype=float)
    out["pred_fp_primary"] = frame["pred_fp_primary"]
    out["pred_chemberta_augmented"] = frame["pred_chemberta_augmented"]
    out["abs_error_fp"] = np.abs(truth - frame["pred_fp_primary"].to_numpy(float))
    out["abs_error_cb"] = np.abs(
        truth - frame["pred_chemberta_augmented"].to_numpy(float)
    )

    # 기준 모형 신호 다섯 종
    raw = {
        "ad_knn": frame["ad_knn_tanimoto_top5_mean"],
        "ad_density": frame["ad_local_density_count_s040"],
    }
    if task_type == "classification":
        raw["disagreement"] = frame["model_disagreement_probability_gap"]
        raw["conformal_cb"] = frame["aps_set_size"]
        raw["conformal_fp"] = frame["fp_aps_set_size"]
    else:
        raw["disagreement"] = frame["model_disagreement_abs"]
        raw["conformal_cb"] = frame["conformal_width"]
        raw["conformal_fp"] = frame["fp_conformal_width"]

    for name, sign in BASELINE_SIGNALS.items():
        out[f"base__{name}"] = sign * pd.to_numeric(raw[name]).to_numpy(dtype=float)

    # A·B 축 신호
    for model in AXIS_MODELS:
        for axis in AXIS_SIGNALS:
            column = f"{model}__{axis}__std"
            if column in frame:
                out[f"axis__{model}__{axis}"] = frame[column]

    # 물성 안 백분위 정규화. 제거 실험은 이 열을 쓴다.
    for column in [c for c in out.columns if c.startswith(("base__", "axis__"))]:
        out[f"{column}__pct"] = out[column].rank(pct=True, method="average")

    return out


def summarize(frame: pd.DataFrame) -> list[dict]:
    """test 분할에서 각 신호와 실제 오차의 상관. 지문 대표 모델 오차 기준."""
    subset = frame[frame["split"].eq("test")]
    error = subset["abs_error_fp"].to_numpy(dtype=float)
    rows = []
    for column in [
        c for c in subset.columns if c.startswith(("base__", "axis__")) and not c.endswith("__pct")
    ]:
        values = subset[column].to_numpy(dtype=float)
        rho = (
            np.nan
            if np.allclose(values, values[0]) or np.allclose(error, error[0])
            else float(spearmanr(values, error).statistic)
        )
        rows.append(
            {
                "dataset": subset["dataset"].iloc[0],
                "task_type": subset["task_type"].iloc[0],
                "group": "기준" if column.startswith("base__") else "축",
                "signal": column,
                "spearman_vs_abs_error": rho,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="기준선과 A·B 신호 통합")
    parser.add_argument("--jiye-dir", required=True, help="담당2 outputs 최상위")
    parser.add_argument("--scores-dir", required=True, help="signals·fp_conformal 상위")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()

    jiye_dir = Path(args.jiye_dir)
    scores_dir = Path(args.scores_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets or sorted(
        path.name
        for path in (scores_dir / "signals").iterdir()
        if path.is_dir() and not path.name.startswith("_")
    )

    summaries = []
    for index, dataset in enumerate(datasets, 1):
        frame = build(dataset, jiye_dir, scores_dir)
        dataset_out = out_dir / dataset
        dataset_out.mkdir(parents=True, exist_ok=True)
        frame.to_csv(dataset_out / "evaluation_signals.csv", index=False)
        summaries.extend(summarize(frame))
        print(f"[{index}/{len(datasets)}] {dataset}: {len(frame):,}행", flush=True)

    summary = pd.DataFrame(summaries)
    summary_dir = out_dir / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_dir / "evaluation_signal_summary.csv", index=False)

    print()
    print("=== 신호별 오차 상관 (지문 대표 모델 기준, 22종 중앙값) ===")
    table = (
        summary.groupby(["group", "signal"])["spearman_vs_abs_error"]
        .agg(중앙="median", 양수=lambda s: int((s > 0).sum()), 유효="count")
        .reset_index()
        .sort_values(["group", "중앙"], ascending=[True, False])
    )
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
