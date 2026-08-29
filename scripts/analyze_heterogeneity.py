#!/usr/bin/env python
"""물성별 효과 크기와 조절 변수 분석 (연구계획서 6.5절, 사후 탐색).

RQ3의 전체 평균은 음성이지만 물성별 편차가 크다. 이 스크립트는 어떤 물성에서
B 추가가 효과를 내는지, 그 차이를 설명하는 변수가 있는지를 살핀다.

주의. 이 분석은 사전 지정되지 않은 사후 탐색이다. 전체 결과가 음성인 것을
확인한 뒤 수행했으므로 여기서 나온 관계는 가설을 확정하는 근거가 아니라
후속 연구가 검정할 가설을 제안하는 자료로만 쓴다. 6.5절에 따라 다중 비교는
Benjamini-Hochberg 절차로 통제한다.

검토하는 조절 변수

  변형 허용성    담당1의 06_transformation_allowance_revised.csv에서 온다.
                 해당 물성의 측정 조건이 pH를 보존하는지에 따라 허용과 주의로
                 나뉜다. 화학적으로 변형이 유의미한 물성에서만 신호가 작동한다면
                 이 변수가 효과를 설명해야 한다. 가장 해석 가능한 후보다.
  meta 표본 수   결합 규칙을 학습하는 표본의 크기다. 작으면 규칙이 잡음을 학습한다.
  기준 성능      기준 모형이 이미 강하면 추가 신호가 기여할 여지가 줄어든다.
  신호 크기      B 변형이 실제로 만들어낸 예측 변동의 크기다. 변동이 없으면
                 신호가 있을 수 없다.
  과제 유형      분류와 회귀
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

warnings.filterwarnings("ignore")

ALLOWANCE_COLUMNS = ["B1_tautomer", "B1_protonation", "B3_stereo"]


def load_effects(scores_dir: Path) -> pd.DataFrame:
    detection = pd.read_csv(
        scores_dir / "error_detection" / "error_detection_by_dataset.csv"
    )
    ablation = pd.read_csv(scores_dir / "ablation" / "ablation_by_dataset.csv")
    frame = detection.merge(
        ablation[["dataset", "n_meta", "aurc__기준", "aurc__기준+B"]], on="dataset"
    )
    # AURC는 낮을수록 좋으므로 부호를 뒤집어 두 지표의 방향을 개선 쪽으로 맞춘다.
    frame["effect_aurc"] = frame["aurc__기준"] - frame["aurc__기준+B"]
    frame["effect_auprc"] = frame["delta_auprc__기준+B"]
    return frame


def add_moderators(
    frame: pd.DataFrame, reports_dir: Path, signals_dir: Path
) -> pd.DataFrame:
    allowance = pd.read_csv(reports_dir / "06_transformation_allowance_revised.csv")
    allowance = allowance[["dataset", *ALLOWANCE_COLUMNS, "evidence_level"]]
    frame = frame.merge(allowance, on="dataset", how="left")

    # B 변형이 실제로 만들어낸 예측 변동의 크기
    sizes = []
    for dataset in frame["dataset"]:
        signals = pd.read_csv(signals_dir / dataset / "ab_signals.csv")
        signals = signals[signals["split"].eq("test")]
        column = "fp_primary__B_combined__std"
        sizes.append(
            {
                "dataset": dataset,
                "b_signal_mean": float(signals[column].mean()),
                "b_signal_nonzero": float((signals[column] > 0).mean()),
            }
        )
    return frame.merge(pd.DataFrame(sizes), on="dataset", how="left")


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues)
    ranked = pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0, 1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="물성별 효과와 조절 변수 사후 분석")
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--reports-dir", required=True, help="담당1 reports 폴더")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    scores_dir = Path(args.scores_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = load_effects(scores_dir)
    frame = add_moderators(frame, Path(args.reports_dir), scores_dir / "signals")
    frame.to_csv(out_dir / "heterogeneity_by_dataset.csv", index=False)

    print("=== 물성별 B 추가 효과 (AUPRC 기준, 신뢰구간 포함) ===")
    view = frame.sort_values("effect_auprc", ascending=False)
    for _, row in view.iterrows():
        mark = "＊" if row["ci_low__기준+B"] > 0 else ("－" if row["ci_high__기준+B"] < 0 else " ")
        print(
            f"  {mark} {row['dataset']:32s} {row['effect_auprc']:+.4f} "
            f"[{row['ci_low__기준+B']:+.3f}, {row['ci_high__기준+B']:+.3f}]  "
            f"{row['task_type'][:5]}  meta {int(row['n_meta']):4d}  "
            f"B1양성자화 {row['B1_protonation']}"
        )

    # 연속형 조절 변수
    continuous = {
        "meta 표본 수": "n_meta",
        "기준 AUPRC": "auprc__기준",
        "B 신호 크기": "b_signal_mean",
        "B 신호 비영 비율": "b_signal_nonzero",
        "오차 사건 비율": "error_rate",
        "골격군 수": "n_scaffold_group",
    }
    rows = []
    for label, column in continuous.items():
        for metric, name in [("effect_auprc", "AUPRC"), ("effect_aurc", "AURC")]:
            stat = spearmanr(frame[column], frame[metric])
            rows.append(
                {
                    "조절변수": label,
                    "지표": name,
                    "spearman": stat.statistic,
                    "p": stat.pvalue,
                }
            )

    # 범주형 조절 변수. 허용과 주의를 나눠 효과 크기를 비교한다.
    for column in ALLOWANCE_COLUMNS:
        groups = frame.groupby(column)
        if len(groups) < 2:
            continue
        levels = [g for g, _ in groups if len(frame[frame[column] == g]) >= 3]
        if len(levels) < 2:
            continue
        a, b = levels[0], levels[1]
        for metric, name in [("effect_auprc", "AUPRC"), ("effect_aurc", "AURC")]:
            x = frame.loc[frame[column] == a, metric]
            y = frame.loc[frame[column] == b, metric]
            stat = mannwhitneyu(x, y)
            rows.append(
                {
                    "조절변수": f"{column} ({a} {len(x)}종 대 {b} {len(y)}종)",
                    "지표": name,
                    "spearman": float(x.mean() - y.mean()),
                    "p": float(stat.pvalue),
                }
            )

    moderators = pd.DataFrame(rows)
    moderators["p_adj"] = benjamini_hochberg(moderators["p"].to_numpy())
    moderators = moderators.sort_values("p")
    moderators.to_csv(out_dir / "moderator_analysis.csv", index=False)

    print()
    print("=== 조절 변수 분석 (사후 탐색, BH 보정) ===")
    print(f"{'조절변수':44s}{'지표':>6s}{'통계량':>9s}{'p':>8s}{'p보정':>8s}")
    for _, row in moderators.iterrows():
        print(
            f"{row['조절변수']:44s}{row['지표']:>6s}{row['spearman']:>9.3f}"
            f"{row['p']:>8.3f}{row['p_adj']:>8.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
