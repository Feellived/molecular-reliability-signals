#!/usr/bin/env python
"""상세 보고서에만 들어가는 그림. 간결판 그림(make_report_figures.py)과 같은 양식을 쓴다.

D1은 물성 22종 각각의 B축 효과와 골격 군집 부트스트랩 신뢰구간이다. 간결판은
평균만 보였으므로, 어느 물성의 구간이 0을 배제하는지는 여기서 처음 드러난다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

for candidate in ("AppleGothic", "Apple SD Gothic Neo", "NanumGothic"):
    if any(f.name == candidate for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = candidate
        break
plt.rcParams.update({
    "axes.unicode_minus": False, "figure.dpi": 200, "savefig.bbox": "tight",
    "savefig.facecolor": "#fcfcfb", "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#c3c2b7", "axes.linewidth": .8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": "#0b0b0b",
    "axes.labelcolor": "#52514e", "font.size": 11,
    "xtick.major.size": 0, "ytick.major.size": 0,
})
BLUE, GRAY, AMBER, MUTED, GRID = "#2a78d6", "#898781", "#eb6834", "#c3c2b7", "#e1e0d9"
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"
OUT = ROOT / "docs/figures"


def fig_forest() -> None:
    """물성별 효과와 골격 신뢰구간. 위에서부터 효과가 큰 순서다."""
    table = pd.read_csv(PROC / "scores_role4_r2/b1_statistics/b1_by_dataset.csv")
    table = table.sort_values("effect__기준+B").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.8, 7.9))
    for y, row in table.iterrows():
        lo, hi, eff = row.scaffold_ci_low, row.scaffold_ci_high, row["effect__기준+B"]
        cyp = row.dataset.startswith("cyp")
        if hi < 0:
            color, alpha = AMBER, 1.0          # 구간 전체가 0 아래: B를 더하면 나빠진다
        elif cyp:
            color, alpha = BLUE, 1.0
        else:
            color, alpha = GRAY, .55
        ax.plot([lo, hi], [y, y], color=color, lw=2.2, alpha=alpha, solid_capstyle="round")
        significant = row.p_bh < 0.05
        ax.plot([eff], [y], "o", ms=8.5 if significant else 6.5, zorder=3,
                color=color, alpha=alpha,
                markeredgecolor="#0b0b0b" if significant else color,
                markeredgewidth=1.1 if significant else 0)
        # 표본 수는 축 바깥 오른쪽 열에 둔다. 구간 끝이나 범례와 겹치지 않게.
        ax.text(1.015, y, f"{int(row.n_test):,}", va="center", ha="left", fontsize=9,
                color="#898781", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.axvline(0, color=MUTED, lw=1.2, zorder=1)
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels(table.dataset, fontsize=9.5)
    for tick, name in zip(ax.get_yticklabels(), table.dataset):
        tick.set_color("#0b0b0b" if name.startswith("cyp") else "#6b6a65")
    ax.set_xlim(-0.33, 0.49)
    ax.set_xlabel("기준 모형에 B축을 더했을 때의 정규화 AURC 개선량 (골격 군집 부트스트랩 95% 신뢰구간)")
    ax.grid(axis="x", color=GRID, lw=.7)
    ax.set_axisbelow(True)
    ax.plot([], [], "s", color=BLUE, ms=8, label="CYP 물성 6종")
    ax.plot([], [], "s", color=GRAY, ms=8, alpha=.55, label="그 외 16종")
    ax.plot([], [], "s", color=AMBER, ms=8, label="구간 전체가 0 아래")
    ax.plot([], [], "o", color="#fcfcfb", ms=8, markeredgecolor="#0b0b0b",
            markeredgewidth=1.1, label="BH 보정 후 p < 0.05")
    ax.text(1.015, len(table) - 0.2, "시험 분자", va="bottom", ha="left", fontsize=9,
            color="#52514e", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=4,
              fontsize=9.5, handletextpad=.4, columnspacing=1.4)
    fig.savefig(OUT / "D1_물성별신뢰구간.png")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_forest()
    print("만든 그림:", ", ".join(sorted(p.name for p in OUT.glob("D*.png"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
