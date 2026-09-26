#!/usr/bin/env python
"""최종 보고서용 그림. 발표 자료에 그대로 옮길 수 있게 크게, 글자도 크게 그린다.

색은 역할로만 쓴다. 파랑이 제안 신호, 회색이 기준선, 주황이 경고다. 값의
크기를 색으로 한 번 더 칠하지 않는다.
"""

from __future__ import annotations

import json
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
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": "#0b0b0b",
    "axes.labelcolor": "#52514e", "font.size": 11.5,
    "xtick.major.size": 0, "ytick.major.size": 0,
})
BLUE, GRAY, AMBER, MUTED, GRID = "#2a78d6", "#898781", "#eb6834", "#c3c2b7", "#e1e0d9"
ROOT = Path("/Users/zzuhyeong2/Library/CloudStorage/GoogleDrive-a01056371120@gmail.com/"
            "My Drive/Conference_2026/Juhyeong")
PROC = ROOT / "data/processed"
OUT = PROC / "scores_role4_r2/figures"
RNG = np.random.default_rng(20260902)


def ci(values: np.ndarray) -> tuple[float, float]:
    draws = [RNG.choice(values, len(values)).mean() for _ in range(8000)]
    return tuple(np.percentile(draws, [2.5, 97.5]))


def fig_ablation() -> None:
    """제거 실험. 0을 세로선으로 두고 신뢰구간을 가로 막대로 그린다."""
    table = pd.read_csv(PROC / "scores_role4_r2/expanded_ablation_22/preregistered_ablation.csv")
    base = table["aurc__기준(AD+컨포멀)"]
    rows = [("기준 + 모델 불일치", "기준+모델불일치", GRAY),
            ("기준 + A축", "기준+A", GRAY),
            ("기준 + B축", "기준+B", BLUE),
            ("기준 + A축 + B축", "기준+A+B", BLUE),
            ("전체", "전체", GRAY)]
    fig, ax = plt.subplots(figsize=(8.6, 3.5))
    for i, (label, column, color) in enumerate(rows):
        effect = (base - table[f"aurc__{column}"]).dropna().to_numpy()
        lo, hi = ci(effect)
        y = len(rows) - 1 - i
        ax.plot([lo, hi], [y, y], color=color, lw=2.4, solid_capstyle="round",
                alpha=1 if color == BLUE else .55)
        ax.plot([effect.mean()], [y], "o", color=color, ms=9, zorder=3,
                alpha=1 if color == BLUE else .55)
        ax.text(hi + .004, y, f"{effect.mean():+.4f}", va="center", fontsize=11,
                color="#0b0b0b" if color == BLUE else "#52514e",
                fontweight="bold" if color == BLUE else "normal")
    ax.axvline(0, color=MUTED, lw=1.2, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=12)
    ax.set_xlabel("정규화 AURC 개선량  (오른쪽이 좋다 · 95퍼센트 신뢰구간)")
    ax.set_xlim(-.05, .185)
    ax.grid(axis="x", color=GRID, lw=.7)
    ax.set_axisbelow(True)
    fig.savefig(OUT / "R1_제거실험.png")
    plt.close(fig)


def fig_cyp_split() -> None:
    """물성별 B축 효과. CYP 여섯 종에 몰려 있다는 것이 한눈에 보이게."""
    table = pd.read_csv(PROC / "scores_role4_r2/explore_fp_only.csv").set_index("dataset")
    effect = (table["기준"] - table["+B 모양"]).sort_values()
    colors = [BLUE if d.startswith("cyp") else GRAY for d in effect.index]
    # 막대마다 투명도를 달리하려면 색에 직접 섞어야 한다. alpha는 목록을 받지 않는다.
    fills = [(*matplotlib.colors.to_rgb(c), 1 if c == BLUE else .5) for c in colors]
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    y = np.arange(len(effect))
    ax.barh(y, effect.to_numpy(), color=fills, height=.66)
    ax.axvline(0, color=MUTED, lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(effect.index, fontsize=10)
    for tick, color in zip(ax.get_yticklabels(), colors):
        tick.set_color("#0b0b0b" if color == BLUE else "#898781")
    ax.set_xlabel("B축을 더했을 때의 정규화 AURC 개선량")
    ax.grid(axis="x", color=GRID, lw=.7)
    ax.set_axisbelow(True)
    ax.plot([], [], "s", color=BLUE, ms=9, label="CYP 물성 6종")
    ax.plot([], [], "s", color=GRAY, ms=9, alpha=.5, label="그 외 16종")
    ax.legend(frameon=False, loc="lower right", fontsize=11)
    fig.savefig(OUT / "R2_물성별효과.png")
    plt.close(fig)


def fig_replication() -> None:
    """사전 등록한 재현·확장 결과. 판정 근거가 신뢰구간이라는 것을 그대로 보인다."""
    new = json.loads((PROC / "scores_new/replication/replication_result.json").read_text())
    bio = json.loads((PROC / "scores_biogen/replication/replication_result.json").read_text())
    rows = [("CYP1A2 억제", new["cyp1a2_veith"], BLUE, "재현"),
            ("CYP2C19 억제", new["cyp2c19_veith"], BLUE, "재현"),
            ("사람 간 미세소체", bio["biogen_hlm_clint"], GRAY, "확장"),
            ("쥐 간 미세소체", bio["biogen_rlm_clint"], GRAY, "확장"),
            ("MDR1-MDCK 유출 (대조)", bio["biogen_mdr1_mdck_er"], GRAY, "확장"),
            ("용해도 pH 6.8 (대조)", bio["biogen_solubility_ph68"], GRAY, "확장")]
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    for i, (label, record, color, _) in enumerate(rows):
        lo, hi = record["1차"]["ci"]
        effect = record["1차"]["effect"]
        y = len(rows) - 1 - i
        excludes_zero = lo > 0
        ax.plot([lo, hi], [y, y], color=color, lw=2.4, solid_capstyle="round",
                alpha=1 if excludes_zero else .45)
        ax.plot([effect], [y], "o", color=color, ms=9, zorder=3,
                alpha=1 if excludes_zero else .45)
        ax.text(hi + .008, y, f"{effect:+.3f}" + ("  ✓" if excludes_zero else ""),
                va="center", fontsize=11,
                color="#0b0b0b" if excludes_zero else "#898781",
                fontweight="bold" if excludes_zero else "normal")
    ax.axvline(0, color=MUTED, lw=1.2)
    ax.axhline(3.5, color=GRID, lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=11.5)
    ax.set_xlabel("정규화 AURC 개선량  (골격 군집 부트스트랩 95퍼센트 신뢰구간)")
    ax.text(.30, 4.6, "사전 등록 재현", fontsize=11, color=BLUE, fontweight="bold")
    ax.text(.30, 2.4, "사전 등록 확장", fontsize=11, color="#898781")
    ax.set_xlim(-.12, .40)
    ax.grid(axis="x", color=GRID, lw=.7)
    ax.set_axisbelow(True)
    fig.savefig(OUT / "R3_재현확장.png")
    plt.close(fig)


def fig_practical() -> None:
    """실제 쓰임새. 상위·하위 20퍼센트를 걸러냈을 때 크게 틀린 예측의 비율."""
    table = pd.read_csv(PROC / "scores_role4_r2/single_signal_benchmark.csv")
    labels = ["가장 안전하다고\n고른 20퍼센트", "전체 평균", "가장 위험하다고\n고른 20퍼센트"]
    base = [table["safe|기준(AD+컨포멀)"].mean(), 1.0, table["risk|기준(AD+컨포멀)"].mean()]
    with_b = [table["safe|기준+B"].mean(), 1.0, table["risk|기준+B"].mean()]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.bar(x - .19, base, width=.34, color=GRAY, alpha=.5, label="기준 모형")
    ax.bar(x + .19, with_b, width=.34, color=BLUE, label="기준 + B축")
    for xi, (a, b) in enumerate(zip(base, with_b)):
        ax.text(xi - .19, a + .04, f"{a:.2f}배", ha="center", fontsize=11, color="#52514e")
        ax.text(xi + .19, b + .04, f"{b:.2f}배", ha="center", fontsize=11,
                color="#0b0b0b", fontweight="bold")
    ax.axhline(1, color=MUTED, lw=1, ls="-")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("크게 틀린 예측의 비율\n(전체 평균 대비)")
    ax.set_ylim(0, 1.95)
    ax.grid(axis="y", color=GRID, lw=.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    fig.savefig(OUT / "R4_실사용.png")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_ablation()
    fig_cyp_split()
    fig_replication()
    fig_practical()
    print("만든 그림:", ", ".join(sorted(p.name for p in OUT.glob("R*.png"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
