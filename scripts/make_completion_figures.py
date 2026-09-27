"""계획서 미실행 분석의 그림 셋. 05 겹침 구조, 09 골격 거리, 10 벤치마크 표."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")

for cand in ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand; break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "grid.linestyle": "-"})

B = "/Users/zzuhyeong2/Library/CloudStorage/GoogleDrive-a01056371120@gmail.com/My Drive/Conference_2026"
SRC = f"{B}/Juhyeong/data/processed/scores_role4/plan_completion"
OUT = f"{B}/Juhyeong/data/processed/scores_role4/figures"
ACC, GREY, WARM = "#0E6A5E", "#8A9793", "#B4653A"
OURS = {"표현 불안정성 A", "B1 호변이성질체", "B1 양성자화", "B3 입체 표기", "통합 B (조건부)", "결합 점수"}

# --- 05 겹침 구조: 상위 10퍼센트 집합의 자카드 행렬 ---
ov = pd.read_csv(f"{SRC}/overlap_summary.csv")
names = [n for n in ["적용가능도메인 이웃", "적용가능도메인 밀도", "컨포멀 지문", "컨포멀 ChemBERTa",
                     "모델 불일치", "표현 불안정성 A", "통합 B (조건부)"]
         if n in set(ov.signal_a) | set(ov.signal_b)]
M = np.full((len(names), len(names)), np.nan)
idx = {n: i for i, n in enumerate(names)}
for r in ov.itertuples():
    if r.signal_a in idx and r.signal_b in idx:
        M[idx[r.signal_a], idx[r.signal_b]] = M[idx[r.signal_b], idx[r.signal_a]] = r.자카드
np.fill_diagonal(M, 1.0)

fig, ax = plt.subplots(figsize=(7.4, 6.2))
im = ax.imshow(M, cmap="BuGn", vmin=0, vmax=0.25)
ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
ax.set_yticklabels(names, fontsize=9)
for i in range(len(names)):
    for j in range(len(names)):
        if i == j: continue
        ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8.5,
                color="white" if M[i, j] > 0.15 else "#333")
for i, n in enumerate(names):
    if n in OURS:
        ax.get_yticklabels()[i].set_color(ACC); ax.get_yticklabels()[i].set_weight("bold")
        ax.get_xticklabels()[i].set_color(ACC); ax.get_xticklabels()[i].set_weight("bold")
ax.set_title("상위 10퍼센트 위험 집합이 서로 얼마나 겹치나\n자카드 지수, 22종 중앙값 "
             "(초록 글자가 우리 신호)", fontsize=11, pad=12)
ax.grid(False)
fig.colorbar(im, ax=ax, shrink=.75, label="자카드 지수")
fig.savefig(f"{OUT}/05_겹침구조.png"); plt.close(fig)

# --- 09 골격 거리: 거리 분위별 축 백분위와 오차 ---
di = pd.read_csv(f"{SRC}/scaffold_distance_by_dataset.csv")
axes_cols = [c for c in ["표현 불안정성 A", "통합 B (조건부)"] if c in di.columns]
g = di.groupby("거리 분위")[axes_cols + ["정규화 오차"]].mean()
se = di.groupby("거리 분위")[axes_cols + ["정규화 오차"]].sem()

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
x = g.index.to_numpy()
a1.errorbar(x, g["정규화 오차"], yerr=se["정규화 오차"], color=WARM, marker="o",
            lw=2, capsize=4, label="실제 오차")
a1.axhline(1.0, color=GREY, ls=":", lw=1)
a1.set_xticks(x); a1.set_xlabel("학습 골격으로부터의 거리 분위 (1=가까움)")
a1.set_ylabel("정규화 오차 (물성 평균 = 1)")
a1.set_title("멀어질수록 오차는 커진다", fontsize=11)
a1.legend(frameon=False, fontsize=9)

for col, c, m in zip(axes_cols, [ACC, "#3C7D9E"], ["o", "s"]):
    a2.errorbar(x, g[col], yerr=se[col], color=c, marker=m, lw=2, capsize=4, label=col)
a2.axhline(0.5, color=GREY, ls=":", lw=1)
a2.set_xticks(x); a2.set_xlabel("학습 골격으로부터의 거리 분위 (1=가까움)")
a2.set_ylabel("축 신호 백분위 (물성 내)"); a2.set_ylim(0.40, 0.60)
a2.set_title("그런데 우리 축은 커지지 않는다", fontsize=11)
a2.legend(frameon=False, fontsize=9)
fig.suptitle("우리 신호는 화학 공간상의 위치를 대신 재는 것이 아니다", fontsize=12.5, y=1.02)
fig.savefig(f"{OUT}/09_골격거리.png"); plt.close(fig)

# --- 10 벤치마크: 신호별 오차 탐지 AUPRC와 정규화 AURC ---
bt = pd.read_csv(f"{SRC}/benchmark_table.csv").set_index("signal")
order = [n for n in ["결합 점수", "모델 불일치", "표현 불안정성 A", "컨포멀 지문",
                     "통합 B (조건부)", "B1 양성자화", "적용가능도메인 이웃", "B1 호변이성질체",
                     "B3 입체 표기", "적용가능도메인 밀도", "컨포멀 ChemBERTa"] if n in bt.index]
bt = bt.loc[order]
colors = [ACC if n in OURS else GREY for n in bt.index]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
y = np.arange(len(bt))[::-1]
a1.barh(y, bt["AUPRC"], color=colors, height=.68)
a1.axvline(bt["AUPRC"].min() * 0, color=GREY, lw=.8)
a1.set_yticks(y); a1.set_yticklabels(bt.index, fontsize=9.5)
a1.set_xlabel("오차 탐지 AUPRC (높을수록 좋음)"); a1.set_xlim(0.20, 0.38)
a1.set_title("1차 지표", fontsize=11)
for yi, v in zip(y, bt["AUPRC"]):
    a1.text(v + .003, yi, f"{v:.3f}", va="center", fontsize=8.5)

a2.barh(y, bt["정규화 AURC"], color=colors, height=.68)
a2.axvline(1.0, color=WARM, ls="--", lw=1.2)
a2.text(1.005, y[-1] + .6, "무작위", color=WARM, fontsize=8.5)
a2.set_xlabel("정규화 AURC (낮을수록 좋음)"); a2.set_xlim(0.45, 1.15)
a2.set_title("위험-커버리지", fontsize=11)
for yi, v in zip(y, bt["정규화 AURC"]):
    a2.text(v + .008, yi, f"{v:.3f}", va="center", fontsize=8.5)
fig.suptitle("신뢰성 신호 벤치마크 · 22종 평균 (초록이 우리 신호)", fontsize=12.5, y=.98)
fig.savefig(f"{OUT}/10_벤치마크.png"); plt.close(fig)

print("저장 완료")
for f in ["05_겹침구조.png", "09_골격거리.png", "10_벤치마크.png"]:
    print(f"  {f}  {os.path.getsize(f'{OUT}/{f}')//1024} KB")
