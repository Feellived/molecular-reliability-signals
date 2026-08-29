import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd, numpy as np, glob, os, warnings
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

for cand in ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand; break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "grid.linestyle": "-"})

B = "/Users/zzuhyeong2/Library/CloudStorage/GoogleDrive-a01056371120@gmail.com/My Drive/Conference_2026"
OUT = f"{B}/Juhyeong/data/processed/scores_role4/figures"
os.makedirs(OUT, exist_ok=True)
ACC, GREY, WARM = "#0E6A5E", "#8A9793", "#B4653A"

sig = []
for f in sorted(glob.glob(f"{B}/Juhyeong/data/processed/scores_role4/signals/*/ab_signals.csv")):
    d = os.path.basename(os.path.dirname(f))
    a = pd.read_csv(f); a = a[a.split == "test"].copy(); a["dataset"] = d
    sig.append(a)
allsig = pd.concat(sig, ignore_index=True)
summary = pd.read_csv(f"{B}/Juhyeong/data/processed/scores_role4/signals/_summary/ab_signal_summary.csv")

# --- 그림 1: 신호별 오차 판별력 (기준선과 축 신호를 같은 오차 기준으로) ---
ev = pd.read_csv(f"{B}/Juhyeong/data/processed/scores_role4/evaluation/_summary/evaluation_signal_summary.csv")
SIG = {"base__conformal_fp": "컨포멀 (지문)", "base__ad_knn": "적용가능도메인\n이웃 유사도",
       "base__ad_density": "적용가능도메인\n국소 밀도", "base__disagreement": "모델 불일치",
       "base__conformal_cb": "컨포멀 (ChemBERTa)",
       "axis__fp_primary__B_combined": "B 통합 (지문)", "axis__fp_primary__B1_protonation": "B1 양성자화 (지문)",
       "axis__cb_augmented__A": "A 표기 (ChemBERTa)", "axis__fp_primary__B1_tautomer": "B1 호변이성질체 (지문)",
       "axis__cb_augmented__B1_protonation": "B1 양성자화 (ChemBERTa)",
       "axis__cb_augmented__B3_stereo": "B3 입체 (ChemBERTa)", "axis__fp_primary__B3_stereo": "B3 입체 (지문)",
       "axis__cb_augmented__B_combined": "B 통합 (ChemBERTa)",
       "axis__cb_augmented__B1_tautomer": "B1 호변이성질체 (ChemBERTa)"}
gg = (ev.dropna(subset=["spearman_vs_abs_error"]).groupby(["group", "signal"])["spearman_vs_abs_error"]
        .median().reset_index().sort_values("spearman_vs_abs_error"))
gg = gg[gg.signal.isin(SIG)]
fig, ax = plt.subplots(figsize=(8.6, 6.0))
ax.barh(np.arange(len(gg)), gg.spearman_vs_abs_error, height=.72,
        color=[ACC if r == "축" else "#98A6A1" for r in gg.group])
ax.set_yticks(np.arange(len(gg))); ax.set_yticklabels([SIG[x] for x in gg.signal], fontsize=9)
for yi, v in zip(np.arange(len(gg)), gg.spearman_vs_abs_error):
    ax.text(v + .006, yi, f"{v:.3f}", va="center", fontsize=8.5, color="#333")
ax.set_xlabel("실제 오차와의 Spearman 상관 (22종 중앙값, test)")
ax.set_xlim(0, .44)
ax.set_title("신호별 오차 판별력\n감사 대상은 지문 대표 모델의 예측", fontsize=11, pad=12)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=ACC, label="이번 연구의 축 신호"), Patch(color="#98A6A1", label="기존 기준선")],
          frameon=False, fontsize=9, loc="lower right")
fig.savefig(f"{OUT}/01_상관.png"); plt.close(fig)

# --- 그림 2: 기존 신호와의 중복성 ---
OURS = {"cb_augmented__A__std": "A 표기\n(ChemBERTa 증강)",
        "cb_augmented__B_combined__std": "B 통합\n(ChemBERTa 증강)",
        "fp_primary__B1_protonation__std": "B1 양성자화\n(지문)",
        "fp_primary__B_combined__std": "B 통합\n(지문)"}
rows = []
for d, m0 in allsig.groupby("dataset"):
    r = pd.read_csv(f"{B}/Jiye/outputs/{d}/role2_signals.csv")
    m = m0.merge(r, on="row_uid", suffixes=("", "_r"))
    dis = "model_disagreement_probability_gap" if m.task_type.iloc[0] == "classification" else "model_disagreement_abs"
    for o in OURS:
        for b, bl in [("ad_knn_tanimoto_top5_mean", "AD\n이웃 유사도"),
                      ("ad_local_density_count_s040", "AD\n국소 밀도"), (dis, "모델\n불일치")]:
            if b in m and m[o].nunique() > 1 and m[b].nunique() > 1:
                rows.append({"ours": OURS[o], "base": bl, "rho": abs(spearmanr(m[o], m[b]).statistic)})
h = pd.DataFrame(rows).pivot_table(index="ours", columns="base", values="rho", aggfunc="median")
h = h.reindex(list(OURS.values()))[["모델\n불일치", "AD\n이웃 유사도", "AD\n국소 밀도"]]
fig, ax = plt.subplots(figsize=(8.2, 4.2))
im = ax.imshow(h.values, cmap="BuGn", vmin=0, vmax=1)
ax.set_xticks(range(h.shape[1])); ax.set_xticklabels(h.columns, fontsize=9.5)
ax.tick_params(axis="x", pad=6)
ax.set_yticks(range(h.shape[0])); ax.set_yticklabels(h.index, fontsize=9)
for i in range(h.shape[0]):
    for j in range(h.shape[1]):
        ax.text(j, i, f"{h.values[i, j]:.3f}", ha="center", va="center", fontsize=10, color="#16211F")
ax.set_title("우리 신호와 기존 신호의 겹침 정도\n(0에 가까울수록 다른 정보, 22종 중앙값)", fontsize=11, pad=12)
ax.grid(False)
cb = fig.colorbar(im, ax=ax, shrink=.8); cb.set_label("Spearman 상관 절댓값", fontsize=9)
fig.savefig(f"{OUT}/02_중복성.png"); plt.close(fig)

# --- 그림 3: 흔들림 분위별 평균 오차 ---
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
for ax, (col, name) in zip(axes, [("fp_primary__B_combined__std", "B 통합 (지문 모델)"),
                                  ("cb_augmented__A__std", "A 표기 (ChemBERTa 증강)")]):
    err = "fp_primary__abs_error" if col.startswith("fp") else "cb_augmented__abs_error"
    prof = []
    for d, g in allsig.groupby("dataset"):
        if g[col].nunique() < 5: continue
        q = pd.qcut(g[col].rank(method="first"), 4, labels=["하위 25%", "25~50%", "50~75%", "상위 25%"])
        e = g.groupby(q, observed=True)[err].mean()
        prof.append(e / e.mean())
    p = pd.DataFrame(prof)
    ax.bar(range(4), p.mean(), color=[GREY, GREY, WARM, ACC],
           yerr=p.std() / np.sqrt(len(p)), capsize=3, ecolor="#666")
    ax.axhline(1.0, ls="--", lw=1, color="#444")
    ax.set_xticks(range(4)); ax.set_xticklabels(p.columns, fontsize=9)
    ax.set_ylabel("평균 오차 (물성별 평균=1로 정규화)")
    ax.set_title(f"{name}\n흔들림 분위별 실제 오차 ({len(p)}종 평균)", fontsize=10)
    ax.set_ylim(0.8, None)
fig.tight_layout(); fig.savefig(f"{OUT}/03_분위별오차.png"); plt.close(fig)

# --- 그림 4: 물성별 상관 분포 ---
fig, ax = plt.subplots(figsize=(8.4, 4.0))
sel = [("cb_augmented", "A", "A 표기\nChemBERTa 증강"), ("fp_primary", "B1_protonation", "B1 양성자화\n지문"),
       ("fp_primary", "B_combined", "B 통합\n지문"), ("cb_augmented", "B_combined", "B 통합\nChemBERTa 증강")]
data = [summary[(summary.model == m) & (summary.axis == a)].spearman_vs_abs_error.dropna().values for m, a, _ in sel]
bp = ax.boxplot(data, tick_labels=[l for _, _, l in sel], patch_artist=True, widths=.55,
                medianprops=dict(color=ACC, lw=2), flierprops=dict(marker="o", ms=4, mfc=GREY, mec="none"))
for b in bp["boxes"]: b.set(facecolor="#DCE9E1", edgecolor=GREY)
ax.axhline(0, color="#444", lw=1)
ax.set_ylabel("실제 오차와의 Spearman 상관")
ax.set_title("물성 22종에 걸친 신호 성능 분포 (test)", fontsize=11, pad=12)
fig.savefig(f"{OUT}/04_물성별분포.png"); plt.close(fig)

print("생성 완료:", OUT)
for f in sorted(os.listdir(OUT)): print("  ", f)
