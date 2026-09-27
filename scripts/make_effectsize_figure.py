import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
for c in ["AppleGothic","Apple SD Gothic Neo","NanumGothic"]:
    if any(f.name==c for f in font_manager.fontManager.ttflist): plt.rcParams["font.family"]=c; break
plt.rcParams["axes.unicode_minus"]=False
plt.rcParams.update({"figure.dpi":150,"savefig.bbox":"tight","axes.spines.top":False,
                     "axes.spines.right":False,"axes.grid":True,"grid.alpha":.25})
B="/Users/zzuhyeong2/Library/CloudStorage/GoogleDrive-a01056371120@gmail.com/My Drive/Conference_2026"
OUT=f"{B}/Juhyeong/data/processed/scores_role4/figures"
ACC,GREY,WARM="#0E6A5E","#98A6A1","#B4653A"
t=pd.read_csv(f"{B}/Juhyeong/data/processed/scores_role4/expanded_ablation_22/preregistered_ablation.csv")
h=pd.read_csv(f"{B}/Juhyeong/data/processed/scores_role4/heterogeneity/heterogeneity_by_dataset.csv")
rng=np.random.default_rng(20260902)
def boot(v):
    v=np.asarray(v,float); v=v[~np.isnan(v)]
    bs=np.array([rng.choice(v,len(v),replace=True).mean() for _ in range(8000)])
    return v.mean(), *np.percentile(bs,[2.5,97.5]), int((v>0).sum()), len(v)
rows=[]
for lbl,col in [("A 추가","기준+A"),("B 추가","기준+B"),("A+B 추가","기준+A+B")]:
    rows.append((lbl,)+boot(t["aurc__기준(AD+컨포멀)"]-t[f"aurc__{col}"]))
rows.append(("A+B 추가\n(pH 조건 충족 15종)",)+boot(h[h.B1_protonation=="허용"].effect_aurc))
rows.append(("A+B 추가\n(조건 미충족 7종)",)+boot(h[h.B1_protonation=="주의"].effect_aurc))
fig,ax=plt.subplots(figsize=(8.8,4.6))
y=np.arange(len(rows))[::-1]
for (lbl,m,lo,hi,pos,n),yi in zip(rows,y):
    c=ACC if lo>0 else (WARM if hi<0 else GREY)
    ax.errorbar(m,yi,xerr=[[m-lo],[hi-m]],fmt="o",color=c,capsize=4,ms=8,elinewidth=2)
    ax.text(hi+0.004,yi,f"{m:+.4f}   {pos}/{n}종 개선",va="center",fontsize=9,color="#333")
ax.axvline(0,color="#444",lw=1.2)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows],fontsize=9.5)
ax.set_xlabel("정규화 AURC 개선 폭 (양수면 위험 선별이 나아짐)")
ax.set_title("기준 모형 대비 효과 크기와 95퍼센트 신뢰구간\n계획서 6.5절이 요구하는 보고 형식",fontsize=11,pad=12)
ax.set_xlim(-0.05,0.105)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],color=ACC,marker="o",ls="",label="신뢰구간이 0보다 큼"),
                   Line2D([],[],color=GREY,marker="o",ls="",label="신뢰구간이 0을 포함"),
                   Line2D([],[],color=WARM,marker="o",ls="",label="신뢰구간이 0보다 작음")],
          frameon=False,fontsize=8.5,loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/08_효과크기.png"); plt.close(fig)
for r in rows: print(f"{r[0][:22]:24s} {r[1]:+.4f} [{r[2]:+.4f}, {r[3]:+.4f}]  {r[4]}/{r[5]}")
