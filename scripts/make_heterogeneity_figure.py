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
ACC,WARM="#0E6A5E","#B4653A"
f=pd.read_csv(f"{B}/Juhyeong/data/processed/scores_role4/heterogeneity/heterogeneity_by_dataset.csv")
f=f.sort_values(["B1_protonation","effect_auprc"],ascending=[True,True])
fig,axes=plt.subplots(1,2,figsize=(12.4,6.2),gridspec_kw={"width_ratios":[1.45,1]})

# (a) 숲 그림 — 물성별 AUPRC 효과와 신뢰구간
ax=axes[0]; y=np.arange(len(f))
for lvl,c in [("허용",ACC),("주의",WARM)]:
    mask=(f.B1_protonation==lvl).to_numpy()
    sub=f[mask]; ys=y[mask]
    ax.errorbar(sub.effect_auprc,ys,
                xerr=[sub.effect_auprc-sub["ci_low__기준+B"],sub["ci_high__기준+B"]-sub.effect_auprc],
                fmt="o",color=c,ecolor=c,elinewidth=1.4,capsize=2.5,ms=5,zorder=3)
ax.axvline(0,color="#444",lw=1)
ax.set_yticks(y); ax.set_yticklabels([f"{d}" for d in f.dataset],fontsize=7.5)
ax.set_xlabel("오차 탐지 AUPRC 변화 (기준 대비 기준+B)")
ax.set_title("물성별 효과 크기와 95퍼센트 신뢰구간\n골격 군집 부트스트랩",fontsize=10.5)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],color=ACC,marker="o",ls="",label="B1 양성자화 허용 (15종)"),
                   Line2D([],[],color=WARM,marker="o",ls="",label="B1 양성자화 주의 (7종)")],
          frameon=False,fontsize=8.5,loc="lower right")

# (b) 허용성별 집계
ax=axes[1]
rng=np.random.default_rng(20260829); pos=0; labels=[]
for metric,name in [("effect_aurc","정규화 AURC 개선"),("effect_auprc","오차 탐지 AUPRC 개선")]:
    for lvl,c in [("허용",ACC),("주의",WARM)]:
        v=f.loc[f.B1_protonation==lvl,metric].to_numpy()
        bs=np.array([rng.choice(v,len(v),replace=True).mean() for _ in range(5000)])
        lo,hi=np.percentile(bs,[2.5,97.5])
        ax.errorbar(v.mean(),pos,xerr=[[v.mean()-lo],[hi-v.mean()]],fmt="o",color=c,capsize=4,ms=7,elinewidth=1.8)
        labels.append(f"{name}\n{lvl} {len(v)}종"); pos+=1
    pos+=0.6
ax.axvline(0,color="#444",lw=1)
ax.set_yticks([0,1,2.6,3.6]); ax.set_yticklabels(labels,fontsize=8.5)
ax.set_xlabel("효과 크기 (양수면 B 추가가 개선)")
ax.set_title("변형이 화학적으로 유의미한 물성에서만\nB가 효과를 낸다 (사후 탐색)",fontsize=10.5)
ax.invert_yaxis()
fig.tight_layout(); fig.savefig(f"{OUT}/06_이질성.png"); plt.close(fig)
print("생성 완료")
