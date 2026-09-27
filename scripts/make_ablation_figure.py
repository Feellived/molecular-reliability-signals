import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd, numpy as np, glob, os, sys, warnings
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
for c in ["AppleGothic","Apple SD Gothic Neo","NanumGothic"]:
    if any(f.name==c for f in font_manager.fontManager.ttflist): plt.rcParams["font.family"]=c; break
plt.rcParams["axes.unicode_minus"]=False
plt.rcParams.update({"figure.dpi":150,"savefig.bbox":"tight","axes.spines.top":False,
                     "axes.spines.right":False,"axes.grid":True,"grid.alpha":.25})
B="/Users/zzuhyeong2/Library/CloudStorage/GoogleDrive-a01056371120@gmail.com/My Drive/Conference_2026"
sys.path.insert(0,f"{B}/Juhyeong/scripts")
from run_ablation import BASE_FEATURES, B_FEATURES
OUT=f"{B}/Juhyeong/data/processed/scores_role4/figures"
ACC,WARM,GREY="#0E6A5E","#B4653A","#98A6A1"

fig,axes=plt.subplots(1,2,figsize=(11.2,4.4))

# (a) 위험-커버리지 곡선 — 개선폭이 중앙값에 가까운 물성
ds="cyp3a4_veith"
m=pd.read_csv(f"{B}/Juhyeong/data/processed/scores_role4/evaluation/{ds}/evaluation_signals.csv")
me,te=m[m.split=="meta"],m[m.split=="test"]
em,et=me.abs_error_fp.to_numpy(float),te.abs_error_fp.to_numpy(float)
tgt=rankdata(em)/len(em)
ax=axes[0]
for name,feats,col,ls in [("기준",BASE_FEATURES,GREY,"-"),("기준+B",BASE_FEATURES+B_FEATURES,ACC,"-")]:
    u=[c for c in feats if c in m and m[c].nunique()>1]
    s=Ridge(alpha=1.0).fit(me[u].to_numpy(float),tgt).predict(te[u].to_numpy(float))
    o=np.argsort(s,kind="stable"); r=np.cumsum(et[o])/np.arange(1,len(et)+1)
    ax.plot(np.arange(1,len(et)+1)/len(et),r,color=col,ls=ls,lw=2,label=name)
o=np.argsort(et); ax.plot(np.arange(1,len(et)+1)/len(et),np.cumsum(et[o])/np.arange(1,len(et)+1),
                          color="#444",ls=":",lw=1.3,label="최적 (오차 순 정렬)")
ax.axhline(et.mean(),color="#999",ls="--",lw=1,label="무작위")
ax.set_xlabel("채택 비율 (안전한 예측부터)"); ax.set_ylabel("채택분의 평균 오차")
ax.set_title(f"위험-커버리지 곡선 — {ds}",fontsize=10.5); ax.legend(frameon=False,fontsize=8.5,loc="upper left")

# (b) 물성별 AURC 변화
t=pd.read_csv(f"{B}/Juhyeong/data/processed/scores_role4/ablation/ablation_by_dataset.csv")
d=(t["aurc__기준+B"]-t["aurc__기준"]).sort_values()
ax=axes[1]
ax.barh(np.arange(len(d)),d.values,color=[ACC if v<0 else WARM for v in d.values],height=.75)
ax.set_yticks(np.arange(len(d))); ax.set_yticklabels(t.loc[d.index,"dataset"],fontsize=7.5)
ax.axvline(0,color="#444",lw=1)
ax.set_xlabel("정규화 AURC 변화 (음수면 B 추가가 개선)")
ax.set_title(f"기준 대비 기준+B — 22종 중 {int((d<0).sum())}종 개선\n평균 {d.mean():+.4f}, Wilcoxon p = 0.156",fontsize=10.5)
fig.tight_layout(); fig.savefig(f"{OUT}/05_제거실험.png"); plt.close(fig)
print("생성:",f"{OUT}/05_제거실험.png")
