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
ORDER=["적용가능도메인 단독","기준(AD+컨포멀)","기준+A","기준+B","기준+A+B","기준+모델불일치","전체"]
LBL=["적용가능도메인\n단독","기준\n(AD+컨포멀)","기준+A","기준+B","기준+A+B","기준+\n모델불일치","전체"]
COL=[GREY,"#4A5A56",ACC,ACC,ACC,GREY,"#0B4F46"]

fig,axes=plt.subplots(1,2,figsize=(12.6,4.6))
for ax,met,title,arrow in [(axes[0],"aurc","정규화 AURC","낮을수록 좋음"),
                            (axes[1],"auprc","오차 탐지 AUPRC","높을수록 좋음")]:
    v=[t[f"{met}__{n}"].mean() for n in ORDER]
    bars=ax.bar(range(len(v)),v,color=COL,width=.68)
    base=v[1]
    ax.axhline(base,ls="--",lw=1.2,color="#444")
    ax.text(len(v)-.4,base,"  기준 모형",va="center",fontsize=8.5,color="#444")
    for i,val in enumerate(v):
        ax.text(i,val+(0.012 if met=="aurc" else 0.006),f"{val:.3f}",ha="center",fontsize=8.5)
    ax.set_xticks(range(len(v))); ax.set_xticklabels(LBL,fontsize=8.5)
    ax.set_ylabel(f"{title} ({arrow})")
    ax.set_title(f"{title}\n계획서 6.4절 제거 실험 순서",fontsize=10.5)
    ax.set_ylim(0, max(v)*1.18)
from matplotlib.patches import Patch
axes[0].legend(handles=[Patch(color=ACC,label="이번 연구의 축"),Patch(color=GREY,label="기존 신호")],
               frameon=False,fontsize=9,loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/07_제거실험_사전지정.png"); plt.close(fig)
print("생성 완료")
