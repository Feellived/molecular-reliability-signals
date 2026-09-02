import pandas as pd, numpy as np, warnings
from pathlib import Path
from scipy.stats import rankdata, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score
warnings.filterwarnings("ignore")
B=Path("/Users/zzuhyeong2/Library/CloudStorage/GoogleDrive-a01056371120@gmail.com/My Drive/Conference_2026")
EV=B/"Juhyeong/data/processed/scores_role4/evaluation"
# 계획서 4.4절: 기준 모형은 적용가능도메인과 컨포멀만
BASE_PRE=["base__ad_knn__pct","base__ad_density__pct","base__conformal_cb__pct","base__conformal_fp__pct"]
DIS=["base__disagreement__pct"]
A=["axis__cb_augmented__A__pct"]
Bx=["cond_B__fp_primary__std__pct","cond_B__cb_augmented__std__pct"]
# 6.4절 제거 실험 순서
CFG={"적용가능도메인 단독":["base__ad_knn__pct","base__ad_density__pct"],
     "기준(AD+컨포멀)":BASE_PRE,
     "기준+모델불일치":BASE_PRE+DIS,
     "기준+A":BASE_PRE+A,
     "기준+B":BASE_PRE+Bx,
     "기준+A+B":BASE_PRE+A+Bx,
     "전체":BASE_PRE+DIS+A+Bx}
def naurc(s,e):
    f=lambda x: float(np.mean(np.cumsum(e[np.argsort(x,kind="stable")])/np.arange(1,len(e)+1)))
    o,r=f(e),float(np.mean(e)); return np.nan if r-o<1e-12 else (f(s)-o)/(r-o)
rows=[]
for d in sorted(p.name for p in EV.iterdir() if p.is_dir() and not p.name.startswith("_")):
    m=pd.read_csv(EV/d/"evaluation_signals.csv"); me,te=m[m.split=="meta"],m[m.split=="test"]
    err=te.abs_error_fp.to_numpy(float); tgt=rankdata(me.abs_error_fp)/len(me)
    lab=(((te.pred_fp_primary.to_numpy(float)>=.5).astype(int)!=pd.to_numeric(te.Y_final).astype(int)).astype(int)
         if m.task_type.iloc[0]=="classification" else (err>=np.quantile(err,.8)).astype(int))
    r={"dataset":d}
    for n,f in CFG.items():
        u=[c for c in f if c in m and m[c].nunique()>1]
        s=Ridge(alpha=1.0).fit(me[u].to_numpy(float),tgt).predict(te[u].to_numpy(float))
        r[f"auprc__{n}"]=average_precision_score(lab,s) if lab.min()!=lab.max() else np.nan
        r[f"aurc__{n}"]=naurc(s,err)
    rows.append(r)
t=pd.DataFrame(rows)
t.to_csv(B/"Juhyeong/data/processed/scores_role4/expanded_ablation_22/preregistered_ablation.csv",index=False)
print("=== 계획서 6.4절 순서, 사전 지정 기준 모형 (AD+컨포멀) ===")
print(f"{'구성':20s}{'AUPRC':>9s}{'AURC':>9s}")
for n in CFG: print(f"{n:20s}{t[f'auprc__{n}'].mean():9.4f}{t[f'aurc__{n}'].mean():9.4f}")
print()
print("=== 기준 모형 대비 (1차·2차 가설 판정) ===")
for n in ("기준+모델불일치","기준+A","기준+B","기준+A+B","전체"):
    for met,sign in (("auprc",1),("aurc",-1)):
        d=((t[f"{met}__{n}"]-t[f"{met}__기준(AD+컨포멀)"])*sign).dropna()
        print(f"  {n:16s} {met.upper():5s} {d.mean():+.4f}  개선 {int((d>0).sum())}/22  p={wilcoxon(d)[1]:.3f}")
