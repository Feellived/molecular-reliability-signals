import pandas as pd, numpy as np, warnings, json
from pathlib import Path
from scipy.stats import rankdata, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score
warnings.filterwarnings("ignore")
B=Path("/Users/zzuhyeong2/Library/CloudStorage/GoogleDrive-a01056371120@gmail.com/My Drive/Conference_2026")
EV=B/"Juhyeong/data/processed/scores_role4/evaluation"
BASE=["base__ad_knn__pct","base__ad_density__pct","base__disagreement__pct",
      "base__conformal_cb__pct","base__conformal_fp__pct"]
A=["axis__cb_augmented__A__pct"]
BOLD=["axis__fp_primary__B_combined__pct","axis__cb_augmented__B_combined__pct"]
BNEW=["cond_B__fp_primary__std__pct","cond_B__cb_augmented__std__pct"]
CFG={"기준":BASE,"기준+A":BASE+A,"기준+B(무조건)":BASE+BOLD,
     "기준+B(조건부)":BASE+BNEW,"기준+A+B(조건부)":BASE+A+BNEW}
def naurc(s,e):
    f=lambda x:float(np.mean(np.cumsum(e[np.argsort(x,kind="stable")])/np.arange(1,len(e)+1)))
    o,r=f(e),float(np.mean(e))
    return np.nan if r-o<1e-12 else (f(s)-o)/(r-o)
rows=[]
for d in sorted(p.name for p in EV.iterdir() if p.is_dir() and not p.name.startswith("_")):
    m=pd.read_csv(EV/d/"evaluation_signals.csv")
    me,te=m[m.split=="meta"],m[m.split=="test"]
    err=te.abs_error_fp.to_numpy(float); tgt=rankdata(me.abs_error_fp)/len(me)
    if m.task_type.iloc[0]=="classification":
        lab=((te.pred_fp_primary.to_numpy(float)>=.5).astype(int)!=pd.to_numeric(te.Y_final).astype(int)).astype(int)
    else:
        lab=(err>=np.quantile(err,.8)).astype(int)
    r={"dataset":d,"task":m.task_type.iloc[0]}
    for n,f in CFG.items():
        u=[c for c in f if c in m and m[c].nunique()>1]
        if not u: r[f"aurc__{n}"]=np.nan; r[f"auprc__{n}"]=np.nan; continue
        s=Ridge(alpha=1.0).fit(me[u].to_numpy(float),tgt).predict(te[u].to_numpy(float))
        r[f"aurc__{n}"]=naurc(s,err)
        r[f"auprc__{n}"]=average_precision_score(lab,s) if lab.min()!=lab.max() else np.nan
    rows.append(r)
t=pd.DataFrame(rows)
t.to_csv(B/"Juhyeong/data/processed/scores_role4/conditional_signals/conditional_ablation_by_dataset.csv",index=False)
for met,arrow,sign in [("auprc","높을수록 좋음",1),("aurc","낮을수록 좋음",-1)]:
    print(f"[{met.upper()} {arrow}]")
    for n in CFG:
        c=t[f"{met}__{n}"].dropna(); print(f"  {n:18s} 평균 {c.mean():.4f}  중앙 {c.median():.4f}")
    for n in list(CFG)[1:]:
        dd=(t[f"{met}__{n}"]-t[f"{met}__기준"]).dropna()
        p=wilcoxon(dd)[1] if len(dd)>=6 else np.nan
        print(f"    {n:18s} 변화 {dd.mean():+.4f}  개선 {int((dd*sign>0).sum())}/{len(dd)}종  p={p:.3f}")
    print()
