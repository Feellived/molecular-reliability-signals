import pandas as pd, numpy as np, warnings
from pathlib import Path
from scipy.stats import rankdata, wilcoxon
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import average_precision_score
warnings.filterwarnings("ignore")
B=Path("/Users/zzuhyeong2/Library/CloudStorage/GoogleDrive-a01056371120@gmail.com/My Drive/Conference_2026")
EV=B/"Juhyeong/data/processed/scores_role4/evaluation"
BASE=["base__ad_knn__pct","base__ad_density__pct","base__disagreement__pct",
      "base__conformal_cb__pct","base__conformal_fp__pct"]
SIMPLE=["axis__cb_augmented__A__pct","cond_B__fp_primary__std__pct","cond_B__cb_augmented__std__pct"]
RICH=[f"rich__{m}__{g}__{s}__pct" for m in ("fp_primary","cb_augmented")
      for g in ("A","B") for s in ("std","max_dev","shift","rel_std","flip")]
def naurc(s,e):
    f=lambda x: float(np.mean(np.cumsum(e[np.argsort(x,kind="stable")])/np.arange(1,len(e)+1)))
    o,r=f(e),float(np.mean(e)); return np.nan if r-o<1e-12 else (f(s)-o)/(r-o)
def run(learner, feats):
    P,A=[],[]
    for d in sorted(p.name for p in EV.iterdir() if p.is_dir() and not p.name.startswith("_")):
        m=pd.read_csv(EV/d/"evaluation_signals.csv"); me,te=m[m.split=="meta"],m[m.split=="test"]
        err=te.abs_error_fp.to_numpy(float); cls=m.task_type.iloc[0]=="classification"
        def lab(fr):
            e=fr.abs_error_fp.to_numpy(float)
            return (((fr.pred_fp_primary.to_numpy(float)>=.5).astype(int)!=pd.to_numeric(fr.Y_final).astype(int)).astype(int)
                    if cls else (e>=np.quantile(e,.8)).astype(int))
        u=[c for c in feats if c in m and m[c].nunique()>1]
        if not u: P.append(np.nan); A.append(np.nan); continue
        Xm,Xt=me[u].to_numpy(float),te[u].to_numpy(float)
        ym=lab(me)
        if learner=="ridge":
            s=Ridge(alpha=1.0).fit(Xm,rankdata(me.abs_error_fp)/len(me)).predict(Xt)
        else:
            if ym.min()==ym.max(): P.append(np.nan); A.append(np.nan); continue
            s=LogisticRegression(C=0.5,max_iter=2000).fit(Xm,ym).predict_proba(Xt)[:,1]
        yt=lab(te)
        P.append(average_precision_score(yt,s) if yt.min()!=yt.max() else np.nan); A.append(naurc(s,err))
    return np.array(P),np.array(A)
cfg=[("Ridge / 기준만","ridge",BASE),("Ridge / +단순축","ridge",BASE+SIMPLE),
     ("Ridge / +확장축","ridge",BASE+RICH),
     ("로지스틱 / 기준만","logit",BASE),("로지스틱 / +단순축","logit",BASE+SIMPLE),
     ("로지스틱 / +확장축","logit",BASE+RICH)]
out={}
for n,l,f in cfg: out[n]=run(l,f)
print(f"{'구성':22s}{'AUPRC':>9s}{'AURC':>9s}")
for n,_,_ in cfg: print(f"{n:22s}{np.nanmean(out[n][0]):9.4f}{np.nanmean(out[n][1]):9.4f}")
print()
print("=== 축 추가 효과 (같은 학습기 안에서 기준만 대비) ===")
for lname,bkey in [("Ridge","Ridge / 기준만"),("로지스틱","로지스틱 / 기준만")]:
    for suffix in ["+단순축","+확장축"]:
        k=f"{lname} / {suffix}"
        dp=out[k][0]-out[bkey][0]; da=out[bkey][1]-out[k][1]
        dp2,da2=dp[~np.isnan(dp)],da[~np.isnan(da)]
        print(f"  {k:22s} AUPRC {np.mean(dp2):+.4f} {int((dp2>0).sum())}/{len(dp2)}종 p={wilcoxon(dp2)[1]:.3f}"
              f"   AURC {np.mean(da2):+.4f} {int((da2>0).sum())}/{len(da2)}종 p={wilcoxon(da2)[1]:.3f}")
