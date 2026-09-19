import pandas as pd, numpy as np, warnings
from pathlib import Path
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")

B = Path("/content/drive/MyDrive/Conference_2026")
EV = B / "Juhyeong/data/processed/scores_role4/evaluation"
SPLITS_DIR = B / "Juhyeong/data/processed/pipeline_yoonsoo"  # 군집 정보가 있는 경로

BASE_PRE = ["base__ad_knn__pct", "base__ad_density__pct", "base__conformal_cb__pct", "base__conformal_fp__pct"]
Bx = ["cond_B__fp_primary__std__pct", "cond_B__cb_augmented__std__pct"]

CFG = {
     "기준(AD+컨포멀)": BASE_PRE,
     "기준+B": BASE_PRE + Bx,
}

def naurc(s, e):
    f = lambda x: float(np.mean(np.cumsum(e[np.argsort(x, kind="stable")]) / np.arange(1, len(e) + 1)))
    o, r = f(e), float(np.mean(e))
    return np.nan if r - o < 1e-12 else (f(s) - o) / (r - o)

print("1. 시험 분자별 위험 점수 추출 및 군집 정보 병합 중...")
datasets_data = {}

for d in sorted(p.name for p in EV.iterdir() if p.is_dir() and not p.name.startswith("_")):
    m = pd.read_csv(EV / d / "evaluation_signals.csv")
    me, te = m[m.split == "meta"], m[m.split == "test"]
    
    # splits.csv에서 scaffold_group 정보 가져와서 합치기
    splits_path = SPLITS_DIR / d / "splits.csv"
    if splits_path.exists():
        splits_df = pd.read_csv(splits_path, low_memory=False)
        te = pd.merge(te, splits_df[['row_uid', 'scaffold_group']], on='row_uid', how='left')
    else:
        te['scaffold_group'] = np.arange(len(te)) # 파일이 없을 경우 대비 안전장치
    
    err = te.abs_error_fp.to_numpy(float)
    tgt = rankdata(me.abs_error_fp) / len(me)
    
    scores = {}
    for n, f in CFG.items():
        u = [c for c in f if c in m and m[c].nunique() > 1]
        s = Ridge(alpha=1.0).fit(me[u].to_numpy(float), tgt).predict(te[u].to_numpy(float))
        scores[n] = s
    
    df_test = pd.DataFrame({
        'row_uid': te['row_uid'].values if 'row_uid' in te else np.arange(len(te)),
        'scaffold_group': te['scaffold_group'].values,
        'err': err,
        'score_base': scores["기준(AD+컨포멀)"],
        'score_base_B': scores["기준+B"]
    })
    datasets_data[d] = df_test

print(f"총 {len(datasets_data)}개 물성에 대한 예측 완료.")
print("\n2. 골격 군집 단위 부트스트랩 2000회 진행 중... (시간이 다소 소요됩니다)")

N_BOOTSTRAP = 2000
boot_mean_improvements = []

for i in range(N_BOOTSTRAP):
    if (i + 1) % 500 == 0:
        print(f" - Bootstrap 진행 상황: {i + 1} / {N_BOOTSTRAP}")
        
    improvements_in_iter = []
    for dataset_name, df in datasets_data.items():
        df['scaffold_group'] = df['scaffold_group'].fillna('unknown') # 결측치 안전장치
        unique_scaffolds = df['scaffold_group'].unique()
        sampled_scaffolds = np.random.choice(unique_scaffolds, size=len(unique_scaffolds), replace=True)
        
        scaf_dict = dict(tuple(df.groupby('scaffold_group')))
        sample_df = pd.concat([scaf_dict[scaf] for scaf in sampled_scaffolds], ignore_index=True)
        
        err_sample = sample_df['err'].values
        s_base = sample_df['score_base'].values
        s_base_B = sample_df['score_base_B'].values
        
        aurc_base = naurc(s_base, err_sample)
        aurc_base_B = naurc(s_base_B, err_sample)
        
        if not np.isnan(aurc_base) and not np.isnan(aurc_base_B):
            improvements_in_iter.append(aurc_base - aurc_base_B)
            
    mean_imp = np.mean(improvements_in_iter)
    boot_mean_improvements.append(mean_imp)

boot_mean_improvements = np.array(boot_mean_improvements)
ci_lower = np.percentile(boot_mean_improvements, 2.5)
ci_upper = np.percentile(boot_mean_improvements, 97.5)
p_value = np.sum(boot_mean_improvements <= 0) / N_BOOTSTRAP
real_mean = np.mean(boot_mean_improvements)

print("\n=== 최종 분석 결과 (GitHub 보고서 업로드용) ===")
print(f"22개 물성 평균 개선량 (기준 - 기준+B): {real_mean:.4f}")
print(f"95% 신뢰구간: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"진짜 p-value: {p_value:.4f}")
