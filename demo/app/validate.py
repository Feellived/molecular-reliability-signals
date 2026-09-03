"""데모 채점기가 파이프라인 값을 재현하는지 검증한다.

시험 분할에 이미 있는 분자를 데모로 다시 채점해 저장된 값과 대조한다.
A축은 무작위 SMILES 생성 시드가 달라 변형 자체가 다르므로 완전 일치를
기대하지 않는다. B축과 적용가능도메인은 결정적이므로 일치해야 한다.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from score import score

BASE = Path(__file__).resolve().parents[2]
BUNDLE = BASE / "data/processed/scores_role4/demo_bundle"
N = 6

rows = []
for dataset in ("herg", "lipophilicity_astrazeneca"):
    ev = pd.read_csv(BASE / f"data/processed/scores_role4/evaluation/{dataset}/evaluation_signals.csv")
    sp = pd.read_csv(BASE / f"data/processed/pipeline_yoonsoo/{dataset}/splits.csv",
                     usecols=["row_uid", "parent_smiles"], low_memory=False)
    test = ev[ev.split == "test"].merge(sp, on="row_uid").head(N)
    for r in test.itertuples():
        try:
            out = score(BUNDLE, dataset, r.parent_smiles)
        except Exception as exc:
            rows.append({"dataset": dataset, "row": r.row_uid, "error": type(exc).__name__})
            continue
        rows.append({
            "dataset": dataset, "row": r.row_uid,
            "예측_파이프라인": round(float(r.pred_fp_primary), 4),
            "예측_데모": out["prediction"],
            "B흩어짐_파이프라인": round(float(r.cond_B__fp_primary__std), 5),
            "B흩어짐_데모": round(out["reliability_axes"]["입력 상태 민감성"]["dispersion"], 5),
            "AD_파이프라인": round(-float(r.base__ad_knn), 4),
            "AD_데모": out["reliability_axes"]["화학 공간 위치"]["nearest5_tanimoto"],
        })
d = pd.DataFrame(rows)
print(d.to_string(index=False)); print()
for a, b, label in (("예측_파이프라인", "예측_데모", "예측값"),
                    ("B흩어짐_파이프라인", "B흩어짐_데모", "B축 흩어짐"),
                    ("AD_파이프라인", "AD_데모", "적용가능도메인")):
    if a not in d: continue
    x, y = d[a].to_numpy(float), d[b].to_numpy(float)
    print(f"{label:12s} 최대 절대차 {np.abs(x-y).max():.5f}   상관 {np.corrcoef(x,y)[0,1]:.4f}")
