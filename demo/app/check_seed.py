"""데모가 쓰는 시드 42 체크포인트가 담당2의 pred_chemberta_augmented와 맞는지 확인한다.

담당2가 시드 세 개로 다시 학습했으므로 점추정을 어느 시드에서 가져왔는지가
명시되어 있지 않다. 데모는 42를 쓰기로 했는데 그 가정이 맞는지 재본다.
"""
import sys, warnings, json
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from engine import predict_chemberta

BASE = Path("../../..").resolve()
ROOT = BASE / "Jiye/checkpoints"
rows = []
for ds in ("bbb_martins", "herg", "lipophilicity_astrazeneca", "caco2_wang"):
    j = pd.read_csv(BASE / f"Jiye/outputs/{ds}/role2_signals.csv",
                    usecols=["row_uid", "pred_chemberta_augmented", "split"])
    sp = pd.read_csv(BASE / f"Juhyeong/data/processed/pipeline_yoonsoo/{ds}/splits.csv",
                     usecols=["row_uid", "parent_smiles"], low_memory=False)
    m = j.merge(sp, on="row_uid")
    m = m[m.split == "test"].head(20)
    record = {"dataset": ds, "n": len(m)}
    for seed in (42, 43, 44):
        path = ROOT / f"chemberta_seed_{seed}" / "checkpoints" / ds / "augmented"
        pred = predict_chemberta(path, m.parent_smiles.tolist())
        if pred is None:
            record[f"seed{seed}"] = None; continue
        record[f"seed{seed}"] = float(np.abs(pred - m.pred_chemberta_augmented.to_numpy()).max())
    mean3 = np.mean([predict_chemberta(ROOT / f"chemberta_seed_{s}" / "checkpoints" / ds / "augmented",
                                       m.parent_smiles.tolist()) for s in (42, 43, 44)], axis=0)
    record["3시드평균"] = float(np.abs(mean3 - m.pred_chemberta_augmented.to_numpy()).max())
    rows.append(record)
    print(f"  {ds} 완료", flush=True)

d = pd.DataFrame(rows)
print("\n=== 담당2 pred_chemberta_augmented와의 최대 절대차 ===")
print(d.to_string(index=False))
best = d[["seed42", "seed43", "seed44", "3시드평균"]].mean().idxmin()
print(f"\n가장 잘 맞는 것: {best}")
