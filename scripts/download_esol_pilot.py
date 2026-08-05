"""MoleculeNet ESOL(Delaney) 데이터셋을 내려받는다.

계획서 5.1절에 따라 이 데이터셋은 파이프라인 구축과 디버깅용 예비
데이터로만 쓰며, 최종 결론의 근거로는 사용하지 않는다. 최종 데이터는
scripts/download_admet.py가 받는 TDC ADMET Benchmark Group 22종이다.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "raw" / "esol_pilot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"

df = pd.read_csv(URL)
out_path = OUT_DIR / "delaney-processed.csv"
df.to_csv(out_path, index=False)
print(f"[ok] esol_pilot  n={len(df)}  saved to {out_path}")
