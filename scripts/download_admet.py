"""TDC ADMET Benchmark Group(22종)을 전부 내려받아 data/raw/에 저장한다.

TDC가 제공하는 골격 기반 분할(train_val/test)을 그대로 저장하지만,
본 프로젝트의 최종 분할은 계획서 7.1절(부모 분자 확정 -> 중복 제거 ->
골격 분할 -> 변형 생성) 절차를 별도로 다시 적용한다. 이 스크립트는
원본 라벨 데이터를 확보하는 착수 단계용이다.
"""

import json
from pathlib import Path

from tdc.benchmark_group import admet_group

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

group = admet_group(path=str(RAW_DIR / "_tdc_cache"))
names = group.dataset_names

summary = []
for name in names:
    benchmark = group.get(name)
    train_val = benchmark["train_val"]
    test = benchmark["test"]

    out_dir = RAW_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    train_val.to_csv(out_dir / "train_val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    n_total = len(train_val) + len(test)
    label_col = [c for c in train_val.columns if c not in ("Drug_ID", "Drug", "Y")]
    summary.append({
        "name": name,
        "n_train_val": len(train_val),
        "n_test": len(test),
        "n_total": n_total,
        "columns": list(train_val.columns),
    })
    print(f"[ok] {name:35s} train_val={len(train_val):>6d}  test={len(test):>6d}  total={n_total:>6d}")

summary_path = RAW_DIR / "download_summary.json"
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\n{len(names)}개 데이터셋 다운로드 완료. 요약: {summary_path}")
