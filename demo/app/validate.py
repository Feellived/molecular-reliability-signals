"""데모 채점기가 파이프라인 값을 재현하는지 22종 전체에서 검증한다.

시험 분할에 이미 있는 분자를 데모로 다시 채점해 저장된 값과 대조한다.
표현 불안정성 A는 무작위 SMILES 생성 시드가 파이프라인과 달라 변형 집합
자체가 다르므로 대조 대상에서 뺀다. B축과 적용가능도메인은 결정적이므로
일치해야 한다.
"""
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from score import score

BASE = Path(__file__).resolve().parents[2]
BUNDLE = BASE / "data/processed/scores_role4/demo_bundle"
EV = BASE / "data/processed/scores_role4/evaluation"
N = 8

rows, failures = [], []
datasets = sorted(p.name for p in EV.iterdir() if p.is_dir() and not p.name.startswith("_"))
for index, dataset in enumerate(datasets, 1):
    started = time.time()
    ev = pd.read_csv(EV / dataset / "evaluation_signals.csv")
    sp = pd.read_csv(BASE / f"data/processed/pipeline_yoonsoo/{dataset}/splits.csv",
                     usecols=["row_uid", "parent_smiles"], low_memory=False)
    test = ev[ev.split == "test"].merge(sp, on="row_uid").head(N)
    for r in test.itertuples():
        try:
            out = score(BUNDLE, dataset, r.parent_smiles)
        except Exception as exc:
            failures.append({"dataset": dataset, "row": r.row_uid,
                             "error": f"{type(exc).__name__}: {exc}"})
            continue
        rows.append({
            "dataset": dataset, "task": out["task_type"],
            "pred_pipe": float(r.pred_fp_primary), "pred_demo": out["prediction"],
            "b_pipe": float(r.cond_B__fp_primary__std),
            "b_demo": out["reliability_axes"]["입력 상태 민감성"]["dispersion"],
            "ad_pipe": -float(r.base__ad_knn),
            "ad_demo": out["reliability_axes"]["화학 공간 위치"]["nearest5_tanimoto"],
            "verdict": out["verdict"]["level"],
        })
    print(f"[{index}/{len(datasets)}] {dataset:32s} {len(test)}건 {time.time()-started:5.1f}초",
          flush=True)

d = pd.DataFrame(rows)
d.to_csv(Path(__file__).parent / "validation_report.csv", index=False)

print(f"\n채점 {len(d)}건, 실패 {len(failures)}건")
for f in failures[:10]:
    print(f"  {f['dataset']:28s} {f['error'][:70]}")

print(f"\n{'항목':14s}{'최대 절대차':>12s}{'중앙 절대차':>12s}{'상관':>9s}")
for a, b, label in (("pred_pipe", "pred_demo", "예측값"),
                    ("b_pipe", "b_demo", "B축 흩어짐"),
                    ("ad_pipe", "ad_demo", "적용가능도메인")):
    x, y = d[a].to_numpy(float), d[b].to_numpy(float)
    print(f"{label:14s}{np.abs(x-y).max():12.5f}{np.median(np.abs(x-y)):12.5f}"
          f"{np.corrcoef(x, y)[0, 1]:9.4f}")

print("\n물성별 예측값 절대차 상위 5종")
worst = (d.assign(diff=(d.pred_pipe - d.pred_demo).abs())
         .groupby(["dataset", "task"])["diff"].mean().sort_values(ascending=False).head(5))
print(worst.round(5).to_string())

print("\n판정 분포:", d["verdict"].value_counts().to_dict())
