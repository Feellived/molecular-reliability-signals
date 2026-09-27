#!/usr/bin/env python
"""위험-커버리지 기반 제거 실험 (연구계획서 6.3·6.4절).

1차 가설은 기준 모형에 입력 상태 민감성 B를 추가하면 골격 기반 시험 집합의
정규화 AURC가 감소한다는 것이다. 2차 가설은 표현 불안정성 A에 대한 같은 진술이다.
이 스크립트가 그 두 비교를 수행한다.

위험-커버리지 곡선은 신뢰성 신호로 예측을 정렬한 뒤, 안전하다고 판단된 것부터
차례로 채택하면서 채택 비율마다 남은 예측의 평균 오차를 기록한 것이다. 좋은
신호일수록 위험한 예측을 먼저 걸러내므로 곡선이 아래로 눌린다.

AURC는 그 곡선 아래 면적이며 작을수록 좋다. 다만 원값은 물성마다 라벨 단위가
달라 비교할 수 없으므로 다음과 같이 정규화한다.

    정규화 AURC = (AURC - AURC_최적) / (AURC_무작위 - AURC_최적)

AURC_최적은 실제 오차 순으로 완벽하게 정렬했을 때의 값이고, AURC_무작위는
정렬하지 않았을 때의 값이다. 따라서 0은 완벽, 1은 무작위와 같음을 뜻하며
물성 간 비교와 평균이 가능해진다.

신호 결합 규칙은 meta 분할에서 학습하고 test 분할에서 한 번만 평가한다.
백분위 정규화한 신호를 입력으로 오차의 백분위 순위를 예측하는 능선 회귀를
쓴다. meta 표본이 작은 물성이 있어 정칙화가 필요하다.

감사 대상은 지문 대표 모델의 예측이다. 분류에서 ChemBERTa보다 성능이 높고
(ROC-AUC 0.806 대 0.763) 배포 상황에 더 가깝기 때문이다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.linear_model import Ridge

RIDGE_ALPHA = 1.0

BASE_FEATURES = [
    "base__ad_knn__pct",
    "base__ad_density__pct",
    "base__disagreement__pct",
    "base__conformal_cb__pct",
    "base__conformal_fp__pct",
]
B_FEATURES = [
    "axis__fp_primary__B_combined__pct",
    "axis__cb_augmented__B_combined__pct",
]
A_FEATURES = ["axis__cb_augmented__A__pct"]

CONFIGURATIONS = {
    "기준": BASE_FEATURES,
    "기준+B": BASE_FEATURES + B_FEATURES,
    "기준+A": BASE_FEATURES + A_FEATURES,
    "기준+A+B": BASE_FEATURES + A_FEATURES + B_FEATURES,
    "B단독": B_FEATURES,
    "A단독": A_FEATURES,
}


def risk_coverage_aurc(risk_score: np.ndarray, error: np.ndarray) -> float:
    """위험 점수가 낮은 것부터 채택하며 누적 평균 오차의 평균을 구한다."""
    order = np.argsort(risk_score, kind="stable")
    ordered = error[order]
    return float(np.mean(np.cumsum(ordered) / np.arange(1, len(ordered) + 1)))


def normalized_aurc(risk_score: np.ndarray, error: np.ndarray) -> float:
    actual = risk_coverage_aurc(risk_score, error)
    oracle = risk_coverage_aurc(error, error)
    random = float(np.mean(error))
    if random - oracle < 1e-12:
        return float("nan")
    return (actual - oracle) / (random - oracle)


def evaluate_dataset(frame: pd.DataFrame) -> dict:
    meta = frame[frame["split"].eq("meta")]
    test = frame[frame["split"].eq("test")]
    error_meta = meta["abs_error_fp"].to_numpy(dtype=float)
    error_test = test["abs_error_fp"].to_numpy(dtype=float)

    # 결합 규칙의 목표는 오차의 백분위 순위다. 라벨 단위에 좌우되지 않게 한다.
    target = rankdata(error_meta) / len(error_meta)

    record = {
        "dataset": frame["dataset"].iloc[0],
        "task_type": frame["task_type"].iloc[0],
        "n_meta": len(meta),
        "n_test": len(test),
    }
    for name, features in CONFIGURATIONS.items():
        usable = [f for f in features if f in frame.columns and frame[f].nunique() > 1]
        if not usable:
            record[f"aurc__{name}"] = float("nan")
            continue
        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit(meta[usable].to_numpy(dtype=float), target)
        score = model.predict(test[usable].to_numpy(dtype=float))
        record[f"aurc__{name}"] = normalized_aurc(score, error_test)
        record[f"nfeat__{name}"] = len(usable)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="위험-커버리지 제거 실험")
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    evaluation_dir = Path(args.evaluation_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = sorted(
        path.name
        for path in evaluation_dir.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    )

    records = []
    for index, dataset in enumerate(datasets, 1):
        frame = pd.read_csv(evaluation_dir / dataset / "evaluation_signals.csv")
        record = evaluate_dataset(frame)
        records.append(record)
        print(
            f"[{index}/{len(datasets)}] {dataset}: "
            f"기준 {record['aurc__기준']:.4f} → 기준+B {record['aurc__기준+B']:.4f}",
            flush=True,
        )

    table = pd.DataFrame(records)
    table.to_csv(out_dir / "ablation_by_dataset.csv", index=False)

    # 가설 판정. 기준 모형 대비 감소량을 부호 순위 검정으로 본다.
    verdicts = []
    for name in ("기준+B", "기준+A", "기준+A+B"):
        delta = table[f"aurc__{name}"] - table["aurc__기준"]
        delta = delta.dropna()
        improved = int((delta < 0).sum())
        stat, pvalue = wilcoxon(delta) if len(delta) >= 6 else (np.nan, np.nan)
        verdicts.append(
            {
                "비교": f"기준 대비 {name}",
                "물성수": len(delta),
                "개선된물성": improved,
                "AURC변화_중앙": float(delta.median()),
                "AURC변화_평균": float(delta.mean()),
                "wilcoxon_p": float(pvalue),
            }
        )
    verdict = pd.DataFrame(verdicts)
    verdict.to_csv(out_dir / "ablation_verdict.csv", index=False)

    # 민감도 분석. 사전 지정한 결합 규칙이 결과를 좌우하는지 확인한다.
    # 판정은 위의 사전 지정 결과로 하고 이것은 보조 분석이다.
    robustness = []
    for label, alpha in [("Ridge alpha=1 (사전지정)", 1.0), ("Ridge alpha=10", 10.0),
                         ("Ridge alpha=100", 100.0), ("백분위 단순평균", None)]:
        values = {"기준": [], "기준+B": []}
        for dataset in datasets:
            frame = pd.read_csv(evaluation_dir / dataset / "evaluation_signals.csv")
            meta = frame[frame["split"].eq("meta")]
            test = frame[frame["split"].eq("test")]
            error_meta = meta["abs_error_fp"].to_numpy(dtype=float)
            error_test = test["abs_error_fp"].to_numpy(dtype=float)
            target = rankdata(error_meta) / len(error_meta)
            for name, features in [("기준", BASE_FEATURES), ("기준+B", BASE_FEATURES + B_FEATURES)]:
                usable = [f for f in features if f in frame.columns and frame[f].nunique() > 1]
                if alpha is None:
                    score = test[usable].to_numpy(dtype=float).mean(axis=1)
                else:
                    model = Ridge(alpha=alpha)
                    model.fit(meta[usable].to_numpy(dtype=float), target)
                    score = model.predict(test[usable].to_numpy(dtype=float))
                values[name].append(normalized_aurc(score, error_test))
        delta = pd.Series(values["기준+B"]) - pd.Series(values["기준"])
        delta = delta.dropna()
        robustness.append(
            {
                "결합규칙": label,
                "기준": float(np.mean(values["기준"])),
                "기준+B": float(np.mean(values["기준+B"])),
                "AURC변화_평균": float(delta.mean()),
                "개선된물성": int((delta < 0).sum()),
                "wilcoxon_p": float(wilcoxon(delta)[1]),
            }
        )
    pd.DataFrame(robustness).to_csv(out_dir / "ablation_robustness.csv", index=False)

    summary = {
        "audited_model": "fingerprint primary",
        "combination_rule": f"Ridge(alpha={RIDGE_ALPHA}) on meta, evaluated on test",
        "metric": "normalized AURC (0=oracle, 1=random), lower is better",
        "configurations": {k: v for k, v in CONFIGURATIONS.items()},
        "mean_normalized_aurc": {
            name: float(table[f"aurc__{name}"].mean()) for name in CONFIGURATIONS
        },
    }
    (out_dir / "ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print()
    print("=== 구성별 정규화 AURC (22종 평균, 낮을수록 좋음) ===")
    for name in CONFIGURATIONS:
        column = table[f"aurc__{name}"]
        print(f"  {name:10s} {column.mean():.4f}   (중앙 {column.median():.4f})")
    print()
    print("=== 가설 판정 ===")
    print(verdict.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
