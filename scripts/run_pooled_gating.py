#!/usr/bin/env python
"""물성을 통합해 축 활용 조건을 학습하는 결합 규칙 (사후 탐색).

문제. 지금까지 결합 규칙은 물성마다 따로 학습했다. meta 표본이 47개인 물성도
있어 규칙이 잡음을 학습하고, 축을 추가할수록 오히려 나빠졌다. 실제로 meta
크기와 개선폭의 상관이 +0.336이었다.

착안. 축이 유리한 조건 자체를 학습 대상으로 삼는다. 신호를 물성 안 백분위로
정규화했으므로 물성 간 비교가 가능하고, 목표도 물성 안 오차 순위이므로
22종을 한데 모아 하나의 규칙을 학습할 수 있다. 표본이 7,986개로 늘어난다.

축이 언제 쓸모 있는지를 사람이 규칙으로 정하지 않고 모델이 배우게 한다.
그래서 선형 모형 대신 경사부스팅을 쓰고, 각 축의 적용 여부를 알려주는
문맥 변수를 함께 넣는다. 예를 들어 입체 표기가 없는 분자는 B3 변형 수가
0이므로, 모델은 그 경우 B3 신호를 무시하는 규칙을 스스로 만들 수 있다.

평가. 물성 하나를 통째로 빼고 나머지 21종의 meta로 학습한 뒤, 빠진 물성의
test에서 평가한다. 같은 물성의 정보가 학습에 들어가지 않으므로 새로운 물성에
규칙을 적용하는 상황을 그대로 재현한다.

성격. 사전 지정되지 않은 사후 탐색이다. 전체 결과가 음성인 것을 확인한 뒤
설계했으므로 판정 근거가 아니라 후속 연구가 검정할 가설을 제안하는 자료다.
사전 지정 분석 결과를 대체하지 않고 병기한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score

REGRESSION_ERROR_QUANTILE = 0.80
MODELS = ("fp_primary", "cb_augmented")
AXES = ("A", "B1_tautomer", "B1_protonation", "B3_stereo", "B_combined")

BASE_FEATURES = [
    "base__ad_knn__pct",
    "base__ad_density__pct",
    "base__disagreement__pct",
    "base__conformal_cb__pct",
    "base__conformal_fp__pct",
]


def axis_signal_features() -> list[str]:
    return [f"axis__{m}__{a}__pct" for m in MODELS for a in AXES]


def context_features() -> list[str]:
    """축이 이 분자에 적용되는지를 알려주는 변수. 모델이 게이트를 학습하게 한다."""
    return [f"ctx__n_{a}" for a in AXES if a != "B_combined"] + ["ctx__is_classification"]


def load_dataset(dataset: str, evaluation_dir: Path, signals_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(evaluation_dir / dataset / "evaluation_signals.csv")
    signals = pd.read_csv(signals_dir / dataset / "ab_signals.csv")

    counts = signals[["row_uid"]].copy()
    for axis in AXES:
        column = f"fp_primary__{axis}__n"
        counts[f"ctx__n_{axis}"] = signals[column] if column in signals else 0
    frame = frame.merge(counts, on="row_uid", how="left")
    frame["ctx__is_classification"] = int(frame["task_type"].iloc[0] == "classification")

    truth = pd.to_numeric(frame["Y_final"]).to_numpy(dtype=float)
    if frame["task_type"].iloc[0] == "classification":
        predicted = (frame["pred_fp_primary"].to_numpy(dtype=float) >= 0.5).astype(int)
        frame["is_error"] = (predicted != truth.astype(int)).astype(int)
    else:
        error = frame["abs_error_fp"].to_numpy(dtype=float)
        threshold = np.quantile(
            error[frame["split"].eq("test")], REGRESSION_ERROR_QUANTILE
        )
        frame["is_error"] = (error >= threshold).astype(int)

    # 목표는 물성 안 오차 순위다. 물성마다 라벨 단위가 달라 원값은 통합할 수 없다.
    frame["error_rank"] = np.nan
    for split in ("meta", "test"):
        mask = frame["split"].eq(split).to_numpy()
        if mask.sum():
            frame.loc[mask, "error_rank"] = (
                rankdata(frame.loc[mask, "abs_error_fp"]) / mask.sum()
            )
    return frame


def normalized_aurc(risk: np.ndarray, error: np.ndarray) -> float:
    def aurc(score):
        ordered = error[np.argsort(score, kind="stable")]
        return float(np.mean(np.cumsum(ordered) / np.arange(1, len(ordered) + 1)))

    oracle, random = aurc(error), float(np.mean(error))
    if random - oracle < 1e-12:
        return float("nan")
    return (aurc(risk) - oracle) / (random - oracle)


def main() -> int:
    parser = argparse.ArgumentParser(description="물성 통합 게이팅 결합 규칙")
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--signals-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    evaluation_dir = Path(args.evaluation_dir)
    signals_dir = Path(args.signals_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = sorted(
        p.name for p in evaluation_dir.iterdir()
        if p.is_dir() and not p.name.startswith("_")
    )
    frames = {d: load_dataset(d, evaluation_dir, signals_dir) for d in datasets}

    configs = {
        "기준": BASE_FEATURES,
        "기준+축": BASE_FEATURES + axis_signal_features(),
        "기준+축+문맥": BASE_FEATURES + axis_signal_features() + context_features(),
    }

    records = []
    for held_out in datasets:
        # 분류와 회귀는 신호의 성격이 달라 한 규칙으로 묶으면 서로 타협한다.
        # 같은 과제 유형의 물성만 모아 학습한다.
        task = frames[held_out]["task_type"].iloc[0]
        train = pd.concat(
            [
                f[f["split"].eq("meta")]
                for d, f in frames.items()
                if d != held_out and f["task_type"].iloc[0] == task
            ],
            ignore_index=True,
        )
        test = frames[held_out]
        test = test[test["split"].eq("test")]
        error = test["abs_error_fp"].to_numpy(dtype=float)
        labels = test["is_error"].to_numpy(dtype=int)

        record = {"dataset": held_out, "task_type": test["task_type"].iloc[0],
                  "n_test": len(test), "n_train_pooled": len(train)}
        for name, features in configs.items():
            usable = [f for f in features if f in train.columns]
            model = HistGradientBoostingRegressor(
                max_depth=4, max_iter=300, learning_rate=0.06,
                min_samples_leaf=40, l2_regularization=1.0, random_state=0,
            )
            model.fit(train[usable].to_numpy(dtype=float),
                      train["error_rank"].to_numpy(dtype=float))
            score = model.predict(test[usable].to_numpy(dtype=float))
            record[f"aurc__{name}"] = normalized_aurc(score, error)
            record[f"auprc__{name}"] = (
                float(average_precision_score(labels, score))
                if labels.min() != labels.max() else np.nan
            )
        records.append(record)
        print(
            f"{held_out:32s} 기준 {record['auprc__기준']:.3f} → "
            f"+축 {record['auprc__기준+축']:.3f} → +문맥 {record['auprc__기준+축+문맥']:.3f}",
            flush=True,
        )

    table = pd.DataFrame(records)
    table.to_csv(out_dir / "pooled_gating_by_dataset.csv", index=False)

    print()
    print("=== 물성 통합 규칙, 물성 단위 교차검증 ===")
    for metric, arrow in [("auprc", "높을수록 좋음"), ("aurc", "낮을수록 좋음")]:
        print(f"  [{metric.upper()} {arrow}]")
        for name in configs:
            column = table[f"{metric}__{name}"].dropna()
            print(f"    {name:14s} 평균 {column.mean():.4f}   중앙 {column.median():.4f}")
        base = table[f"{metric}__기준"]
        sign = 1 if metric == "auprc" else -1
        for name in ("기준+축", "기준+축+문맥"):
            delta = (table[f"{metric}__{name}"] - base).dropna()
            print(f"      {name:14s} 변화 {delta.mean():+.4f}  "
                  f"개선 {int((delta * sign > 0).sum())}/{len(delta)}종")
        print()

    (out_dir / "pooled_gating_summary.json").write_text(
        json.dumps(
            {
                "protocol": "leave-one-dataset-out; 21종 meta로 학습, 남긴 물성 test로 평가",
                "model": "HistGradientBoostingRegressor(max_depth=4, max_iter=300)",
                "n_pooled_meta": int(table["n_train_pooled"].iloc[0]),
                "mean": {
                    f"{m}__{n}": float(table[f"{m}__{n}"].mean())
                    for m in ("auprc", "aurc") for n in configs
                },
                "status": "사후 탐색. 사전 지정 분석을 대체하지 않는다.",
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
