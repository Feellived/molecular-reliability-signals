#!/usr/bin/env python
"""계획서에 지정됐으나 아직 수행하지 않은 분석을 채운다.

  6.1절  학습 골격으로부터의 거리 구간별 A와 B의 변화
  6.2절  상위 10퍼센트 집합의 겹침 비율 및 자카드 지수
  6.3절  AUROC 보조 지표, 고위험 집단의 조건부 미달률
  7.3절  신뢰성 신호 벤치마크 표

거리 구간 분석은 결과가 어느 쪽으로 나와도 얻는 것이 있다. 거리가 멀수록
축이 커지면 6.1절의 특성화가 채워지고, 관계가 약하면 축이 화학 공간상의
위치와 다른 원천에서 나온다는 7.1절의 예상이 확인되어 비중복성 주장이
오히려 강해진다.

미달은 오차 탐지 라벨과 같게 정의한다. 회귀는 상위 20퍼센트 오차, 분류는
오분류다. 조건부 미달률은 신호가 고위험으로 지목한 상위 20퍼센트 안에서의
미달 비율이며, 전체 미달률과의 비를 함께 본다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score

TOP_FRACTION = 0.10      # 6.2절 상위 10퍼센트 겹침
FLAG_FRACTION = 0.20     # 6.3절 고위험 집단
BASE_SIGNALS = {
    "적용가능도메인 이웃": "base__ad_knn",
    "적용가능도메인 밀도": "base__ad_density",
    "모델 불일치": "base__disagreement",
    "컨포멀 ChemBERTa": "base__conformal_cb",
    "컨포멀 지문": "base__conformal_fp",
}
AXIS_SIGNALS = {
    "표현 불안정성 A": "axis__cb_augmented__A",
    "B1 호변이성질체": "axis__fp_primary__B1_tautomer",
    "B1 양성자화": "axis__fp_primary__B1_protonation",
    "B3 입체 표기": "axis__fp_primary__B3_stereo",
    "통합 B (조건부)": "cond_B__fp_primary__std",
}
COMBINED = "결합 점수"


def normalized_aurc(score, error) -> float:
    curve = lambda x: float(np.mean(
        np.cumsum(error[np.argsort(x, kind="stable")]) / np.arange(1, len(error) + 1)))
    oracle, random = curve(error), float(np.mean(error))
    return np.nan if random - oracle < 1e-12 else (curve(score) - oracle) / (random - oracle)


def shortfall_label(frame) -> np.ndarray:
    """미달 정의. 회귀는 상위 20퍼센트 오차, 분류는 오분류."""
    error = frame["abs_error_fp"].to_numpy(float)
    if frame["task_type"].iloc[0] == "classification":
        predicted = (frame["pred_fp_primary"].to_numpy(float) >= 0.5).astype(int)
        return (predicted != pd.to_numeric(frame["Y_final"]).astype(int)).to_numpy().astype(int)
    return (error >= np.quantile(error, 1 - FLAG_FRACTION)).astype(int)


def partial_spearman(signal, error, controls) -> float:
    """기준선 신호를 통제한 뒤 남는 순위 상관."""
    design = np.column_stack([np.ones(len(signal))] + [rankdata(c) for c in controls])
    resid = lambda y: y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    a, b = resid(rankdata(signal)), resid(rankdata(error))
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def score_signal(values, error, label) -> dict:
    if np.nanstd(values) < 1e-12:
        return {}
    return {
        "오차 순위상관": float(spearmanr(values, error).statistic),
        "AUPRC": float(average_precision_score(label, values)) if label.min() != label.max() else np.nan,
        "AUROC": float(roc_auc_score(label, values)) if label.min() != label.max() else np.nan,
        "정규화 AURC": normalized_aurc(values, error),
    }


def process(dataset: str, evaluation_dir: Path) -> tuple[list[dict], list[dict], pd.DataFrame]:
    frame = pd.read_csv(evaluation_dir / dataset / "evaluation_signals.csv")
    meta, test = frame[frame.split.eq("meta")], frame[frame.split.eq("test")]
    error = test["abs_error_fp"].to_numpy(float)
    label = shortfall_label(test)
    controls = [test[c].to_numpy(float) for c in BASE_SIGNALS.values() if c in test]

    # 결합 점수: 기준선 다섯에 축을 더해 meta에서 학습한 규칙
    features = [c for c in (*BASE_SIGNALS.values(), *AXIS_SIGNALS.values())
                if f"{c}__pct" in frame and frame[f"{c}__pct"].nunique() > 1]
    columns = [f"{c}__pct" for c in features]
    combined = Ridge(alpha=1.0).fit(
        meta[columns].to_numpy(float), rankdata(meta["abs_error_fp"]) / len(meta)
    ).predict(test[columns].to_numpy(float))

    rows, top_sets, cut = [], {}, max(1, int(round(len(test) * TOP_FRACTION)))
    flag_n = max(1, int(round(len(test) * FLAG_FRACTION)))
    catalogue = {**BASE_SIGNALS, **AXIS_SIGNALS, COMBINED: None}
    for name, column in catalogue.items():
        values = combined if column is None else (
            test[column].to_numpy(float) if column in test else None)
        if values is None:
            continue
        metrics = score_signal(values, error, label)
        if not metrics:
            continue
        order = np.argsort(-values, kind="stable")
        top_sets[name] = set(test["row_uid"].to_numpy()[order[:cut]])
        flagged = label[order[:flag_n]]
        metrics["조건부 미달률"] = float(flagged.mean())
        metrics["미달률 배수"] = float(flagged.mean() / label.mean()) if label.mean() > 0 else np.nan
        if name in AXIS_SIGNALS or name == COMBINED:
            metrics["통제 후 부분상관"] = partial_spearman(values, error, controls)
        rows.append({"dataset": dataset, "signal": name, **metrics})

    # 6.2절 겹침: 상위 10퍼센트 집합의 겹침 비율과 자카드
    overlaps = []
    for a in top_sets:
        for b in top_sets:
            if a >= b:
                continue
            inter = len(top_sets[a] & top_sets[b])
            union = len(top_sets[a] | top_sets[b])
            overlaps.append({"dataset": dataset, "signal_a": a, "signal_b": b,
                             "겹침 비율": inter / cut, "자카드": inter / union if union else np.nan})

    # 6.1절 골격 거리: 적용가능도메인 이웃 신호가 거리 대리변수다 (부호 반전 완료)
    distance = test["base__ad_knn"].to_numpy(float)
    bins = pd.qcut(rankdata(distance, method="ordinal"), 4, labels=False)
    dist_rows = []
    for index in range(4):
        mask = bins == index
        record = {"dataset": dataset, "거리 분위": index + 1, "n": int(mask.sum())}
        for name, column in AXIS_SIGNALS.items():
            if column in test and test[column].nunique() > 1:
                pct = test[column].rank(pct=True).to_numpy()
                record[name] = float(pct[mask].mean())
        record["정규화 오차"] = float(error[mask].mean() / error.mean())
        dist_rows.append(record)
    return rows, overlaps, pd.DataFrame(dist_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="계획서 미실행 분석 보충")
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    evaluation_dir, out_dir = Path(args.evaluation_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = sorted(p.name for p in evaluation_dir.iterdir()
                      if p.is_dir() and not p.name.startswith("_"))

    bench, overlap, distance = [], [], []
    for dataset in datasets:
        rows, overlaps, dist = process(dataset, evaluation_dir)
        bench += rows; overlap += overlaps; distance.append(dist)
        print(f"  {dataset:32s} 신호 {len(rows):2d}종", flush=True)

    bench = pd.DataFrame(bench); overlap = pd.DataFrame(overlap)
    distance = pd.concat(distance, ignore_index=True)
    bench.to_csv(out_dir / "benchmark_by_dataset.csv", index=False)
    overlap.to_csv(out_dir / "overlap_by_dataset.csv", index=False)
    distance.to_csv(out_dir / "scaffold_distance_by_dataset.csv", index=False)

    order = [*BASE_SIGNALS, *AXIS_SIGNALS, COMBINED]
    table = (bench.groupby("signal").mean(numeric_only=True)
             .reindex([s for s in order if s in set(bench.signal)]))
    table["물성 수"] = bench.groupby("signal").size()
    table.to_csv(out_dir / "benchmark_table.csv")
    print("\n=== 7.3절 신뢰성 신호 벤치마크 표 (22종 평균) ===")
    print(table.round(4).to_string())

    print("\n=== 6.1절 학습 골격 거리 구간별 (거리 분위 1=가까움) ===")
    axis_cols = [c for c in AXIS_SIGNALS if c in distance.columns]
    print(distance.groupby("거리 분위")[[*axis_cols, "정규화 오차"]].mean().round(4).to_string())

    print("\n=== 6.2절 상위 10퍼센트 집합 겹침 (22종 중앙값) ===")
    pivot = (overlap.groupby(["signal_a", "signal_b"])[["겹침 비율", "자카드"]]
             .median().reset_index())
    ours = set(AXIS_SIGNALS) | {COMBINED}
    cross = pivot[(pivot.signal_a.isin(ours) & ~pivot.signal_b.isin(ours))
                  | (~pivot.signal_a.isin(ours) & pivot.signal_b.isin(ours))]
    within_base = pivot[~pivot.signal_a.isin(ours) & ~pivot.signal_b.isin(ours)]
    print(f"  기준선끼리          겹침 {within_base['겹침 비율'].median():.3f}  "
          f"자카드 {within_base['자카드'].median():.3f}")
    print(f"  우리 신호 대 기준선  겹침 {cross['겹침 비율'].median():.3f}  "
          f"자카드 {cross['자카드'].median():.3f}")
    pivot.to_csv(out_dir / "overlap_summary.csv", index=False)

    (out_dir / "plan_completion_summary.json").write_text(json.dumps({
        "overlap_within_baseline_jaccard": float(within_base["자카드"].median()),
        "overlap_ours_vs_baseline_jaccard": float(cross["자카드"].median()),
        "distance_bins": distance.groupby("거리 분위")[axis_cols].mean().to_dict(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
