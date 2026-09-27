#!/usr/bin/env python
"""지문 모델 기반 적응형 컨포멀 예측을 산출한다 (연구계획서 5.6절).

담당2의 컨포멀은 회귀와 분류 모두 점추정과 척도를 ChemBERTa에서 가져온다.
기준 모형 전체가 SMILES 언어 모델 하나에 걸리는 구조인데, 이는 적용가능도메인을
언어 모델에서 분리한 5.6절의 취지와 어긋난다. 분류에서는 지문 대표 모델보다
성능이 낮은 모델 위에 기준선이 서는 문제도 있다.

그래서 같은 절차를 지문 모델 재료로 한 번 더 수행해 언어 모델에 의존하지 않는
기준선을 추가한다. 새로 학습하는 것은 없다. 컨포멀은 이미 산출된 예측값을
보정 분할에서 후처리하는 절차이므로 담당2의 signals 파일만 있으면 된다.

  회귀   정규화 분할 컨포멀. 척도는 시드 간 표준편차 std_fp_primary에
         하한을 더한 값이다. 하한은 담당2와 같이 보정 분할의 25백분위수를 쓴다.
         척도가 0에 가까운 분자에서 구간이 발산하는 것을 막기 위한 장치다.
  분류   무작위화 APS. 확률은 pred_fp_primary를 쓴다.

점추정과 척도는 담당2의 원본 예측을 그대로 쓴다. 기준 모형은 팀이 공식으로
보고하는 모델이어야 하고, 여기서는 새 입력을 채점할 일이 없어 재적합이
필요하지 않기 때문이다. A·B 신호 산출에 쓴 재적합 모델과는 용도가 다르다.

무작위화는 row_uid 해시로 결정하므로 재실행해도 같은 결과가 나온다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ALPHA = 0.1
CALIB_SPLIT = "calib"
RANDOMIZATION_TAG = "mist-fp-aps-v1"


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """분할 컨포멀의 유한표본 보정 분위수."""
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def _uniform_from_uid(row_uid: str) -> float:
    """row_uid에서 결정적으로 0과 1 사이 난수를 만든다."""
    digest = hashlib.sha256(f"{RANDOMIZATION_TAG}:{row_uid}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _regression(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    truth = pd.to_numeric(frame["Y_final"]).to_numpy(dtype=float)
    point = frame["pred_fp_primary"].to_numpy(dtype=float)
    raw_scale = frame["std_fp_primary"].to_numpy(dtype=float)

    calib = frame["split"].eq(CALIB_SPLIT).to_numpy()
    floor = float(np.quantile(raw_scale[calib], 0.25))
    if not np.isfinite(floor) or floor <= 0:
        floor = float(np.std(truth[calib])) * 0.01 or 1e-6
    scale = raw_scale + floor

    scores = np.abs(truth[calib] - point[calib]) / scale[calib]
    qhat = _conformal_quantile(scores, ALPHA)

    width = qhat * scale
    out = pd.DataFrame(
        {
            "row_uid": frame["row_uid"],
            "fp_conformal_scale": scale,
            "fp_conformal_lower": point - width,
            "fp_conformal_upper": point + width,
            "fp_conformal_width": 2 * width,
            "fp_conformal_qhat": qhat,
        }
    )
    covered = (truth >= out["fp_conformal_lower"]) & (truth <= out["fp_conformal_upper"])
    test = frame["split"].eq("test").to_numpy()
    meta = {
        "method": "normalized_split_conformal",
        "scale": "std_fp_primary + calib_q25_floor",
        "scale_floor": floor,
        "qhat": qhat,
        "calib_coverage": float(covered[calib].mean()),
        "test_coverage": float(covered[test].mean()),
        "test_mean_width": float(out.loc[test, "fp_conformal_width"].mean()),
    }
    return out, meta


def _classification(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    truth = pd.to_numeric(frame["Y_final"]).to_numpy(dtype=int)
    p1 = np.clip(frame["pred_fp_primary"].to_numpy(dtype=float), 0.0, 1.0)
    probs = np.column_stack([1.0 - p1, p1])
    noise = np.array([_uniform_from_uid(u) for u in frame["row_uid"]])

    # APS 점수: 확률이 높은 클래스부터 누적하여 참 클래스까지 더하고,
    # 참 클래스 자신의 확률 일부를 무작위로 덜어낸다.
    order = np.argsort(-probs, axis=1)
    ranks = np.argsort(order, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cumulative = np.cumsum(sorted_probs, axis=1)
    true_rank = ranks[np.arange(len(truth)), truth]
    true_prob = probs[np.arange(len(truth)), truth]
    scores = cumulative[np.arange(len(truth)), true_rank] - noise * true_prob

    calib = frame["split"].eq(CALIB_SPLIT).to_numpy()
    qhat = _conformal_quantile(scores[calib], ALPHA)

    # 예측 집합: 같은 규칙으로 각 클래스의 점수를 구해 qhat 이하인 것만 담는다.
    class_scores = np.empty_like(probs)
    for label in (0, 1):
        rank = ranks[:, label]
        class_scores[:, label] = (
            cumulative[np.arange(len(truth)), rank] - noise * probs[:, label]
        )
    included = class_scores <= qhat

    out = pd.DataFrame(
        {
            "row_uid": frame["row_uid"],
            "fp_aps_prediction_set": [
                "[" + ",".join(str(c) for c in (0, 1) if row[c]) + "]" for row in included
            ],
            "fp_aps_set_size": included.sum(axis=1),
            "fp_aps_true_score": scores,
            "fp_aps_qhat": qhat,
        }
    )
    covered = included[np.arange(len(truth)), truth]
    test = frame["split"].eq("test").to_numpy()
    meta = {
        "method": "randomized_APS",
        "randomization": f"SHA256(row_uid, {RANDOMIZATION_TAG}), reproducible",
        "qhat": qhat,
        "calib_coverage": float(covered[calib].mean()),
        "test_coverage": float(covered[test].mean()),
        "test_mean_set_size": float(out.loc[test, "fp_aps_set_size"].mean()),
        "test_empty_set_rate": float((out.loc[test, "fp_aps_set_size"] == 0).mean()),
    }
    return out, meta


def process_dataset(dataset: str, signals_dir: Path, out_dir: Path) -> dict:
    frame = pd.read_csv(signals_dir / dataset / "role2_signals.csv")
    task_type = frame["task_type"].iloc[0]

    out, meta = (
        _classification(frame) if task_type == "classification" else _regression(frame)
    )
    out.insert(1, "dataset", dataset)
    out.insert(2, "split", frame["split"])

    dataset_out = out_dir / dataset
    dataset_out.mkdir(parents=True, exist_ok=True)
    out.to_csv(dataset_out / "fp_conformal.csv", index=False)

    record = {
        "dataset": dataset,
        "task_type": task_type,
        "alpha": ALPHA,
        "calibration_split": CALIB_SPLIT,
        "point_estimate": "pred_fp_primary",
        "primary_model": frame["fp_primary_model"].iloc[0],
        "n_calib": int(frame["split"].eq(CALIB_SPLIT).sum()),
        **meta,
    }
    (dataset_out / "fp_conformal_metadata.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="지문 기반 적응형 컨포멀 산출")
    parser.add_argument("--signals-dir", required=True, help="담당2 outputs 최상위")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()

    signals_dir = Path(args.signals_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets or sorted(
        path.name
        for path in signals_dir.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    )

    records = []
    for index, dataset in enumerate(datasets, 1):
        record = process_dataset(dataset, signals_dir, out_dir)
        records.append(record)
        print(
            f"[{index}/{len(datasets)}] {dataset} ({record['task_type'][:5]}): "
            f"test 포함률 {record['test_coverage']:.3f}, 대표 {record['primary_model']}",
            flush=True,
        )

    frame = pd.DataFrame(records)
    summary_dir = out_dir / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(summary_dir / "fp_conformal_summary.csv", index=False)

    print()
    for task in ("classification", "regression"):
        subset = frame[frame["task_type"].eq(task)]
        if subset.empty:
            continue
        print(
            f"{task:15s} {len(subset):2d}종  test 포함률 평균 "
            f"{subset['test_coverage'].mean():.4f} (목표 {1 - ALPHA:.2f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
