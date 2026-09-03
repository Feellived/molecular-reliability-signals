#!/usr/bin/env python
"""컨포멀 척도를 train 폴드 외 예측으로 적합해 다시 판정한다.

run_conformal_scale.py는 척도를 meta 분할에서 교차적합했다. 변형을 meta와
test에만 만들었기 때문인데, meta가 66행인 물성이 있어 적합 표본이 너무 작았다.
축이 척도로 쓸모없어서 진 것인지 적합할 데이터가 없어서 진 것인지 구분이
되지 않는 상태였다.

지금은 train 행의 폴드 외 예측이 있다. 자기 폴드를 제외하고 학습한 모형의
예측이므로 train 행에서도 정직한 오차를 얻는다. 척도를 여기서 적합하면
표본이 물성당 수백 행으로 늘고, meta는 보정에만 쓴다. 적합과 보정과 시험이
서로 다른 분할이 되어 구조도 더 깨끗하다.

  적합   train (폴드 외 예측)     척도 함수를 학습
  보정   meta                    qhat 산출
  시험   test                    커버리지와 구간 폭 측정

구성은 앞과 같다. S1은 s0만, S3는 s0에 축 확장 통계를 더한다. 둘의 차이만이
축의 기여다. 여기서도 축이 폭을 줄이지 못하면 표본 부족이 원인이라는 설명은
성립하지 않는다.

적합 분할과 적용 분할은 서로 다른 모형이 만든 예측이다. 폴드 외 모형과
최종 모형은 예측 폭도 오차 수준도 다르므로, 원값을 그대로 쓰면 분포가
어긋나 척도가 발산한다. 두 가지로 막는다. 목표를 절대 오차 대신 기준선
척도 대비 비율로 두어 수준 차이를 약분하고, 축 특징은 분할 안의 백분위로
바꾸어 눈금을 맞춘다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

ALPHA = 0.1
EPS = 1e-6
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
MODELS = {"fp_primary": "pred_fp_primary", "cb_augmented": "pred_chemberta_augmented"}
STATS = ("std", "max_dev", "rel_std", "shift")


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    return float(np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))


def statistics(values: np.ndarray, parent: float, scale: float) -> dict:
    sample = np.append(values, parent)
    std = float(np.std(sample, ddof=0)) if len(sample) > 1 else 0.0
    deviation = np.abs(values - parent) if len(values) else np.array([0.0])
    return {"std": std, "max_dev": float(deviation.max()),
            "shift": float(values.mean() - parent) if len(values) else 0.0,
            "rel_std": float(std / scale) if scale > 1e-9 else 0.0}


def rich_frame(variants, origin, axes) -> pd.DataFrame:
    """meta·test와 같은 규칙으로 train 행의 확장 통계를 만든다."""
    scales = {k: float(origin[c].std()) for k, c in MODELS.items()}
    groups = {"A": ("A",)}
    if axes:
        groups["B"] = tuple(axes)
    columns = [f"rich__{m}__{g}__{s}" for m in MODELS for g in ("A", "B") for s in STATS]

    rows = []
    for row_uid, group in variants.groupby("parent_row_uid"):
        if row_uid not in origin.index:
            continue
        record = {"row_uid": row_uid}
        for model_key, column in MODELS.items():
            parent = float(origin.at[row_uid, column])
            for group_name, group_axes in groups.items():
                values = group.loc[group["axis"].isin(group_axes), column].to_numpy(float)
                for name, value in statistics(values, parent, scales[model_key]).items():
                    record[f"rich__{model_key}__{group_name}__{name}"] = value
        rows.append(record)
    frame = pd.DataFrame(rows).reindex(columns=["row_uid"] + columns).fillna(0.0)
    frame["row_uid"] = [r["row_uid"] for r in rows]
    return frame


def fit_apply(x_fit, y_fit, blocks):
    """기준선 척도 대비 배율을 예측한다. 예측은 적합 표본 범위로 자른다."""
    scaler = StandardScaler().fit(x_fit)
    model = RidgeCV(alphas=RIDGE_ALPHAS).fit(scaler.transform(x_fit), y_fit)
    lo, hi = np.percentile(y_fit, [1, 99])
    return [np.exp(np.clip(model.predict(scaler.transform(x)), lo, hi)) for x in blocks]


def percentiles(frame, columns) -> np.ndarray:
    """분할 안의 백분위. 분할마다 모형이 달라 원값 눈금이 맞지 않는다."""
    if not columns:
        return np.empty((len(frame), 0))
    return np.column_stack([pd.Series(np.asarray(frame[c], dtype=float))
                            .rank(pct=True, method="average").to_numpy() for c in columns])


def evaluate(error_calib, error_test, scale_calib, scale_test, wobble_test):
    qhat = conformal_quantile(error_calib / scale_calib, ALPHA)
    covered = error_test <= qhat * scale_test
    width = 2.0 * qhat * scale_test
    cut = np.nanquantile(wobble_test, 0.75)
    high = wobble_test > cut if (wobble_test > cut).any() else wobble_test >= cut
    return {"qhat": float(qhat), "test_coverage": float(covered.mean()),
            "test_mean_width": float(width.mean()),
            "high_wobble_coverage": float(covered[high].mean()) if high.any() else np.nan,
            "low_wobble_coverage": float(covered[~high].mean()) if (~high).any() else np.nan}


def process(dataset, scores_dir, evaluation_dir, role2_dir, pipeline_dir, allowance, decision):
    frame = pd.read_csv(evaluation_dir / dataset / "evaluation_signals.csv")
    if frame["task_type"].iloc[0] != "regression":
        return None
    frame = frame.merge(pd.read_csv(role2_dir / dataset / "role2_signals.csv",
                                    usecols=["row_uid", "std_fp_primary"]), on="row_uid")

    axes = [a for a in ("B1_tautomer", "B1_protonation")
            if allowance.at[dataset, a] == "허용" and decision.at[dataset, f"use_{a}"] == "사용"]
    if decision.at[dataset, "use_B3_stereo"] == "사용":
        axes.append("B3_stereo")

    oof = scores_dir / "train_oof_fingerprint" / dataset
    oof_cb = scores_dir / "train_oof_chemberta" / dataset
    origin = pd.read_csv(oof / "origin_predictions_oof.csv").merge(
        pd.read_csv(oof_cb / "origin_predictions_chemberta_oof.csv")
        .drop(columns=["dataset", "split", "cv_fold"], errors="ignore"), on="row_uid")
    variants = pd.read_csv(oof / "variant_predictions_fp_oof.csv").merge(
        pd.read_csv(oof_cb / "variant_predictions_chemberta_oof.csv")
        .drop(columns=["dataset", "axis", "split", "cv_fold", "parent_row_uid"],
              errors="ignore"), on="variant_uid")

    truth = pd.read_csv(pipeline_dir / dataset / "splits.csv", usecols=["row_uid", "Y_final"])
    train = origin.merge(truth, on="row_uid").merge(
        rich_frame(variants, origin.set_index("row_uid"), axes), on="row_uid", how="inner")
    if len(train) < 50:
        return None

    feats = [c for c in train.columns if c.startswith("rich__")]
    feats = [c for c in feats if train[c].nunique() > 1 and c in frame.columns]

    floor = float(np.quantile(frame.loc[frame.split.eq("meta"), "std_fp_primary"], 0.25)) or 1e-6
    base_tr = train["std_fp_primary"].to_numpy(float) + floor
    err_tr = np.abs(train["Y_final"] - train["pred_fp_primary"]).to_numpy(float)
    # 목표는 절대 오차가 아니라 기준선 척도 대비 배율이다. 폴드 외 모형과
    # 최종 모형의 오차 수준 차이가 약분된다.
    y_tr = np.log((err_tr + EPS) / base_tr)

    calib, test = frame[frame.split.eq("meta")], frame[frame.split.eq("test")]
    base_ca = calib["std_fp_primary"].to_numpy(float) + floor
    base_te = test["std_fp_primary"].to_numpy(float) + floor
    err_ca = np.abs(calib["Y_final"] - calib["pred_fp_primary"]).to_numpy(float)
    err_te = np.abs(test["Y_final"] - test["pred_fp_primary"]).to_numpy(float)
    wob_te = (test["rich__fp_primary__B__rel_std"].fillna(0)
              + test["rich__fp_primary__A__rel_std"].fillna(0)).to_numpy(float)

    record = {"dataset": dataset, "n_train_fit": len(train), "n_calib": len(calib),
              "n_test": len(test), "n_axis_features": len(feats)}
    for key, value in evaluate(err_ca, err_te, base_ca, base_te, wob_te).items():
        record[f"S0_기준선__{key}"] = value

    # S1은 특징이 없어 배율이 상수가 되고 폭이 기준선과 정확히 같아진다.
    # S2는 기준선 척도 자신을 백분위로 다시 넣어 재눈금 여지를 준다.
    # 축의 기여는 S3와 S2의 차이다.
    for name, cols, use_base in (("S1_적합_축없음", [], False),
                                 ("S2_기준선_재눈금", [], True),
                                 ("S3_적합_축확장", feats, True)):
        block = lambda f, b: np.column_stack(
            [np.ones(len(f))]
            + ([pd.Series(b).rank(pct=True, method="average").to_numpy()] if use_base else [])
            + [percentiles(f, cols)])
        x_tr, x_ca, x_te = block(train, base_tr), block(calib, base_ca), block(test, base_te)
        mult_ca, mult_te = fit_apply(x_tr, y_tr, [x_ca, x_te])
        for key, value in evaluate(err_ca, err_te,
                                   np.maximum(base_ca * mult_ca, floor * 1e-3),
                                   np.maximum(base_te * mult_te, floor * 1e-3),
                                   wob_te).items():
            record[f"{name}__{key}"] = value
    return record


def main() -> int:
    p = argparse.ArgumentParser(description="train 폴드 외 예측으로 컨포멀 척도 재판정")
    for flag in ("--scores-dir", "--evaluation-dir", "--role2-dir", "--pipeline-dir",
                 "--reports-dir", "--out-dir"):
        p.add_argument(flag, required=True)
    args = p.parse_args()

    scores_dir, evaluation_dir = Path(args.scores_dir), Path(args.evaluation_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    reports = Path(args.reports_dir)
    allowance = pd.read_csv(reports / "06_transformation_allowance_revised.csv").set_index("dataset")
    decision = pd.read_csv(reports / "07_axis_decision.csv").set_index("dataset")

    records = []
    for dataset in sorted(d.name for d in evaluation_dir.iterdir()
                          if d.is_dir() and not d.name.startswith("_")):
        try:
            record = process(dataset, scores_dir, evaluation_dir, Path(args.role2_dir),
                             Path(args.pipeline_dir), allowance, decision)
        except (FileNotFoundError, KeyError) as exc:
            print(f"  {dataset:28s} 건너뜀 ({type(exc).__name__})", flush=True); continue
        if record:
            records.append(record)
            print(f"  {dataset:28s} 적합 {record['n_train_fit']:5d}행  "
                  f"축 특징 {record['n_axis_features']:2d}개", flush=True)

    frame = pd.DataFrame(records)
    frame.to_csv(out_dir / "conformal_scale_oof_by_dataset.csv", index=False)

    print(f"\n회귀 {len(frame)}종  목표 커버리지 {1 - ALPHA:.2f}  "
          f"적합 표본 중앙값 {int(frame.n_train_fit.median())}행 (기존 meta 교차적합 "
          f"{int(frame.n_calib.median())}행)\n")
    print(f"{'구성':16s}{'커버리지':>9s}{'평균 폭':>10s}{'S2 대비':>9s}"
          f"{'고흔들림':>10s}{'저흔들림':>10s}")
    ref = frame["S2_기준선_재눈금__test_mean_width"]
    for name in ("S0_기준선", "S1_적합_축없음", "S2_기준선_재눈금", "S3_적합_축확장"):
        print(f"{name:16s}{frame[f'{name}__test_coverage'].mean():9.4f}"
              f"{frame[f'{name}__test_mean_width'].mean():10.4f}"
              f"{(frame[f'{name}__test_mean_width'] / ref).mean():9.4f}"
              f"{frame[f'{name}__high_wobble_coverage'].mean():10.4f}"
              f"{frame[f'{name}__low_wobble_coverage'].mean():10.4f}")

    rng = np.random.default_rng(42)
    gain = 1.0 - frame["S3_적합_축확장__test_mean_width"] / ref
    boot = [rng.choice(gain, len(gain), replace=True).mean() for _ in range(10000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n축 추가 폭 감소  {gain.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  "
          f"개선 {int((gain > 0).sum())}/{len(gain)}종")
    print(frame[["dataset", "n_train_fit", "n_axis_features"]].assign(
        폭감소=gain.round(4)).to_string(index=False))
    (out_dir / "conformal_scale_oof_summary.json").write_text(json.dumps(
        {"mean_width_reduction": float(gain.mean()), "ci": [float(lo), float(hi)],
         "n_improved": int((gain > 0).sum()), "n_total": len(gain)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
