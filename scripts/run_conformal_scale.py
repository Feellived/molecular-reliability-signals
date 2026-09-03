#!/usr/bin/env python
"""축 분산을 컨포멀 척도 함수로 쓴다 (연구계획서 5.6절 확장).

지금까지 축 신호는 위험 순위를 매기는 데만 썼다. 순위 개선은 기존 신호와
성능을 겨루는 구도가 되고, 우리 신호는 보조 신호 수준이라 그 구도에서
얻을 수 있는 것이 크지 않다.

컨포멀은 다른 자리를 내준다. 정규화 분할 컨포멀은 분자마다 척도 s(x)를
정해 구간을 s(x)에 비례해 넓히는데, 커버리지 보장은 s를 무엇으로 쓰든
그대로 성립한다. s가 좋으면 같은 커버리지를 더 좁은 구간으로 달성한다.
즉 척도 함수는 보장을 건드리지 않고 효율만 바꾸는 자리이고, 보조 신호가
들어갈 자리로는 순위 경쟁보다 이쪽이 맞는다.

비교 구성

  S0  기존 기준선. 척도는 시드 간 표준편차에 보정 하한을 더한 값
  S1  s0만 특징으로 쓴 적합 척도. 적합 절차 자체의 효과를 분리한다
  S2  s0에 축 요약 통계 둘을 더한 적합 척도 (A와 B의 상대 흩어짐)
  S3  s0에 축 확장 통계 전부를 더한 적합 척도
  S4  s0에 축 흔들림 백분위를 곱하는 모수 하나짜리 척도. 보정 분할이
      66행뿐인 물성이 있어 특징 17개를 적합하면 분산이 커진다. 계수 하나만
      두어 축이 척도에 기여할 여지를 가장 유리한 조건에서 확인한다
  S5  축 흔들림만으로 만든 척도. s0을 빼서 축 단독의 척도 정보를 본다

S1과 S2·S3의 차이만이 축의 기여다. S0과 S1을 함께 두는 이유는 적합
척도가 원래 좋아서 생긴 개선을 축 덕분이라고 오독하지 않기 위해서다.

적합 척도는 log 절대오차를 능선 회귀로 예측해 exp를 취한다. 축 특징은
원값 대신 데이터셋 안의 백분위를 쓴다. 원값에는 극단치가 있어 지수를
취하는 순간 척도가 발산했다. 예측한 log 척도도 보정 분할 오차의 범위로
잘라 같은 발산을 막는다. 보정 분할의
척도는 cv_fold 교차적합으로 얻어 자기 자신을 보고 정한 척도로 보정하는
일이 없게 한다. 시험 분할의 척도는 보정 분할 전체로 적합한 모형에서 얻는다.

변형은 meta와 test 분할에만 생성했으므로 보정 분할로 meta를 쓴다. 네 구성
모두 같은 분할에서 보정하므로 비교는 대등하다.
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
CALIB = "meta"
EPS = 1e-6
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
MODELS = ("fp_primary", "cb_augmented")
AXES = ("A", "B")
MINIMAL_STATS = ("rel_std",)
RICH_STATS = ("std", "max_dev", "rel_std", "shift")


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def feature_columns(frame: pd.DataFrame, stats: tuple[str, ...]) -> list[str]:
    cols = []
    for model in MODELS:
        for axis in AXES:
            for stat in stats:
                name = f"rich__{model}__{axis}__{stat}__pct"
                if name in frame.columns and frame[name].nunique() > 1:
                    cols.append(name)
    return cols


def fit_scale(x_fit, y_fit, x_apply):
    """log 절대오차를 예측해 척도를 만든다. 예측은 적합 표본 범위로 자른다."""
    scaler = StandardScaler().fit(x_fit)
    model = RidgeCV(alphas=RIDGE_ALPHAS).fit(scaler.transform(x_fit), y_fit)
    lo, hi = np.percentile(y_fit, [1, 99])
    return np.exp(np.clip(model.predict(scaler.transform(x_apply)), lo, hi))


LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def build_lambda_scale(frame, base, wobble, folds, use_base=True):
    """s0 * (1 + lambda * 흔들림). lambda는 보정 분할 교차적합으로 고른다."""
    calib = frame["split"].eq(CALIB).to_numpy()
    test = frame["split"].eq("test").to_numpy()
    error = np.abs(frame["__error"].to_numpy())
    anchor = base if use_base else np.full(len(frame), float(np.mean(base[calib])))

    def widths(lam, fit, apply_to):
        s = anchor * (1.0 + lam * wobble)
        q = conformal_quantile(error[fit] / s[fit], ALPHA)
        return float(np.mean(2.0 * q * s[apply_to]))

    fold = folds[calib]
    idx = np.flatnonzero(calib)
    scores = []
    for lam in LAMBDAS:
        total = []
        for value in np.unique(fold):
            held = idx[fold == value]
            rest = idx[fold != value]
            if len(held) == 0 or len(rest) < 5:
                continue
            total.append(widths(lam, rest, held))
        scores.append(np.mean(total) if total else np.inf)
    best = LAMBDAS[int(np.argmin(scores))]
    return anchor * (1.0 + best * wobble), best


def build_scales(frame, base, columns, folds):
    """보정 분할은 교차적합, 시험 분할은 전체 적합으로 척도를 만든다."""
    calib = frame["split"].eq(CALIB).to_numpy()
    test = frame["split"].eq("test").to_numpy()
    target = np.log(np.abs(frame["__error"].to_numpy()) + EPS)

    design = np.column_stack([np.log(base + EPS)] + [frame[c].to_numpy(float) for c in columns])
    scale = np.full(len(frame), np.nan)

    fold = folds[calib]
    x_calib, y_calib = design[calib], target[calib]
    out = np.empty(calib.sum())
    for value in np.unique(fold):
        held = fold == value
        if held.all() or held.sum() == 0:
            out[held] = np.exp(y_calib[held].mean() if held.sum() else 0.0)
            continue
        out[held] = fit_scale(x_calib[~held], y_calib[~held], x_calib[held])
    scale[calib] = out
    scale[test] = fit_scale(x_calib, y_calib, design[test])
    return scale


def evaluate(frame, scale, floor_ref):
    """보정에서 qhat을 얻어 시험 분할의 커버리지와 구간 폭을 잰다."""
    scale = np.maximum(scale, floor_ref * 1e-3)
    calib = frame["split"].eq(CALIB).to_numpy()
    test = frame["split"].eq("test").to_numpy()
    error = np.abs(frame["__error"].to_numpy())

    qhat = conformal_quantile(error[calib] / scale[calib], ALPHA)
    width = 2.0 * qhat * scale
    covered = error <= qhat * scale

    wobble = frame["__wobble"].to_numpy()
    cut = np.nanquantile(wobble[test], 0.75)
    high = wobble > cut if (wobble[test] > cut).any() else wobble >= cut
    hi_group, lo_group = test & high, test & ~high
    return {
        "qhat": qhat,
        "test_coverage": float(covered[test].mean()),
        "test_mean_width": float(width[test].mean()),
        "test_median_width": float(np.median(width[test])),
        "high_wobble_coverage": float(covered[hi_group].mean()) if hi_group.any() else float("nan"),
        "low_wobble_coverage": float(covered[lo_group].mean()) if lo_group.any() else float("nan"),
    }


def process(dataset: str, evaluation_dir: Path, role2_dir: Path) -> dict | None:
    frame = pd.read_csv(evaluation_dir / dataset / "evaluation_signals.csv")
    if frame["task_type"].iloc[0] != "regression":
        return None
    role2 = pd.read_csv(
        role2_dir / dataset / "role2_signals.csv",
        usecols=["row_uid", "std_fp_primary"],
    )
    frame = frame.merge(role2, on="row_uid", how="left")

    frame["__error"] = frame["Y_final"] - frame["pred_fp_primary"]
    frame["__wobble"] = frame["rich__fp_primary__B__rel_std"].fillna(0.0) \
        + frame["rich__fp_primary__A__rel_std"].fillna(0.0)

    calib = frame["split"].eq(CALIB).to_numpy()
    raw = frame["std_fp_primary"].to_numpy(float)
    floor = float(np.quantile(raw[calib], 0.25))
    if not np.isfinite(floor) or floor <= 0:
        floor = float(np.std(frame.loc[calib, "Y_final"])) * 0.01 or 1e-6
    base = raw + floor
    folds = frame["cv_fold"].to_numpy()

    record = {"dataset": dataset, "n_calib": int(calib.sum()),
              "n_test": int(frame["split"].eq("test").sum())}

    configs = {
        "S0_기준선": None,
        "S1_적합_축없음": [],
        "S2_적합_축최소": feature_columns(frame, MINIMAL_STATS),
        "S3_적합_축확장": feature_columns(frame, RICH_STATS),
    }
    for name, columns in configs.items():
        scale = base if columns is None else build_scales(frame, base, columns, folds)
        for key, value in evaluate(frame, scale, floor).items():
            record[f"{name}__{key}"] = value
        if columns is not None:
            record[f"{name}__n_features"] = 1 + len(columns)

    wob = frame["__wobble"].rank(pct=True, method="average").to_numpy()
    for name, use_base in (("S4_곱셈_모수하나", True), ("S5_축단독", False)):
        scale, lam = build_lambda_scale(frame, base, wob, folds, use_base)
        for key, value in evaluate(frame, scale, floor).items():
            record[f"{name}__{key}"] = value
        record[f"{name}__lambda"] = lam
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="축 분산을 컨포멀 척도로 쓰는 실험")
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--role2-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    evaluation_dir = Path(args.evaluation_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = sorted(p.name for p in evaluation_dir.iterdir()
                      if p.is_dir() and not p.name.startswith("_"))
    records = [r for r in (process(d, evaluation_dir, Path(args.role2_dir))
                           for d in datasets) if r is not None]
    frame = pd.DataFrame(records)
    frame.to_csv(out_dir / "conformal_scale_by_dataset.csv", index=False)

    print(f"회귀 {len(frame)}종  목표 커버리지 {1 - ALPHA:.2f}\n")
    print(f"{'구성':16s} {'커버리지':>8s} {'평균 폭':>9s} {'S1 대비':>9s} "
          f"{'고흔들림 커버':>12s} {'저흔들림 커버':>12s}")
    ref = frame["S1_적합_축없음__test_mean_width"]
    for name in ("S0_기준선", "S1_적합_축없음", "S2_적합_축최소", "S3_적합_축확장",
                 "S4_곱셈_모수하나", "S5_축단독"):
        rel = (frame[f"{name}__test_mean_width"] / ref).mean()
        print(f"{name:16s} {frame[f'{name}__test_coverage'].mean():8.4f} "
              f"{frame[f'{name}__test_mean_width'].mean():9.4f} {rel:9.4f} "
              f"{frame[f'{name}__high_wobble_coverage'].mean(skipna=True):12.4f} "
              f"{frame[f'{name}__low_wobble_coverage'].mean(skipna=True):12.4f}")

    summary = {}
    rng = np.random.default_rng(42)
    for name, base_col in (("S2_적합_축최소", ref), ("S3_적합_축확장", ref),
                           ("S4_곱셈_모수하나", frame["S0_기준선__test_mean_width"])):
        gain = 1.0 - frame[f"{name}__test_mean_width"] / base_col
        boot = [rng.choice(gain, len(gain), replace=True).mean() for _ in range(10000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        summary[name] = {"mean_width_reduction": float(gain.mean()),
                         "ci_low": float(lo), "ci_high": float(hi),
                         "n_improved": int((gain > 0).sum()), "n_total": len(gain)}
        print(f"\n{name} 폭 감소  {gain.mean():+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]  개선 {int((gain > 0).sum())}/{len(gain)}종")
    (out_dir / "conformal_scale_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
