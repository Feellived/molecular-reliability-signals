#!/usr/bin/env python
"""신호 정규화를 계획서 5.8절이 지정한 방식으로 바꿔 다시 판정한다.

현행 구현은 두 곳에서 5.8절과 어긋난다.

  백분위 기준  계획서는 "시험 집합이 아니라 보정 집합에서 고정한다"고 정한다.
              현행은 meta와 test를 합친 전체에서 순위를 매겨, 시험 분자의
              신호가 다른 시험 분자들이 누구인지에 따라 달라진다. 라벨을
              쓰지 않으므로 누출은 아니지만 계획서가 금지한 형태이고,
              분자 하나를 입력받는 데모에서는 애초에 쓸 수 없는 방식이다.

  통합 B      계획서는 "각 축의 분포를 백분위로 변환한 뒤 그 백분위의
              최댓값"으로 정한다. 현행은 세 축 변형을 한 표본으로 합쳐
              표준편차를 냈다. 최댓값 방식은 어느 한 축에서라도 크게
              흔들리면 위험으로 보는 반면, 합친 표준편차는 축끼리 상쇄된다.
              변형이 분자당 1개뿐인 B3가 특히 불리하다.

네 구성을 같은 제거 실험 절차로 비교한다. 결과가 바뀌든 바뀌지 않든 둘 다
보고한다. 사전 지정 판정은 현행 구성으로 유지하고, 5.8절 준수 구성은
결론이 정규화 방식에 둔감한지 확인하는 용도로 함께 싣는다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score

BASE = ["base__ad_knn", "base__ad_density", "base__conformal_cb", "base__conformal_fp"]
A_COL = "axis__cb_augmented__A"
B_AXES = ("B1_tautomer", "B1_protonation", "B3_stereo")
MODELS = ("fp_primary", "cb_augmented")


def calibration_percentile(values, reference) -> np.ndarray:
    """보정 집합의 경험 분포로 백분위를 매긴다. 기준이 분할 밖에 고정된다."""
    reference = np.sort(np.asarray(reference, dtype=float))
    if len(reference) == 0:
        return np.zeros(len(values))
    return np.searchsorted(reference, np.asarray(values, dtype=float),
                           side="right") / len(reference)


def normalized_aurc(score, error) -> float:
    curve = lambda x: float(np.mean(
        np.cumsum(error[np.argsort(x, kind="stable")]) / np.arange(1, len(error) + 1)))
    oracle, random = curve(error), float(np.mean(error))
    return np.nan if random - oracle < 1e-12 else (curve(score) - oracle) / (random - oracle)


def make_features(frame, fix_on_calib: bool, b_as_max: bool) -> pd.DataFrame:
    """구성에 따라 백분위 기준과 통합 B 정의를 바꾼다."""
    meta = frame["split"].eq("meta").to_numpy()
    out = pd.DataFrame(index=frame.index)

    def pct(column):
        values = frame[column].to_numpy(float)
        if fix_on_calib:
            return calibration_percentile(values, values[meta])
        return pd.Series(values).rank(pct=True, method="average").to_numpy()

    for column in (*BASE, A_COL):
        if column in frame:
            out[column] = pct(column)

    if b_as_max:
        # 5.8절: 축별 백분위의 최댓값.
        # 그 분자에 변형이 생성되지 않은 축은 값이 정확히 0인데, 보정 집합에도
        # 0이 많아 백분위가 높게 잡힌다. 미적용을 위험으로 읽는 셈이므로
        # 결측으로 두고 최댓값에서 제외한다.
        for model in MODELS:
            columns = [f"axis__{model}__{axis}" for axis in B_AXES
                       if f"axis__{model}__{axis}" in frame
                       and frame[f"axis__{model}__{axis}"].nunique() > 1]
            if not columns:
                out[f"B__{model}"] = 0.0
                continue
            stack = []
            for column in columns:
                values = pct(column)
                stack.append(np.where(frame[column].to_numpy(float) > 0, values, np.nan))
            merged = np.nanmax(np.column_stack(stack), axis=1)
            out[f"B__{model}"] = np.nan_to_num(merged, nan=0.0)
    else:
        for model in MODELS:
            column = f"cond_B__{model}__std"
            out[f"B__{model}"] = pct(column) if column in frame else 0.0
    return out


def evaluate(dataset, evaluation_dir, fix_on_calib, b_as_max) -> dict:
    frame = pd.read_csv(evaluation_dir / dataset / "evaluation_signals.csv")
    features = make_features(frame, fix_on_calib, b_as_max)
    meta, test = frame.split.eq("meta").to_numpy(), frame.split.eq("test").to_numpy()

    error = frame.loc[test, "abs_error_fp"].to_numpy(float)
    target = rankdata(frame.loc[meta, "abs_error_fp"]) / meta.sum()
    label = ((((frame.loc[test, "pred_fp_primary"].to_numpy(float) >= .5).astype(int)
               != pd.to_numeric(frame.loc[test, "Y_final"]).astype(int)).astype(int))
             if frame.task_type.iloc[0] == "classification"
             else (error >= np.quantile(error, .8)).astype(int))

    b_cols = [f"B__{m}" for m in MODELS]
    configs = {"기준": BASE, "기준+A": BASE + [A_COL],
               "기준+B": BASE + b_cols, "기준+A+B": BASE + [A_COL] + b_cols}
    record = {"dataset": dataset}
    for name, columns in configs.items():
        use = [c for c in columns if c in features and features[c].nunique() > 1]
        score = Ridge(alpha=1.0).fit(features.loc[meta, use].to_numpy(),
                                     target).predict(features.loc[test, use].to_numpy())
        record[f"auprc__{name}"] = (average_precision_score(label, score)
                                    if label.min() != label.max() else np.nan)
        record[f"aurc__{name}"] = normalized_aurc(score, error)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="5.8절 준수 정규화로 재판정")
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    evaluation_dir, out_dir = Path(args.evaluation_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = sorted(p.name for p in evaluation_dir.iterdir()
                      if p.is_dir() and not p.name.startswith("_"))

    variants = {"현행": (False, False), "백분위 보정고정": (True, False),
                "통합B 최댓값": (False, True), "5.8절 준수": (True, True)}
    rng = np.random.default_rng(42)
    summary = {}
    print(f"{'구성':16s}{'지표':7s}{'B 추가':>10s}{'95% 구간':>24s}{'개선':>8s}"
          f"{'A+B 추가':>11s}{'개선':>8s}")
    for name, (fix, bmax) in variants.items():
        table = pd.DataFrame([evaluate(d, evaluation_dir, fix, bmax) for d in datasets])
        table.to_csv(out_dir / f"ablation_{name.replace(' ', '_')}.csv", index=False)
        summary[name] = {}
        for metric, label, sign in (("auprc", "AUPRC", 1), ("aurc", "AURC", -1)):
            line = f"{name if metric == 'auprc' else '':16s}{label:7s}"
            for config in ("기준+B", "기준+A+B"):
                delta = ((table[f"{metric}__{config}"]
                          - table[f"{metric}__기준"]) * sign).dropna().to_numpy()
                boot = [rng.choice(delta, len(delta), replace=True).mean() for _ in range(10000)]
                lo, hi = np.percentile(boot, [2.5, 97.5])
                summary[name][f"{metric}__{config}"] = {
                    "mean": float(delta.mean()), "ci": [float(lo), float(hi)],
                    "n_improved": int((delta > 0).sum()), "n": len(delta)}
                if config == "기준+B":
                    line += (f"{delta.mean():+10.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>24s}"
                             f"{f'{int((delta > 0).sum())}/{len(delta)}':>8s}")
                else:
                    line += f"{delta.mean():+11.4f}{f'{int((delta > 0).sum())}/{len(delta)}':>8s}"
            print(line)
        print()
    (out_dir / "compliance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
