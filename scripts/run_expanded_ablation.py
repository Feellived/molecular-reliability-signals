#!/usr/bin/env python
"""폴드 외 예측으로 표본을 확대한 제거 실험 (연구계획서 5.7·6.4절).

결합 규칙을 meta 분할에서만 학습하면 meta가 47개뿐인 물성에서 규칙이 잡음을
학습한다. train 분할의 폴드 외 예측을 확보했으므로 두 분할을 합쳐 규칙을
학습하고, 같은 test에서 평가해 표본 확대의 효과를 측정한다.

대상은 meta가 200건 미만인 15종이다. 나머지 7종은 이미 표본이 충분해 확대
대상이 아니다.

train 행의 신호는 모두 폴드 외 예측에서 다시 계산한다. 원래 예측은 모델이
그 분자를 학습에 포함한 값이라 쓸 수 없다.

  지문 예측·표준편차   폴드별 재적합 결과
  ChemBERTa 예측       폴드별 재미세조정 결과
  적용가능도메인       담당2 산출값을 그대로 쓴다. 자기 자신을 제외하고
                       계산했으므로 train 행에서도 오염되지 않는다
  컨포멀               보정 분할의 qhat을 그대로 적용하되 척도는 폴드 외
                       예측으로 다시 계산한다
  A축·B축              폴드 외 예측으로 변형 분산을 다시 계산한다

신호 방향은 값이 클수록 위험한 쪽으로 통일하고 물성 안에서 백분위로 정규화한다.
학습 집합과 평가 집합의 백분위는 각각 자기 집합 안에서 매긴다.
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

RIDGE_ALPHA = 1.0
REGRESSION_ERROR_QUANTILE = 0.80
DATASET_LEVEL_AXES = ("B1_tautomer", "B1_protonation")
MOLECULE_LEVEL_AXES = ("B3_stereo",)

SIGNALS = ["ad_knn", "ad_density", "disagreement", "conformal_cb", "conformal_fp"]
AXIS_SIGNALS = ["axis_A", "axis_B"]


def allowed_axes(dataset, allowance, decision) -> tuple[str, ...]:
    axes = [
        a for a in DATASET_LEVEL_AXES
        if allowance.at[dataset, a] == "허용" and decision.at[dataset, f"use_{a}"] == "사용"
    ]
    axes += [a for a in MOLECULE_LEVEL_AXES if decision.at[dataset, f"use_{a}"] == "사용"]
    return tuple(axes)


def spread(values: np.ndarray, parent: float) -> float:
    sample = np.append(values, parent)
    return float(np.std(sample, ddof=0)) if len(sample) > 1 else 0.0


def variant_spreads(variants: pd.DataFrame, parents: pd.Series, column: str,
                    axes: tuple[str, ...]) -> pd.Series:
    selected = variants[variants["axis"].isin(axes)]
    if selected.empty:
        return pd.Series(0.0, index=parents.index)
    grouped = selected.groupby("parent_row_uid")[column].apply(
        lambda values: spread(values.to_numpy(dtype=float), float(parents.get(values.name, np.nan)))
    )
    return grouped.reindex(parents.index).fillna(0.0)


def build_train_rows(dataset: str, base: Path, scores: Path, jiye: Path,
                     axes: tuple[str, ...], task: str) -> pd.DataFrame:
    origin = pd.read_csv(scores / "train_oof_fingerprint" / dataset / "origin_predictions_oof.csv")
    origin_cb = pd.read_csv(
        scores / "train_oof_chemberta" / dataset / "origin_predictions_chemberta_oof.csv"
    ).drop(columns=["dataset", "split", "cv_fold"], errors="ignore")
    origin = origin.merge(origin_cb, on="row_uid")

    variants = pd.read_csv(
        scores / "train_oof_fingerprint" / dataset / "variant_predictions_fp_oof.csv"
    ).merge(
        pd.read_csv(
            scores / "train_oof_chemberta" / dataset / "variant_predictions_chemberta_oof.csv"
        ).drop(columns=["dataset", "axis", "split", "cv_fold", "parent_row_uid"], errors="ignore"),
        on="variant_uid",
    )

    role2 = pd.read_csv(jiye / dataset / "role2_signals.csv")
    frame = origin.merge(
        role2[["row_uid", "Y_final", "ad_knn_tanimoto_top5_mean",
               "ad_local_density_count_s040"]],
        on="row_uid",
    ).set_index("row_uid", drop=False)

    conformal = json.loads(
        (scores / "fp_conformal" / dataset / "fp_conformal_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    floor = float(conformal.get("scale_floor", 0.0))
    qhat = float(conformal["qhat"])

    truth = pd.to_numeric(frame["Y_final"]).to_numpy(dtype=float)
    frame["abs_error_fp"] = np.abs(truth - frame["pred_fp_primary"].to_numpy(dtype=float))

    # 방향을 값이 클수록 위험한 쪽으로 통일한다.
    frame["ad_knn"] = -frame["ad_knn_tanimoto_top5_mean"]
    frame["ad_density"] = -frame["ad_local_density_count_s040"]
    frame["disagreement"] = np.abs(
        frame["pred_fp_primary"].to_numpy(dtype=float)
        - frame["pred_chemberta_augmented"].to_numpy(dtype=float)
    )
    frame["conformal_fp"] = qhat * (frame["std_fp_primary"].to_numpy(dtype=float) + floor)
    frame["conformal_cb"] = np.abs(
        frame["pred_chemberta_augmented"].to_numpy(dtype=float)
        - frame["pred_chemberta_regular"].to_numpy(dtype=float)
    )

    frame["axis_A"] = variant_spreads(
        variants, frame["pred_chemberta_augmented"], "pred_chemberta_augmented", ("A",)
    )
    frame["axis_B"] = variant_spreads(
        variants, frame["pred_fp_primary"], "pred_fp_primary", axes
    )
    return frame.reset_index(drop=True)


def build_eval_rows(dataset: str, evaluation: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(evaluation / dataset / "evaluation_signals.csv")
    frame = frame[frame["split"].eq(split)].copy()
    rename = {
        "base__ad_knn": "ad_knn", "base__ad_density": "ad_density",
        "base__disagreement": "disagreement", "base__conformal_cb": "conformal_cb",
        "base__conformal_fp": "conformal_fp",
        "axis__cb_augmented__A": "axis_A",
        "cond_B__fp_primary__std": "axis_B",
    }
    missing = [c for c in rename if c not in frame.columns]
    if missing:
        raise KeyError(f"{dataset}에 없는 열: {missing}")
    for source, target in rename.items():
        frame[target] = frame[source]
    return frame


def percentile(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[f"{column}__pct"] = out[column].rank(pct=True, method="average")
    return out


def error_labels(frame: pd.DataFrame, task: str) -> np.ndarray:
    truth = pd.to_numeric(frame["Y_final"]).to_numpy(dtype=float)
    if task == "classification":
        predicted = (frame["pred_fp_primary"].to_numpy(dtype=float) >= 0.5).astype(int)
        return (predicted != truth.astype(int)).astype(int)
    error = frame["abs_error_fp"].to_numpy(dtype=float)
    return (error >= np.quantile(error, REGRESSION_ERROR_QUANTILE)).astype(int)


def normalized_aurc(risk: np.ndarray, error: np.ndarray) -> float:
    def aurc(score):
        ordered = error[np.argsort(score, kind="stable")]
        return float(np.mean(np.cumsum(ordered) / np.arange(1, len(ordered) + 1)))

    oracle, random = aurc(error), float(np.mean(error))
    return np.nan if random - oracle < 1e-12 else (aurc(risk) - oracle) / (random - oracle)


def main() -> int:
    parser = argparse.ArgumentParser(description="표본 확대 제거 실험")
    parser.add_argument("--base", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    base = Path(args.base)
    scores = base / "Juhyeong/data/processed/scores_role4"
    evaluation = scores / "evaluation"
    jiye = base / "Jiye/outputs"
    reports = base / "Yoonsoo/processed/reports"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    allowance = pd.read_csv(reports / "06_transformation_allowance_revised.csv").set_index("dataset")
    decision = pd.read_csv(reports / "07_axis_decision.csv").set_index("dataset")
    datasets = sorted(
        p.name for p in (scores / "train_oof_chemberta").iterdir()
        if p.is_dir() and not p.name.startswith("_")
    )

    configurations = {"기준": SIGNALS, "기준+A": SIGNALS + ["axis_A"],
                      "기준+B": SIGNALS + ["axis_B"], "기준+A+B": SIGNALS + AXIS_SIGNALS}

    records = []
    for dataset in datasets:
        task = pd.read_csv(evaluation / dataset / "evaluation_signals.csv")["task_type"].iloc[0]
        axes = allowed_axes(dataset, allowance, decision)

        meta = build_eval_rows(dataset, evaluation, "meta")
        test = build_eval_rows(dataset, evaluation, "test")
        train = build_train_rows(dataset, base, scores, jiye, axes, task)

        columns = SIGNALS + AXIS_SIGNALS
        test_pct = percentile(test, columns)
        error = test["abs_error_fp"].to_numpy(dtype=float)
        labels = error_labels(test, task)

        record = {"dataset": dataset, "task_type": task, "n_meta": len(meta),
                  "n_train": len(train), "n_test": len(test)}
        for tag, fit_frame in [
            ("meta만", percentile(meta, columns)),
            ("meta+train", percentile(pd.concat([meta, train], ignore_index=True), columns)),
        ]:
            target = rankdata(fit_frame["abs_error_fp"]) / len(fit_frame)
            for name, features in configurations.items():
                used = [f"{c}__pct" for c in features]
                model = Ridge(alpha=RIDGE_ALPHA)
                model.fit(fit_frame[used].to_numpy(dtype=float), target)
                score = model.predict(test_pct[used].to_numpy(dtype=float))
                record[f"auprc__{tag}__{name}"] = (
                    float(average_precision_score(labels, score))
                    if labels.min() != labels.max() else np.nan
                )
                record[f"aurc__{tag}__{name}"] = normalized_aurc(score, error)
        records.append(record)
        print(
            f"  {dataset:32s} meta {record['n_meta']:4d} + train {record['n_train']:5d}",
            flush=True,
        )

    table = pd.DataFrame(records)
    table.to_csv(out_dir / "expanded_ablation_by_dataset.csv", index=False)

    print()
    print(f"=== 표본 확대 제거 실험, {len(table)}종 ===")
    print(f"  규칙 학습 표본 평균 {table['n_meta'].mean():.0f} → "
          f"{(table['n_meta'] + table['n_train']).mean():.0f}")
    for metric, arrow, sign in [("auprc", "높을수록 좋음", 1), ("aurc", "낮을수록 좋음", -1)]:
        print()
        print(f"[{metric.upper()} {arrow}]")
        for tag in ("meta만", "meta+train"):
            line = f"  {tag:11s}"
            for name in configurations:
                line += f"  {name} {table[f'{metric}__{tag}__{name}'].mean():.4f}"
            print(line)
        print("  축 추가 효과")
        for tag in ("meta만", "meta+train"):
            for name in ("기준+A", "기준+B", "기준+A+B"):
                delta = (table[f"{metric}__{tag}__{name}"]
                         - table[f"{metric}__{tag}__기준"]).dropna() * sign
                p = wilcoxon(delta)[1] if len(delta) >= 6 else np.nan
                print(f"    {tag:11s} {name:8s} {delta.mean():+.4f}  "
                      f"개선 {int((delta > 0).sum())}/{len(delta)}종  p={p:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
