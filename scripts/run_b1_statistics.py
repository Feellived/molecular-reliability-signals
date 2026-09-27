#!/usr/bin/env python
"""B-1 통계 절차 정합 (계획서 5.8·6.5절). 설계는 담당3(영주)의 5주차 수정 노트북을 따른다.

세 가지를 한 번에 낸다.

  골격 단위 부트스트랩  표제 수치(기준 대비 B 추가의 AURC 개선)를 물성 안에서
                       골격 군집째 재표집해 다시 잰다. 결합 규칙은 다시 학습하지
                       않고 meta로 적합한 점수를 그대로 쓴다. 같은 골격의 분자는
                       함께 맞고 함께 틀리므로 분자를 하나씩 뽑으면 신뢰구간이
                       실제보다 좁아진다.
  BH 다중비교          물성별 효과 22개에 대해 부트스트랩 양측 p를 구하고
                       Benjamini-Hochberg로 보정한다. 1차 가설(22종 평균)에는
                       적용하지 않는다.
  MAD 민감도           축 신호의 표준편차 자리를 중앙값 절대편차로 바꿔 같은 제거
                       실험을 다시 한다. 튀는 변형 하나가 결과를 좌우하는지 본다.
                       신호는 물성 안 백분위로만 쓰이므로 MAD의 척도 상수는 결과에
                       영향이 없다.

입력은 run_preregistered_ablation.py와 같은 evaluation_signals.csv다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation, rankdata, wilcoxon
from sklearn.linear_model import Ridge

from analyze_heterogeneity import benjamini_hochberg
from build_conditional_signals import MODELS, allowed_axes

BASE = ["base__ad_knn__pct", "base__ad_density__pct", "base__conformal_cb__pct", "base__conformal_fp__pct"]
B_STD = ["cond_B__fp_primary__std__pct", "cond_B__cb_augmented__std__pct"]
B_MAD = ["cond_B__fp_primary__mad__pct", "cond_B__cb_augmented__mad__pct"]
A_STD = ["axis__cb_augmented__A__pct"]
A_MAD = ["axis__cb_augmented__A__mad__pct"]
CONFIGS = {
    "기준": BASE,
    "기준+B": BASE + B_STD,
    "기준+B(MAD)": BASE + B_MAD,
    "기준+A": BASE + A_STD,
    "기준+A(MAD)": BASE + A_MAD,
}
ZERO_TOLERANCE = 1e-12


def naurc(score: np.ndarray, error: np.ndarray) -> float:
    """run_preregistered_ablation.py와 같은 정규화 AURC. 0이 이상, 1이 무작위."""
    f = lambda x: float(np.mean(np.cumsum(error[np.argsort(x, kind="stable")]) / np.arange(1, len(error) + 1)))
    oracle, rand = f(error), float(np.mean(error))
    return np.nan if rand - oracle < 1e-12 else (f(score) - oracle) / (rand - oracle)


def mad(sample: np.ndarray) -> float:
    if len(sample) < 2:
        return 0.0
    value = float(median_abs_deviation(sample, scale="normal"))
    return 0.0 if value < ZERO_TOLERANCE else value


def mad_signals(dataset: str, scores_dir: Path, axes: tuple[str, ...]) -> pd.DataFrame:
    """조건부 B와 A축을 표준편차 대신 MAD로. 원본을 표본에 넣는 규칙은 같다."""
    fp = pd.read_csv(scores_dir / "fingerprint" / dataset / "variant_predictions_fp.csv")
    cb = pd.read_csv(scores_dir / "chemberta" / dataset / "variant_predictions_chemberta.csv").drop(
        columns=["dataset", "axis", "split", "parent_row_uid"], errors="ignore")
    variants = fp.merge(cb, on="variant_uid")
    origin = {
        "fp_primary": pd.read_csv(scores_dir / "fingerprint" / dataset / "origin_predictions_refit.csv").set_index("row_uid"),
        "cb_augmented": pd.read_csv(scores_dir / "chemberta" / dataset / "origin_predictions_chemberta.csv").set_index("row_uid"),
    }
    rows = []
    for row_uid, group in variants.groupby("parent_row_uid"):
        record = {"row_uid": row_uid}
        chosen = group[group["axis"].isin(axes)]
        for key, column in MODELS.items():
            parent = float(origin[key].at[row_uid, column])
            record[f"cond_B__{key}__mad"] = (
                mad(np.append(chosen[column].to_numpy(float), parent)) if len(chosen) else 0.0)
        a = group[group["axis"] == "A"]["pred_chemberta_augmented"].to_numpy(float)
        record["axis__cb_augmented__A__mad"] = mad(
            np.append(a, float(origin["cb_augmented"].at[row_uid, "pred_chemberta_augmented"])))
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="B-1 골격 부트스트랩, BH, MAD 민감도")
    parser.add_argument("--scores-dir", required=True, help="evaluation, fingerprint, chemberta 상위")
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--reports-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()

    scores_dir, out_dir = Path(args.scores_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = Path(args.reports_dir)
    allowance = pd.read_csv(reports / "06_transformation_allowance_final.csv", encoding="utf-8-sig").set_index("dataset")
    decision = pd.read_csv(reports / "07_axis_decision.csv").set_index("dataset")
    evaluation = scores_dir / "evaluation"
    datasets = sorted(p.name for p in evaluation.iterdir() if p.is_dir() and not p.name.startswith("_"))

    point_rows, per_molecule = [], {}
    for dataset in datasets:
        frame = pd.read_csv(evaluation / dataset / "evaluation_signals.csv")
        frame = frame.merge(mad_signals(dataset, scores_dir, allowed_axes(dataset, allowance, decision)),
                            on="row_uid", how="left")
        for column in ["cond_B__fp_primary__mad", "cond_B__cb_augmented__mad", "axis__cb_augmented__A__mad"]:
            frame[column] = frame[column].fillna(0.0)
            frame[f"{column}__pct"] = frame[column].rank(pct=True, method="average")
        groups = pd.read_csv(Path(args.splits_dir) / dataset / "splits.csv", usecols=["row_uid", "scaffold_group"])
        frame = frame.merge(groups, on="row_uid", how="left")

        meta, test = frame[frame.split == "meta"], frame[frame.split == "test"].reset_index(drop=True)
        target = rankdata(meta.abs_error_fp) / len(meta)
        error = test.abs_error_fp.to_numpy(float)
        record = {"dataset": dataset, "n_test": len(test), "n_scaffold": test.scaffold_group.nunique()}
        molecule = pd.DataFrame({"row_uid": test.row_uid, "scaffold_group": test.scaffold_group, "error": error})
        for name, features in CONFIGS.items():
            used = [c for c in features if c in frame and frame[c].nunique() > 1]
            score = Ridge(alpha=1.0).fit(meta[used].to_numpy(float), target).predict(test[used].to_numpy(float))
            molecule[f"score__{name}"] = score
            record[f"aurc__{name}"] = naurc(score, error)
        molecule.to_csv(out_dir / f"per_molecule__{dataset}.csv", index=False)
        per_molecule[dataset] = molecule
        point_rows.append(record)
        print(f"  {dataset:32s} test {len(test):5d}  골격 {record['n_scaffold']:5d}", flush=True)

    points = pd.DataFrame(point_rows)
    for name in list(CONFIGS)[1:]:
        points[f"effect__{name}"] = points["aurc__기준"] - points[f"aurc__{name}"]

    # 골격 단위 부트스트랩. 반복마다 22종 모두를 재표집해 평균을 낸다.
    rng = np.random.default_rng(args.seed)
    boot = np.empty((args.n_boot, len(datasets)))
    for j, dataset in enumerate(datasets):
        molecule = per_molecule[dataset]
        index = [np.flatnonzero(molecule.scaffold_group.to_numpy() == g) for g in molecule.scaffold_group.unique()]
        err, base, with_b = (molecule[c].to_numpy(float) for c in ("error", "score__기준", "score__기준+B"))
        for i in range(args.n_boot):
            rows = np.concatenate([index[k] for k in rng.integers(0, len(index), len(index))])
            boot[i, j] = naurc(base[rows], err[rows]) - naurc(with_b[rows], err[rows])
    np.save(out_dir / "scaffold_bootstrap_effects.npy", boot)

    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    less, more = np.nanmean(boot <= 0, axis=0), np.nanmean(boot >= 0, axis=0)
    points["scaffold_ci_low"], points["scaffold_ci_high"] = lo, hi
    points["p_bootstrap"] = np.minimum(1.0, 2 * np.minimum(less, more))
    points["p_bh"] = benjamini_hochberg(points["p_bootstrap"].to_numpy())
    points.to_csv(out_dir / "b1_by_dataset.csv", index=False)

    effect = points["effect__기준+B"].to_numpy()
    mean_boot = np.nanmean(boot, axis=1)
    # 기존 보고 방식(물성 단위 재표집)과 같은 시드·횟수로 나란히 낸다.
    rng_ds = np.random.default_rng(20260902)
    ds_boot = np.array([rng_ds.choice(effect, len(effect), replace=True).mean() for _ in range(8000)])

    def summary(values):
        v = values[~np.isnan(values)]
        return {"mean": float(v.mean()), "improved": int((v > 0).sum()), "n": int(len(v)),
                "wilcoxon_p": float(wilcoxon(v)[1])}

    report = {
        "headline_B": {
            **summary(effect),
            "ci_dataset_bootstrap": [float(x) for x in np.percentile(ds_boot, [2.5, 97.5])],
            "ci_scaffold_bootstrap": [float(x) for x in np.nanpercentile(mean_boot, [2.5, 97.5])],
            "n_boot_scaffold": args.n_boot,
        },
        "mad_sensitivity": {name: summary(points[f"effect__{name}"].to_numpy())
                            for name in ("기준+B", "기준+B(MAD)", "기준+A", "기준+A(MAD)")},
        "bh": {"n_raw_p_below_0.05": int((points.p_bootstrap < 0.05).sum()),
               "n_bh_below_0.05": int((points.p_bh < 0.05).sum())},
    }
    (out_dir / "b1_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
