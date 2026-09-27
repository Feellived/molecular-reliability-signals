#!/usr/bin/env python
"""사전 기록 2의 평가. 한 번만 실행한다 (docs/사전기록_2_CYP_재현.md).

신호 정의는 22종 파이프라인의 해당 함수를 그대로 불러 쓴다.
  적용가능도메인·컨포멀 부호와 백분위   assemble_signals.py
  조건부 B 축 선택                   build_conditional_signals.allowed_axes
  B 모양 통계                        build_rich_variant_features.statistics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from scipy.stats import rankdata
from sklearn.linear_model import Ridge

from build_conditional_signals import allowed_axes
from build_rich_variant_features import statistics

STATS = ("std", "max_dev", "shift", "rel_std", "flip")
BASE = ["ad_knn", "ad_density", "conformal_fp"]
DESC = ["heavy", "hetero", "n_conditional_variants"]
SHAPE = [f"shape_{s}" for s in STATS]
CONFIGS = {"기준": BASE, "기준+B 모양": BASE + SHAPE, "기준+B": BASE + ["shape_std"],
           "강화 기준": BASE + DESC, "강화 기준+B 모양": BASE + DESC + SHAPE}
COMPARISONS = {"1차": ("기준", "기준+B 모양"), "2차": ("강화 기준", "강화 기준+B 모양"), "참고": ("기준", "기준+B")}
N_BOOT, SEED = 2000, 20260926


def naurc(score, error):
    f = lambda x: float(np.mean(np.cumsum(error[np.argsort(x, kind="stable")]) / np.arange(1, len(error) + 1)))
    oracle, rand = f(error), float(np.mean(error))
    return np.nan if rand - oracle < 1e-12 else (f(score) - oracle) / (rand - oracle)


def build_frame(dataset, splits_dir, role2_dir, scores_dir, axes, task):
    splits = pd.read_csv(splits_dir / dataset / "splits.csv")
    role2 = pd.read_csv(role2_dir / dataset / "role2_signals.csv")
    conf = pd.read_csv(scores_dir / "fp_conformal" / dataset / "fp_conformal.csv")
    frame = splits[["row_uid", "split", "parent_smiles", "scaffold_group", "Y_final"]].merge(
        role2[["row_uid", "pred_fp_primary", "ad_knn_tanimoto_top5_mean", "ad_local_density_count_s040"]], on="row_uid"
    )
    # 지문 컨포멀은 분류가 예측 집합 크기, 회귀가 구간 폭이다.
    # assemble_signals.py가 22종에서 쓰는 대응과 같다.
    conformal_column = "fp_aps_set_size" if "fp_aps_set_size" in conf else "fp_conformal_width"
    frame = frame.merge(conf[["row_uid", conformal_column]], on="row_uid")
    frame = frame[frame.split.isin(["meta", "test"])].reset_index(drop=True)
    frame["error"] = np.abs(frame.Y_final.astype(float) - frame.pred_fp_primary)
    frame["ad_knn"] = -frame.ad_knn_tanimoto_top5_mean
    frame["ad_density"] = -frame.ad_local_density_count_s040
    frame["conformal_fp"] = frame[conformal_column]

    variants = pd.read_csv(scores_dir / "fingerprint" / dataset / "variant_predictions_fp.csv")
    variants = variants[variants.axis.isin(axes)]
    origin = pd.read_csv(scores_dir / "fingerprint" / dataset / "origin_predictions_refit.csv").set_index("row_uid")
    scale = float(origin.pred_fp_primary.std())
    grouped = {k: g.pred_fp_primary.to_numpy(float) for k, g in variants.groupby("parent_row_uid")}
    shape, counts = [], []
    for uid in frame.row_uid:
        values = grouped.get(uid, np.array([]))
        shape.append(statistics(values, float(origin.at[uid, "pred_fp_primary"]), scale, task))
        counts.append(len(values))
    for s in STATS:
        frame[f"shape_{s}"] = [d[s] for d in shape]
    frame["n_conditional_variants"] = counts
    mols = [Chem.MolFromSmiles(s) for s in frame.parent_smiles]
    frame["heavy"] = [m.GetNumHeavyAtoms() for m in mols]
    frame["hetero"] = [rdMolDescriptors.CalcNumHeteroatoms(m) for m in mols]
    for c in BASE + DESC + SHAPE:
        frame[f"{c}__pct"] = frame[c].rank(pct=True, method="average")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--role2-dir", required=True)
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--reports-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = Path(args.reports_dir)
    allowance = pd.read_csv(reports / "06_transformation_allowance_final.csv", encoding="utf-8-sig").set_index("dataset")
    decision = pd.read_csv(reports / "07_axis_decision.csv").set_index("dataset")
    rng = np.random.default_rng(SEED)
    report = {}
    for dataset in args.datasets:
        axes = allowed_axes(dataset, allowance, decision)
        task = pd.read_csv(Path(args.splits_dir) / dataset / "splits.csv", usecols=["task_type"]).task_type.iloc[0]
        frame = build_frame(dataset, Path(args.splits_dir), Path(args.role2_dir), Path(args.scores_dir), axes, task)
        meta, test = frame[frame.split == "meta"], frame[frame.split == "test"].reset_index(drop=True)
        target = rankdata(meta.error) / len(meta)
        error = test.error.to_numpy(float)
        scores = {}
        for name, feats in CONFIGS.items():
            cols = [f"{c}__pct" for c in feats if frame[f"{c}__pct"].nunique() > 1]
            scores[name] = Ridge(alpha=1.0).fit(meta[cols], target).predict(test[cols])
        index = [np.flatnonzero(test.scaffold_group.to_numpy() == g) for g in test.scaffold_group.unique()]
        result = {"axes": list(axes), "n_test": len(test), "n_scaffold": len(index)}
        for label, (ref, cand) in COMPARISONS.items():
            effect = naurc(scores[ref], error) - naurc(scores[cand], error)
            boot = []
            for _ in range(N_BOOT):
                rows = np.concatenate([index[k] for k in rng.integers(0, len(index), len(index))])
                boot.append(naurc(scores[ref][rows], error[rows]) - naurc(scores[cand][rows], error[rows]))
            lo, hi = np.nanpercentile(boot, [2.5, 97.5])
            result[label] = {"effect": effect, "ci": [float(lo), float(hi)]}
        report[dataset] = result
        print(dataset, json.dumps(result, ensure_ascii=False), flush=True)
    primary = [report[d]["1차"] for d in args.datasets]
    if all(p["effect"] > 0 for p in primary):
        verdict = "재현 성공" if any(p["ci"][0] > 0 for p in primary) else "부분 재현"
    else:
        verdict = "재현 실패"
    report["verdict"] = verdict
    (out_dir / "replication_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("판정:", verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
