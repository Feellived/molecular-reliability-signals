#!/usr/bin/env python
"""1단계 마지막 탐색: 지문 전용 기준에서의 B 구성 비교 (22종, 사전 기록 2 이전).

새 데이터(CYP1A2·2C19)는 ChemBERTa 없이 평가하므로, 같은 조건의 기준을 22종에도
적용해 어떤 B 구성을 사전 기록할지 정한다. 이 스크립트의 결과는 탐색이며, 여기서
고른 구성은 사전 기록 2에 고정한 뒤 새 데이터에 한 번만 적용한다.

  기준      적용가능도메인(이웃·밀도) + 지문 컨포멀
  B         조건부 B 지문 표준편차 (파이프라인과 같은 정의)
  B 호변Δ   호변이성질체 중 RDKit 안정성 점수가 최고점에서 Δ 이내인 것만 남긴 B
  B 모양    변형 예측의 표준편차·최대 이탈·방향 이동·상대 흩어짐·라벨 반전(확장 통계)
  강화 기준  기준 + 분자 크기·헤테로원자 수·조건부 변형 수
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from scipy.stats import rankdata, wilcoxon
from sklearn.linear_model import Ridge

from build_conditional_signals import allowed_axes

RDLogger.DisableLog("rdApp.*")
BASE = ["base__ad_knn__pct", "base__ad_density__pct", "base__conformal_fp__pct"]
DESC = ["heavy__pct", "hetero__pct", "n_conditional_variants__pct"]
B = ["cond_B__fp_primary__std__pct"]
RICH = [f"rich__fp_primary__B__{s}__pct" for s in ("std", "max_dev", "shift", "rel_std", "flip")]


def naurc(score, error):
    f = lambda x: float(np.mean(np.cumsum(error[np.argsort(x, kind="stable")]) / np.arange(1, len(error) + 1)))
    oracle, rand = f(error), float(np.mean(error))
    return np.nan if rand - oracle < 1e-12 else (f(score) - oracle) / (rand - oracle)


def tautomer_score(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    return float(rdMolStandardize.TautomerEnumerator.ScoreTautomer(mol)) if mol is not None else -1e9


def filtered_b(dataset, scores_dir, variants_dir, splits, axes, delta):
    """호변이성질체만 안정성 점수로 거른 조건부 B 지문 표준편차."""
    var = pd.read_csv(variants_dir / dataset / "variants.csv", usecols=["variant_uid", "variant_smiles"])
    fp = pd.read_csv(scores_dir / "fingerprint" / dataset / "variant_predictions_fp.csv").merge(var, on="variant_uid")
    fp = fp[fp["axis"].isin(axes)].copy()
    origin = pd.read_csv(scores_dir / "fingerprint" / dataset / "origin_predictions_refit.csv").set_index("row_uid")
    parent_smiles = splits.set_index("row_uid")["parent_smiles"]
    taut = fp["axis"].eq("B1_tautomer")
    fp.loc[taut, "tscore"] = [tautomer_score(s) for s in fp.loc[taut, "variant_smiles"]]
    rows = {}
    for uid, g in fp.groupby("parent_row_uid"):
        keep = g[~g["axis"].eq("B1_tautomer")]
        t = g[g["axis"].eq("B1_tautomer")]
        if len(t):
            best = max(t["tscore"].max(), tautomer_score(parent_smiles.get(uid, "")))
            keep = pd.concat([keep, t[t["tscore"] >= best - delta]])
        sample = np.append(keep["pred_fp_primary"].to_numpy(float), float(origin.at[uid, "pred_fp_primary"]))
        rows[uid] = float(np.std(sample)) if len(sample) > 1 else 0.0
    return pd.Series(rows, name=f"b_taut{delta}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--variants-dir", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--reports-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    scores_dir, variants_dir = Path(args.scores_dir), Path(args.variants_dir)
    reports = Path(args.reports_dir)
    allowance = pd.read_csv(reports / "06_transformation_allowance_final.csv", encoding="utf-8-sig").set_index("dataset")
    decision = pd.read_csv(reports / "07_axis_decision.csv").set_index("dataset")
    evaluation = scores_dir / "evaluation"
    configs = {"+B": BASE + B, "+B 호변Δ0": BASE + ["b_taut0__pct"], "+B 호변Δ2": BASE + ["b_taut2__pct"],
               "+B 모양": BASE + RICH, "강화 기준": BASE + DESC, "강화 기준+B": BASE + DESC + B,
               "강화 기준+B 모양": BASE + DESC + RICH}
    rows = []
    for dataset in sorted(p.name for p in evaluation.iterdir() if p.is_dir() and not p.name.startswith("_")):
        m = pd.read_csv(evaluation / dataset / "evaluation_signals.csv")
        splits = pd.read_csv(Path(args.splits_dir) / dataset / "splits.csv", usecols=["row_uid", "parent_smiles"])
        m = m.merge(splits, on="row_uid", how="left")
        mols = [Chem.MolFromSmiles(s) for s in m["parent_smiles"]]
        m["heavy"] = [x.GetNumHeavyAtoms() if x else 0 for x in mols]
        m["hetero"] = [rdMolDescriptors.CalcNumHeteroatoms(x) if x else 0 for x in mols]
        axes = allowed_axes(dataset, allowance, decision)
        for delta in (0, 2):
            m = m.merge(filtered_b(dataset, scores_dir, variants_dir, splits, axes, delta).rename_axis("row_uid").reset_index(),
                        on="row_uid", how="left")
        for c in ["heavy", "hetero", "n_conditional_variants", "b_taut0", "b_taut2"]:
            m[f"{c}__pct"] = m[c].fillna(0).rank(pct=True, method="average")
        meta, test = m[m.split == "meta"], m[m.split == "test"]
        error, target = test.abs_error_fp.to_numpy(float), rankdata(meta.abs_error_fp) / len(meta)

        def fit(features):
            used = [c for c in features if c in m and m[c].nunique() > 1]
            return naurc(Ridge(alpha=1.0).fit(meta[used], target).predict(test[used]), error)

        record = {"dataset": dataset, "기준": fit(BASE)}
        record.update({name: fit(f) for name, f in configs.items()})
        rows.append(record)
        print(f"  {dataset}", flush=True)
    table = pd.DataFrame(rows).set_index("dataset")
    table.to_csv(args.out)
    cyp = table.index.str.startswith("cyp")
    print(f"\n{'구성 (기준 대비 AURC 개선)':24s}{'22종':>9s}{'개선':>7s}{'p':>7s}{'CYP':>9s}{'기타':>9s}")
    for name in configs:
        ref = table["강화 기준"] if name.startswith("강화 기준+") else table["기준"]
        v = ref - table[name]
        print(f"{name:24s}{v.mean():+9.4f}{int((v > 0).sum()):5d}/22{wilcoxon(v)[1]:7.3f}{v[cyp].mean():+9.4f}{v[~cyp].mean():+9.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
