#!/usr/bin/env python
"""새 물성을 담당1 규칙 그대로 분할한다 (사전 기록 2의 재현 데이터용).

담당1의 두 단계를 한 스크립트로 옮겼다. 규칙은 바꾸지 않았다.

  eda_and_prescreen.py        부모 분자 = LargestFragmentChooser(preferOrganic=True),
                              골격 = 부모 분자의 Murcko 골격
  02_dedup_allowance_split    부모 분자 기준 중복 제거(분류 라벨 충돌은 통째 제외,
                              회귀는 평균), 골격 그룹 탐욕 분할 70/10/10/10,
                              고리 없는 분자는 단일 그룹으로 두고 고리 유무로 층화,
                              test 밖 영역에 골격 그룹 5겹 교차적합 폴드

입력은 TDC 원본 표(Drug_ID, Drug, Y)이며 TDC 벤치마크 시험 집합이 없는 물성이므로
tdc_split은 모두 "full"로 둔다.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold

RDLogger.DisableLog("rdApp.*")
SEED = 42
FRACTIONS = {"train": 0.70, "calib": 0.10, "meta": 0.10, "test": 0.10}
N_CV_FOLDS = 5
SPLIT_COLS = ["row_uid", "dataset", "task_type", "Drug_ID", "smiles_original", "parent_smiles",
              "Y_final", "scaffold", "scaffold_group", "tdc_split", "split", "cv_fold"]
_CHOOSER = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)


def parent_and_scaffold(smiles: str) -> tuple[str | None, str | None]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    try:
        parent = _CHOOSER.choose(mol)
    except Exception:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        parent = mol if len(frags) == 1 else max(frags, key=lambda m: m.GetNumHeavyAtoms())
    return Chem.MolToSmiles(parent), MurckoScaffold.MurckoScaffoldSmiles(mol=parent)


def dedup_by_parent(d: pd.DataFrame, task_type: str) -> pd.DataFrame:
    d = d.copy()
    d["Y"] = pd.to_numeric(d["Y"], errors="coerce")
    d["is_kept"] = d["parent_smiles"].notna()
    d["Y_final"] = d["Y"]
    valid = d[d["is_kept"]]
    prio = np.where(valid["tdc_split"].values == "test", 0, 1)
    valid = valid.assign(_prio=prio).sort_values("_prio", kind="stable")
    for _, idx in valid.groupby("parent_smiles", sort=False).groups.items():
        idx = list(idx)
        if len(idx) == 1:
            continue
        ys = d.loc[idx, "Y"]
        keeper, others = idx[0], idx[1:]
        if np.allclose(ys.values, ys.values[0], rtol=0, atol=1e-9):
            d.loc[others, "is_kept"] = False
        elif task_type == "regression":
            d.loc[keeper, "Y_final"] = float(ys.mean())
            d.loc[others, "is_kept"] = False
        else:
            d.loc[idx, "is_kept"] = False
    return d


def greedy_scaffold_split(group_ids, fractions, seed=SEED):
    groups = defaultdict(list)
    for i, g in enumerate(group_ids):
        groups[g].append(i)
    n_total = len(group_ids)
    names = list(fractions)
    target = {k: fractions[k] * n_total for k in names}
    count = {k: 0 for k in names}
    order = sorted(groups.items(),
                   key=lambda kv: (-len(kv[1]), hashlib.md5(f"{seed}:{kv[0]}".encode()).hexdigest()))
    assign = [None] * n_total
    for _, idxs in order:
        k = max(names, key=lambda k: (target[k] - count[k], -names.index(k)))
        for i in idxs:
            assign[i] = k
        count[k] += len(idxs)
    return assign


def split(d: pd.DataFrame) -> pd.DataFrame:
    d = d[d["is_kept"]].reset_index(drop=True)
    out = pd.Series(index=d.index, dtype=object)
    for idx in (d.index[~d.is_acyclic], d.index[d.is_acyclic]):
        if len(idx):
            out.loc[idx] = greedy_scaffold_split(d.loc[idx, "scaffold_group"].tolist(), FRACTIONS)
    d["split"] = out
    d["cv_fold"] = -1
    nontest = d.index[d.split != "test"]
    if d.loc[nontest, "scaffold_group"].nunique() >= N_CV_FOLDS:
        sub = d.loc[nontest]
        for fold, (_, va) in enumerate(GroupKFold(n_splits=N_CV_FOLDS).split(sub, groups=sub.scaffold_group)):
            d.loc[sub.index[va], "cv_fold"] = fold
    return d


def prepare(raw: pd.DataFrame, name: str, task_type: str) -> pd.DataFrame:
    d = raw.rename(columns={"Drug": "smiles_original"}).reset_index(drop=True)
    if "tdc_split" not in d:
        d["tdc_split"] = "full"
    d.insert(0, "row_uid", [f"{name}__{i}" for i in range(len(d))])
    d["dataset"], d["task_type"] = name, task_type
    ps = [parent_and_scaffold(s) for s in d["smiles_original"]]
    d["parent_smiles"] = [p for p, _ in ps]
    d["scaffold"] = [s for _, s in ps]
    scaf = d["scaffold"].fillna("")
    d["is_acyclic"] = scaf == ""
    d["scaffold_group"] = np.where(d.is_acyclic, "__ACYCLIC__" + d.row_uid, scaf)
    d = split(dedup_by_parent(d, task_type))
    if task_type == "classification":
        d["Y_final"] = d["Y_final"].astype(int)
    return d[SPLIT_COLS]


def main() -> int:
    parser = argparse.ArgumentParser(description="새 물성 분할 (담당1 규칙)")
    parser.add_argument("--raw", required=True, help="TDC 원본 표 (탭 구분)")
    parser.add_argument("--name", required=True)
    parser.add_argument("--task-type", required=True, choices=["classification", "regression"])
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    raw = pd.read_csv(args.raw, sep="\t")
    out = prepare(raw, args.name, args.task_type)
    target = Path(args.out_dir) / args.name
    target.mkdir(parents=True, exist_ok=True)
    out.to_csv(target / "splits.csv", index=False)
    print(f"{args.name}: 원본 {len(raw):,} → 남은 {len(out):,}  "
          + "  ".join(f"{k} {int((out.split == k).sum()):,}" for k in FRACTIONS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
