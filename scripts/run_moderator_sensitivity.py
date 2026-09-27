#!/usr/bin/env python
"""물성의 상태 민감도와 B축 효과 (docs/사전기록_1_민감도_조절분석.md).

지표 정의와 검정은 사전 기록 문서를 그대로 따른다. 변형 생성과 같은 설정을
보장하려고 generate_variants.py의 열거 함수를 직접 불러 쓴다.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy.stats import mannwhitneyu, spearmanr

import generate_variants as gv
from dataset_repairs import apply_known_repairs

RDLogger.DisableLog("rdApp.*")
MIN_GROUP_SHARE = 0.05
N_PERMUTATIONS = 10_000
PERMUTATION_SEED = 20260928


def _n_charged(mol) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() != 0)


def _flags(smiles: str) -> tuple[bool, bool] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    base = _n_charged(mol)
    ionizable = False
    for state in gv._gen_b1_protonation(smiles):
        other = Chem.MolFromSmiles(state)
        if other is not None and _n_charged(other) != base:
            ionizable = True
            break
    canonical = Chem.MolToSmiles(mol)
    tautomerizable = any(s != canonical for s in gv._gen_b1_tautomer(mol))
    return ionizable, tautomerizable


def _init() -> None:
    gv._init_worker()


def rank_biserial(labels: np.ndarray, flag: np.ndarray) -> float:
    """두 집단 라벨의 순위 이연 상관 절댓값. 한 집단이 5% 미만이면 결측."""
    share = flag.mean()
    if share < MIN_GROUP_SHARE or share > 1 - MIN_GROUP_SHARE:
        return float("nan")
    a, b = labels[flag], labels[~flag]
    u = mannwhitneyu(a, b, alternative="two-sided").statistic
    return abs(2 * u / (len(a) * len(b)) - 1)


def sensitivity(splits_path: Path, workers: int) -> dict:
    frame = pd.read_csv(splits_path, low_memory=False)
    frame, _ = apply_known_repairs(frame)
    with mp.Pool(workers, initializer=_init) as pool:
        flags = pool.map(_flags, frame["parent_smiles"].tolist(), chunksize=64)
    keep = [f is not None for f in flags]
    labels = pd.to_numeric(frame.loc[keep, "Y_final"]).to_numpy(float)
    ion = np.array([f[0] for f in flags if f is not None])
    taut = np.array([f[1] for f in flags if f is not None])
    return {
        "n": int(len(labels)),
        "share_ionizable": float(ion.mean()),
        "share_tautomerizable": float(taut.mean()),
        "S_ion": rank_biserial(labels, ion),
        "S_taut": rank_biserial(labels, taut),
    }


def permutation_test(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """스피어만 ρ와 단측(ρ > 0) 순열 p. 결측은 뺀다."""
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    rho = float(spearmanr(x, y)[0])
    rng = np.random.default_rng(PERMUTATION_SEED)
    null = np.array([spearmanr(x, rng.permutation(y))[0] for _ in range(N_PERMUTATIONS)])
    p = float((np.sum(null >= rho) + 1) / (N_PERMUTATIONS + 1))
    return rho, p, int(ok.sum())


def main() -> int:
    parser = argparse.ArgumentParser(description="상태 민감도 조절 분석")
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--ablation", required=True, help="preregistered_ablation.csv")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--extra-splits", nargs="*", default=[],
                        help="이름=경로. B 효과 없이 지표만 계산할 새 데이터")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ablation = pd.read_csv(args.ablation).set_index("dataset")
    effect = ablation["aurc__기준(AD+컨포멀)"] - ablation["aurc__기준+B"]

    rows = []
    for dataset in effect.index:
        record = {"dataset": dataset, **sensitivity(Path(args.splits_dir) / dataset / "splits.csv", args.workers),
                  "b_effect": float(effect[dataset]), "cyp": dataset.startswith("cyp")}
        rows.append(record)
        print(f"  {dataset:32s} S_ion {record['S_ion']:.3f}  S_taut {record['S_taut']:.3f}", flush=True)
    for item in args.extra_splits:
        name, path = item.split("=", 1)
        record = {"dataset": name, **sensitivity(Path(path), args.workers), "b_effect": np.nan,
                  "cyp": name.startswith("cyp")}
        rows.append(record)
        print(f"  {name:32s} S_ion {record['S_ion']:.3f}  S_taut {record['S_taut']:.3f}  (새 데이터)", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "moderator_by_dataset.csv", index=False)
    known = table[table["b_effect"].notna()]
    tests = {}
    for metric in ("S_ion", "S_taut"):
        for label, subset in (("비CYP 16종", known[~known.cyp]), ("22종", known)):
            rho, p, n = permutation_test(subset[metric].to_numpy(float), subset["b_effect"].to_numpy(float))
            tests[f"{metric} | {label}"] = {"rho": rho, "p_one_sided": p, "n": n}
    (out_dir / "moderator_tests.json").write_text(json.dumps(tests, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(tests, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
