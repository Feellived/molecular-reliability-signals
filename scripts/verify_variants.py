#!/usr/bin/env python
"""변형 생성 산출물 검증.

generate_variants.py가 지켜야 할 규율을 실제 산출물에서 다시 확인한다.
생성 스크립트의 자체 보고를 믿지 않고 원본 분할 파일과 대조한다.

점검 항목

  1. 분할 상속    변형의 split·cv_fold·Y_final이 원본 행과 일치하는가
  2. 결합 무결성  모든 변형이 원본 행에 정확히 대응되는가
  3. 대상 분할    meta·test 외의 분할이 섞이지 않았는가
  4. 게이트 준수  07_axis_decision.csv에서 미사용 판정된 축이 생성되지 않았는가
  5. B2 부재      염 형태 변형이 생성되지 않았는가
  6. 축 의미론    A축은 원본과 같은 분자(InChIKey 동일)인가
                  B3축은 골격층은 같고 입체층만 다른가
  7. SMILES 유효  모든 변형이 파싱되는가
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

TARGET_SPLITS = {"meta", "test"}
SEMANTIC_SAMPLE = 400


def _inchikey(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def verify_dataset(
    dataset: str, variants_path: Path, splits_path: Path, flags: dict[str, bool]
) -> tuple[dict, list[str]]:
    variants = pd.read_csv(variants_path)
    splits = pd.read_csv(splits_path)
    problems: list[str] = []

    source = splits[["row_uid", "split", "cv_fold", "Y_final", "parent_smiles"]]
    merged = variants.merge(
        source, left_on="parent_row_uid", right_on="row_uid", suffixes=("", "_src")
    )

    if len(merged) != len(variants):
        problems.append(
            f"결합 무결성: 변형 {len(variants)}행 중 {len(variants) - len(merged)}행이 원본에 없다"
        )

    for column in ("split", "cv_fold", "Y_final"):
        mismatched = int((merged[column] != merged[f"{column}_src"]).sum())
        if mismatched:
            problems.append(f"분할 상속: {column} 불일치 {mismatched}행")

    stray = set(variants["split"].unique()) - TARGET_SPLITS
    if stray:
        problems.append(f"대상 분할: 예상 밖 분할 {sorted(stray)}")

    for axis, allowed in flags.items():
        produced = int((variants["axis"] == axis).sum())
        if not allowed and produced:
            problems.append(f"게이트 준수: {axis}는 미사용 판정인데 {produced}행 생성됨")

    if int((variants["axis"] == "B2_salt").sum()):
        problems.append("B2 부재: 염 형태 변형이 생성되었다")

    sample = merged.sample(
        min(SEMANTIC_SAMPLE, len(merged)), random_state=0
    ) if len(merged) else merged
    sample = sample.copy()
    sample["ik_variant"] = sample["variant_smiles"].map(_inchikey)
    sample["ik_parent"] = sample["parent_smiles"].map(_inchikey)

    n_unparseable = int(sample["ik_variant"].isna().sum())
    if n_unparseable:
        problems.append(f"SMILES 유효: 파싱 실패 {n_unparseable}건(표본 {len(sample)}중)")

    a_axis = sample[sample["axis"] == "A"].dropna(subset=["ik_variant", "ik_parent"])
    a_identical = (
        float((a_axis["ik_variant"] == a_axis["ik_parent"]).mean()) if len(a_axis) else float("nan")
    )
    if len(a_axis) and a_identical < 1.0:
        problems.append(
            f"축 의미론: A축 변형 중 {int((a_axis['ik_variant'] != a_axis['ik_parent']).sum())}건이 원본과 다른 분자다"
        )

    b3 = sample[sample["axis"] == "B3_stereo"].dropna(subset=["ik_variant", "ik_parent"])
    b3_skeleton = (
        float((b3["ik_variant"].str[:14] == b3["ik_parent"].str[:14]).mean())
        if len(b3)
        else float("nan")
    )
    if len(b3) and b3_skeleton < 1.0:
        problems.append("축 의미론: B3축 변형에서 골격층이 바뀐 건이 있다")

    stats = {
        "dataset": dataset,
        "n_variants": len(variants),
        "n_parents": int(variants["parent_row_uid"].nunique()),
        "a_axis_identical": round(a_identical, 4) if a_identical == a_identical else None,
        "b3_skeleton_identical": round(b3_skeleton, 4) if b3_skeleton == b3_skeleton else None,
        "n_problems": len(problems),
    }
    return stats, problems


def main() -> int:
    parser = argparse.ArgumentParser(description="변형 생성 산출물 검증")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--variants-dir", required=True)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    variants_dir = Path(args.variants_dir)

    decision = pd.read_csv(processed_dir / "reports" / "07_axis_decision.csv")
    flags_by_dataset = {
        row["dataset"]: {
            "B1_tautomer": row["use_B1_tautomer"] == "사용",
            "B1_protonation": row["use_B1_protonation"] == "사용",
            "B3_stereo": row["use_B3_stereo"] == "사용",
        }
        for _, row in decision.iterrows()
    }

    all_stats, all_problems = [], []
    for dataset in sorted(flags_by_dataset):
        variants_path = variants_dir / dataset / "variants.csv"
        if not variants_path.exists():
            all_problems.append(f"{dataset}: variants.csv 없음")
            continue
        stats, problems = verify_dataset(
            dataset,
            variants_path,
            processed_dir / dataset / "splits.csv",
            flags_by_dataset[dataset],
        )
        all_stats.append(stats)
        all_problems.extend(f"{dataset}: {p}" for p in problems)

    frame = pd.DataFrame(all_stats)
    print(frame.to_string(index=False))
    print()
    print(f"물성 {len(frame)}종, 변형 {int(frame['n_variants'].sum()):,}행")
    print()

    if all_problems:
        print(f"문제 {len(all_problems)}건")
        for problem in all_problems:
            print(f"  - {problem}")
        return 1

    print("모든 검증 항목 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
