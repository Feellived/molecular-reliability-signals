#!/usr/bin/env python
"""A축·B축 변형 분자 생성 (연구계획서 5.3절).

입력은 담당1이 확정한 분할 파일이고, 출력은 물성별 변형 목록이다.
변형은 meta·test 분할에만 생성한다. train은 모델 적합에만 쓰이므로
신뢰성 신호를 산출할 필요가 없다.

축 구성

  A            표현 불안정성. 같은 분자를 서로 다른 SMILES 문자열로 쓴다.
               분자 상태는 고정하고 표기만 바꾼다.
  B1_tautomer  호변이성질체. 상태를 바꾸고 표기는 정규형으로 고정한다.
  B1_protonation  양성자화 상태. 마찬가지로 표기는 정규형으로 고정한다.
  B3_stereo    입체 표기. 입체 주석이 있는 분자에서 주석을 제거한 형태.

A축은 표기만, B축은 상태만 바꾼다. 이 분리가 두 축을 구분하는 근거이므로
B축 변형은 정규 SMILES로 기록하고 A축 변형은 정규화하지 않는다.

B2(염 형태)는 생성하지 않는다. 담당1의 07_axis_decision.csv에서 22종 모두
'표본부족' 또는 '허용안됨'으로 판정되어 사용 대상이 아니다.

지켜야 할 규율

  1. 변형 행은 원본 행의 split과 cv_fold를 그대로 상속한다.
     변형이 분할 경계를 넘으면 누출이 된다.
  2. 변형 생성 후 표준화(Cleanup, LargestFragmentChooser 등)를 재적용하지
     않는다. 우리가 만든 변이를 되돌려버린다.
  3. 각 변형은 parent_row_uid로 원본에 묶인다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import pandas as pd
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

# 호변이성질체 상한은 eda_and_prescreen.py 및 게이트 판정과 같은 값을 쓴다.
# 이 값을 바꾸면 담당1의 B1 게이트 판정을 다시 계산해야 한다.
MAX_TAUTOMERS = 20

A_AXIS_K = 10
PH_MIN = 6.4
PH_MAX = 8.4
PH_PRECISION = 1.0
PROT_MAX_VARIANTS = 8

TARGET_SPLITS = ("meta", "test")
AXES = ("A", "B1_tautomer", "B1_protonation", "B3_stereo")

OUT_COLUMNS = [
    "variant_uid",
    "parent_row_uid",
    "dataset",
    "task_type",
    "split",
    "cv_fold",
    "Y_final",
    "axis",
    "variant_index",
    "variant_smiles",
    "equals_parent",
]

_TAUTOMER_ENUMERATOR = None


def _init_worker() -> None:
    global _TAUTOMER_ENUMERATOR
    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(MAX_TAUTOMERS)
    enumerator.SetRemoveSp3Stereo(False)
    enumerator.SetRemoveBondStereo(False)
    _TAUTOMER_ENUMERATOR = enumerator


def _seed_for(row_uid: str) -> int:
    """row_uid에서 결정적으로 난수 씨앗을 만든다. 재실행 시 같은 결과를 준다."""
    return int(hashlib.sha256(row_uid.encode("utf-8")).hexdigest()[:8], 16) % (2**31)


def _gen_a_axis(mol, seed: int, parent_canonical: str) -> tuple[list[str], int]:
    """같은 분자의 서로 다른 SMILES 표기. 정규화하지 않는다.

    무작위 표기는 원자 순서를 바꾸는데, 고리 안에 방향성 결합(/C=C\\)이 있는
    거대·고입체 분자에서는 이때 입체 표기가 어긋나는 경우가 드물게 생긴다.
    그런 변형은 원본과 다른 이성질체이므로 A축의 전제(같은 분자, 다른 표기)를
    깬다. 정규 SMILES로 되돌렸을 때 원본과 일치하는 것만 남긴다.

    되돌린 결과가 원본 정규형과 같다는 것이 곧 같은 분자라는 뜻이므로,
    이 검사는 올바른 변형을 잘못 버리지 않는다.
    """
    drawn = Chem.MolToRandomSmilesVect(mol, A_AXIS_K * 3, randomSeed=seed)
    seen, out, n_rejected = set(), [], 0
    for smi in drawn:
        if smi in seen:
            continue
        roundtrip = Chem.MolFromSmiles(smi)
        if roundtrip is None or Chem.MolToSmiles(roundtrip) != parent_canonical:
            n_rejected += 1
            continue
        seen.add(smi)
        out.append(smi)
        if len(out) >= A_AXIS_K:
            break
    return out, n_rejected


def _gen_b1_tautomer(mol) -> list[str]:
    try:
        tautomers = _TAUTOMER_ENUMERATOR.Enumerate(mol)
    except Exception:
        return []
    seen, out = set(), []
    for taut in tautomers:
        try:
            smi = Chem.MolToSmiles(taut)
        except Exception:
            continue
        if smi and smi not in seen:
            seen.add(smi)
            out.append(smi)
    return out


def _gen_b1_protonation(smiles: str) -> list[str]:
    from dimorphite_dl import protonate_smiles

    try:
        raw = protonate_smiles(
            smiles,
            ph_min=PH_MIN,
            ph_max=PH_MAX,
            precision=PH_PRECISION,
            max_variants=PROT_MAX_VARIANTS,
        )
    except Exception:
        return []
    seen, out = set(), []
    for smi in raw:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def _gen_b3_stereo(mol, parent_canonical: str) -> list[str]:
    """입체 주석을 제거한 형태. 입체가 없는 분자에서는 아무것도 만들지 않는다."""
    flat = Chem.Mol(mol)
    Chem.RemoveStereochemistry(flat)
    try:
        smi = Chem.MolToSmiles(flat)
    except Exception:
        return []
    return [smi] if smi and smi != parent_canonical else []


def _process_row(task: tuple) -> tuple[list[dict], dict]:
    (
        row_uid,
        dataset,
        task_type,
        split,
        cv_fold,
        y_final,
        parent_smiles,
        use_b1_tautomer,
        use_b1_protonation,
        use_b3_stereo,
    ) = task

    stats = {"n_failed_parse": 0, "n_invalid_variant": 0, "n_a_axis_rejected": 0}
    mol = Chem.MolFromSmiles(parent_smiles)
    if mol is None:
        stats["n_failed_parse"] = 1
        return [], stats

    parent_canonical = Chem.MolToSmiles(mol)
    seed = _seed_for(row_uid)

    a_variants, n_a_rejected = _gen_a_axis(mol, seed, parent_canonical)
    stats["n_a_axis_rejected"] = n_a_rejected
    by_axis: dict[str, list[str]] = {"A": a_variants}
    by_axis["B1_tautomer"] = _gen_b1_tautomer(mol) if use_b1_tautomer else []
    by_axis["B1_protonation"] = (
        _gen_b1_protonation(parent_smiles) if use_b1_protonation else []
    )
    by_axis["B3_stereo"] = (
        _gen_b3_stereo(mol, parent_canonical) if use_b3_stereo else []
    )

    rows = []
    for axis in AXES:
        for index, smiles in enumerate(by_axis[axis]):
            if Chem.MolFromSmiles(smiles) is None:
                stats["n_invalid_variant"] += 1
                continue
            rows.append(
                {
                    "variant_uid": f"{row_uid}__{axis}__{index:03d}",
                    "parent_row_uid": row_uid,
                    "dataset": dataset,
                    "task_type": task_type,
                    "split": split,
                    "cv_fold": cv_fold,
                    "Y_final": y_final,
                    "axis": axis,
                    "variant_index": index,
                    "variant_smiles": smiles,
                    "equals_parent": smiles == parent_canonical,
                }
            )
    return rows, stats


def _load_axis_decision(reports_dir: Path) -> dict[str, dict[str, bool]]:
    table = pd.read_csv(reports_dir / "07_axis_decision.csv")
    decision = {}
    for _, row in table.iterrows():
        decision[row["dataset"]] = {
            "B1_tautomer": row["use_B1_tautomer"] == "사용",
            "B1_protonation": row["use_B1_protonation"] == "사용",
            "B2_salt": row["use_B2_salt"] == "사용",
            "B3_stereo": row["use_B3_stereo"] == "사용",
        }
    return decision


def _process_dataset(
    dataset: str, splits_path: Path, flags: dict[str, bool], out_dir: Path, workers: int
) -> dict:
    splits = pd.read_csv(splits_path)
    target = splits[splits["split"].isin(TARGET_SPLITS)].copy()

    tasks = [
        (
            row["row_uid"],
            dataset,
            row["task_type"],
            row["split"],
            row["cv_fold"],
            row["Y_final"],
            row["parent_smiles"],
            flags["B1_tautomer"],
            flags["B1_protonation"],
            flags["B3_stereo"],
        )
        for _, row in target.iterrows()
    ]

    started = time.time()
    all_rows: list[dict] = []
    n_failed_parse = 0
    n_invalid_variant = 0
    n_a_axis_rejected = 0

    with mp.Pool(workers, initializer=_init_worker) as pool:
        for rows, stats in pool.imap_unordered(_process_row, tasks, chunksize=16):
            all_rows.extend(rows)
            n_failed_parse += stats["n_failed_parse"]
            n_invalid_variant += stats["n_invalid_variant"]
            n_a_axis_rejected += stats["n_a_axis_rejected"]

    frame = pd.DataFrame(all_rows, columns=OUT_COLUMNS)
    frame = frame.sort_values(["parent_row_uid", "axis", "variant_index"])

    dataset_dir = out_dir / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dataset_dir / "variants.csv", index=False)

    per_axis = {axis: int((frame["axis"] == axis).sum()) for axis in AXES}
    summary = {
        "dataset": dataset,
        "n_parent_molecules": len(target),
        "n_variants_total": len(frame),
        "elapsed_sec": round(time.time() - started, 1),
        "n_failed_parse": n_failed_parse,
        "n_invalid_variant": n_invalid_variant,
        "n_a_axis_rejected": n_a_axis_rejected,
        **{f"n_{axis}": per_axis[axis] for axis in AXES},
        **{f"use_{axis}": flags[axis] for axis in ("B1_tautomer", "B1_protonation", "B3_stereo")},
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="A축·B축 변형 분자 생성")
    parser.add_argument("--processed-dir", required=True, help="담당1 분할 산출물 최상위")
    parser.add_argument("--out-dir", required=True, help="변형 출력 최상위")
    parser.add_argument("--datasets", nargs="*", default=None, help="일부만 처리할 때")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    parser.add_argument("--resume", action="store_true", help="이미 만든 물성은 건너뛴다")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out_dir)
    reports_dir = processed_dir / "reports"

    decision = _load_axis_decision(reports_dir)
    datasets = args.datasets or sorted(decision)

    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for i, dataset in enumerate(datasets, 1):
        splits_path = processed_dir / dataset / "splits.csv"
        if not splits_path.exists():
            print(f"[{i}/{len(datasets)}] {dataset}: splits.csv 없음, 건너뜀", flush=True)
            continue
        if args.resume and (out_dir / dataset / "variants.csv").exists():
            print(f"[{i}/{len(datasets)}] {dataset}: 이미 있음, 건너뜀", flush=True)
            continue

        flags = decision[dataset]
        summary = _process_dataset(dataset, splits_path, flags, out_dir, args.workers)
        summaries.append(summary)
        print(
            f"[{i}/{len(datasets)}] {dataset}: "
            f"분자 {summary['n_parent_molecules']:,} → 변형 {summary['n_variants_total']:,} "
            f"(A {summary['n_A']:,} / 호변 {summary['n_B1_tautomer']:,} / "
            f"양성자 {summary['n_B1_protonation']:,} / 입체 {summary['n_B3_stereo']:,}) "
            f"{summary['elapsed_sec']}초",
            flush=True,
        )

    if not summaries:
        print("새로 생성한 물성이 없다.")
        return 0

    summary_dir = out_dir / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(summary_dir / "variant_summary.csv", index=False)

    from importlib.metadata import version as _pkg_version  # noqa: PLC0415

    try:
        dimorphite_version = _pkg_version("dimorphite_dl")
    except Exception:
        dimorphite_version = "unknown"

    metadata = {
        "generated_by": "juhyeong (담당4가 담당3 작업 대행)",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_splits": str(processed_dir),
        "target_splits": list(TARGET_SPLITS),
        "axes": list(AXES),
        "parameters": {
            "max_tautomers": MAX_TAUTOMERS,
            "a_axis_k": A_AXIS_K,
            "ph_min": PH_MIN,
            "ph_max": PH_MAX,
            "ph_precision": PH_PRECISION,
            "protonation_max_variants": PROT_MAX_VARIANTS,
        },
        "versions": {
            "rdkit": rdkit.__version__,
            "dimorphite_dl": dimorphite_version,
            "python": sys.version.split()[0],
        },
        "guards": [
            "변형 행은 원본의 split과 cv_fold를 상속한다",
            "변형 생성 후 표준화를 재적용하지 않는다",
            "B2(염 형태)는 게이트 판정에 따라 생성하지 않는다",
            "A축은 정규화하지 않고 B축은 정규 SMILES로 기록한다",
            "A축 변형은 정규 SMILES 왕복이 원본과 일치하는 것만 남긴다",
        ],
        "totals": {
            "n_datasets": len(summary_frame),
            "n_parent_molecules": int(summary_frame["n_parent_molecules"].sum()),
            "n_variants_total": int(summary_frame["n_variants_total"].sum()),
            "n_a_axis_rejected": int(summary_frame["n_a_axis_rejected"].sum()),
            **{
                f"n_{axis}": int(summary_frame[f"n_{axis}"].sum()) for axis in AXES
            },
        },
    }
    (summary_dir / "generation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print()
    print(
        f"완료: {metadata['totals']['n_datasets']}종, "
        f"분자 {metadata['totals']['n_parent_molecules']:,} → "
        f"변형 {metadata['totals']['n_variants_total']:,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
