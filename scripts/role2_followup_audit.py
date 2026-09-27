"""담당 2 후속 점검용 입력·산출물 감사를 수행한다."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
except ImportError:  # 입력 manifest만 만들 때는 RDKit 없이도 실행 가능
    Chem = None


EVALUATION_ONLY = {
    "Y_final",
    "aps_true_pvalue",
    "aps_calibrated_margin",
    "conformal_true_score",
}
IDENTIFIERS = {"row_uid", "dataset", "task_type", "split"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_task(labels: pd.Series) -> str:
    values = set(pd.to_numeric(labels, errors="coerce").dropna().unique())
    return "classification" if len(values) <= 2 and values <= {0, 1} else "regression"


def unique_random_smiles(smiles: str, attempts: int) -> int:
    if Chem is None:
        raise RuntimeError("SMILES 다양성 점검에는 RDKit이 필요합니다")
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return 0
    values = {
        Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=True)
        for _ in range(attempts)
    }
    return len(values)


def audit_inputs(
    processed_dir: Path, out_dir: Path, variant_attempts: int, skip_variant_audit: bool
) -> list[str]:
    rows, diversity = [], []
    datasets = sorted(path.parent.name for path in processed_dir.glob("*/splits.csv"))
    for dataset in datasets:
        path = processed_dir / dataset / "splits.csv"
        frame = pd.read_csv(path, low_memory=False)
        counts = frame["split"].value_counts()
        meta = frame[frame["split"].eq("meta")]
        rows.append(
            {
                "dataset": dataset,
                "task_type": infer_task(frame["Y_final"]),
                "rows": len(frame),
                "train_rows": int(counts.get("train", 0)),
                "calib_rows": int(counts.get("calib", 0)),
                "meta_rows": int(counts.get("meta", 0)),
                "test_rows": int(counts.get("test", 0)),
                "meta_under_200": len(meta) < 200,
                "meta_cv_folds": int(meta["cv_fold"].nunique()),
                "row_uid_unique": bool(frame["row_uid"].is_unique),
                "sha256": sha256(path),
            }
        )
        if skip_variant_audit:
            continue
        train = frame[frame["split"].eq("train")]
        counts_unique = [
            unique_random_smiles(value, variant_attempts)
            for value in train["parent_smiles"].astype(str)
        ]
        diversity.append(
            {
                "dataset": dataset,
                "train_rows": len(train),
                "attempts_per_molecule": variant_attempts,
                "mean_unique_variants": float(np.mean(counts_unique)),
                "median_unique_variants": float(np.median(counts_unique)),
                "min_unique_variants": int(np.min(counts_unique)),
                "molecules_with_fewer_than_3": int(np.sum(np.asarray(counts_unique) < 3)),
                "fraction_with_fewer_than_3": float(np.mean(np.asarray(counts_unique) < 3)),
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "input_manifest.csv", index=False)
    if diversity:
        pd.DataFrame(diversity).to_csv(
            out_dir / "smiles_augmentation_diversity.csv", index=False
        )
    return datasets


def split_safe_outputs(datasets: list[str], output_dir: Path, out_dir: Path) -> dict:
    report = {"checked": 0, "missing": [], "leakage_columns": {}}
    safe_root = out_dir / "safe_signals"
    for dataset in datasets:
        path = output_dir / dataset / "role2_signals.csv"
        if not path.exists():
            report["missing"].append(dataset)
            continue
        frame = pd.read_csv(path, low_memory=False)
        present_eval = sorted((EVALUATION_ONLY | {"split"}) & set(frame.columns))
        report["leakage_columns"][dataset] = present_eval
        identifiers = [column for column in frame if column in IDENTIFIERS]
        evaluation = identifiers + [column for column in frame if column in EVALUATION_ONLY]
        features = [
            column
            for column in frame
            if column not in EVALUATION_ONLY
            and column not in {"split", "dataset", "task_type", "Y_final"}
        ]
        target = safe_root / dataset
        target.mkdir(parents=True, exist_ok=True)
        frame[features].to_csv(target / "role2_features.csv", index=False)
        frame[evaluation].to_csv(target / "role2_evaluation_only.csv", index=False)
        report["checked"] += 1
    (out_dir / "output_leakage_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--variant-attempts", type=int, default=30)
    parser.add_argument("--skip-variant-audit", action="store_true")
    args = parser.parse_args()
    datasets = audit_inputs(
        args.processed_dir,
        args.report_dir,
        args.variant_attempts,
        args.skip_variant_audit,
    )
    report = split_safe_outputs(datasets, args.output_dir, args.report_dir)
    print(f"입력 데이터셋: {len(datasets)}")
    print(f"안전한 feature 파일 생성: {report['checked']}")
    print(f"산출물 미발견: {len(report['missing'])}")


if __name__ == "__main__":
    main()
