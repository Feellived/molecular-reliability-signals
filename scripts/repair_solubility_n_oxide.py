"""AqSolDB의 구형 N-oxide SMILES 2개를 교정한다.

원문은 ``smiles_original``에 보존하고 모델 입력과 골격 정보만 수정한다.
같은 골격은 동일한 split과 CV fold에 배치해 누수를 방지한다.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi
from rdkit.Chem.Scaffolds import MurckoScaffold


REPAIRS = {
    "solubility_aqsoldb__9144": {
        "expected_source": "CC1=CC=C[NH+2]([O-])[CH-]1",
        "parent_smiles": "Cc1ccc[n+]([O-])c1",
        "inchi_key": "DMGGLIWGZFZLIY-UHFFFAOYSA-N",
    },
    "solubility_aqsoldb__9145": {
        "expected_source": "O=C(O)C1=C[NH+2]([O-])[CH-]C=C1",
        "parent_smiles": "O=C(O)c1ccc[n+]([O-])c1",
        "inchi_key": "FJCFFCXMEXZEIM-UHFFFAOYSA-N",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--apply", action="store_true", help="Write verified repairs")
    return parser.parse_args()


def canonical_and_scaffold(smiles: str) -> tuple[str, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Corrected SMILES does not parse: {smiles}")
    canonical = Chem.MolToSmiles(mol)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return canonical, scaffold


def main() -> int:
    args = parse_args()
    dataset_dir = args.processed_dir / "solubility_aqsoldb"
    molecule_path = dataset_dir / "molecules_labeled.csv"
    split_path = dataset_dir / "splits.csv"
    molecules = pd.read_csv(molecule_path, low_memory=False)
    splits = pd.read_csv(split_path, low_memory=False)

    audit = []
    target_scaffold = None
    for uid, repair in REPAIRS.items():
        canonical, scaffold = canonical_and_scaffold(repair["parent_smiles"])
        mol = Chem.MolFromSmiles(canonical)
        if inchi.MolToInchiKey(mol) != repair["inchi_key"]:
            raise ValueError(f"InChIKey mismatch for {uid}")
        if target_scaffold is None:
            target_scaffold = scaffold
        elif target_scaffold != scaffold:
            raise ValueError("The two repaired molecules should share one scaffold")

        for table_name, frame in (("molecules_labeled", molecules), ("splits", splits)):
            mask = frame["row_uid"].eq(uid)
            if int(mask.sum()) != 1:
                raise ValueError(f"Expected one {uid} row in {table_name}")
            row = frame.loc[mask].iloc[0]
            if row["smiles_original"] != repair["expected_source"]:
                raise ValueError(f"Unexpected source SMILES for {uid} in {table_name}")
            audit.append(
                {
                    "row_uid": uid,
                    "table": table_name,
                    "old_parent_smiles": row["parent_smiles"],
                    "new_parent_smiles": canonical,
                    "old_scaffold_group": row["scaffold_group"],
                    "new_scaffold_group": scaffold,
                }
            )
            frame.loc[mask, "parent_smiles"] = canonical
            frame.loc[mask, "scaffold"] = scaffold
            frame.loc[mask, "scaffold_group"] = scaffold

    peer = splits[
        splits["scaffold_group"].eq(target_scaffold)
        & ~splits["row_uid"].isin(REPAIRS)
    ]
    if peer.empty or set(peer["split"]) != {"train"} or peer["cv_fold"].nunique() != 1:
        raise ValueError("Existing repaired scaffold group is not train-only in one CV fold")
    peer_fold = int(peer["cv_fold"].iloc[0])
    target_mask = splits["row_uid"].isin(REPAIRS)
    if set(splits.loc[target_mask, "split"]) != {"train"}:
        raise ValueError("Repair unexpectedly requires a top-level split change")
    splits.loc[target_mask, "cv_fold"] = peer_fold

    if molecules["row_uid"].duplicated().any() or splits["row_uid"].duplicated().any():
        raise ValueError("row_uid duplication detected")
    if splits.groupby("scaffold_group")["split"].nunique().gt(1).any():
        raise ValueError("Scaffold leakage across top-level splits detected")
    if splits.groupby("scaffold_group")["cv_fold"].nunique().gt(1).any():
        raise ValueError("Scaffold leakage across CV folds detected")

    result = {
        "applied": args.apply,
        "target_scaffold": target_scaffold,
        "target_split": "train",
        "target_cv_fold": peer_fold,
        "rows_repaired": len(REPAIRS),
        "audit": audit,
    }
    if args.apply:
        audit_root = args.processed_dir / "_audit" / "role2"
        backup_dir = audit_root / "backups" / "solubility_aqsoldb"
        backup_dir.mkdir(parents=True, exist_ok=True)
        for path in (molecule_path, split_path):
            backup = backup_dir / path.name
            if not backup.exists():
                shutil.copy2(path, backup)
        molecules.to_csv(molecule_path, index=False)
        splits.to_csv(split_path, index=False)
        report_path = audit_root / "reports" / "smiles_repair.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
