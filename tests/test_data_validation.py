from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from data_validation import discover_split_files, validate_split_file


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_uid": ["x1", "x2", "x3", "x4"],
            "dataset": ["demo"] * 4,
            "task_type": ["classification"] * 4,
            "parent_smiles": ["CCO", "CCN", "CCC", "CCCl"],
            "Y_final": [0, 1, 0, 1],
            "split": ["train", "calib", "meta", "test"],
            "scaffold_group": ["a", "b", "c", "d"],
        }
    )


def write_dataset(root: Path, name: str, frame: pd.DataFrame) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    path = directory / "splits.csv"
    frame.to_csv(path, index=False)
    return path


def test_valid_file(tmp_path):
    path = write_dataset(tmp_path, "demo", valid_frame())
    report = validate_split_file(path)
    assert report.valid
    assert report.split_leakage_row_uids == 0
    assert report.invalid_smiles == 0


def test_duplicate_uid_and_invalid_smiles_are_errors(tmp_path):
    frame = valid_frame()
    frame.loc[1, "row_uid"] = "x1"
    frame.loc[2, "parent_smiles"] = "not-a-smiles"
    path = write_dataset(tmp_path, "demo", frame)
    report = validate_split_file(path)
    assert not report.valid
    assert report.duplicate_row_uids == 2
    assert report.invalid_smiles == 1


def test_discovery_ignores_non_dataset_directories(tmp_path):
    write_dataset(tmp_path, "demo", valid_frame())
    write_dataset(tmp_path, "reports", valid_frame())
    assert [path.parent.name for path in discover_split_files(tmp_path)] == ["demo"]
