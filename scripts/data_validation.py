"""전처리 데이터 로드와 누수·형식 검사를 담당한다."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem


REQUIRED_COLUMNS = {
    "row_uid",
    "dataset",
    "task_type",
    "parent_smiles",
    "Y_final",
    "split",
    "scaffold_group",
}
ALLOWED_SPLITS = {"train", "calib", "meta", "test"}
ALLOWED_TASK_TYPES = {"classification", "regression"}


@dataclass
class DatasetValidation:
    dataset: str
    path: str
    rows: int = 0
    task_type: str | None = None
    split_counts: dict[str, int] = field(default_factory=dict)
    missing_required_columns: list[str] = field(default_factory=list)
    missing_values: dict[str, int] = field(default_factory=dict)
    duplicate_row_uids: int = 0
    invalid_split_values: list[str] = field(default_factory=list)
    missing_splits: list[str] = field(default_factory=list)
    invalid_task_types: list[str] = field(default_factory=list)
    dataset_name_mismatches: int = 0
    invalid_smiles: int = 0
    split_leakage_row_uids: int = 0
    classification_labels: list[Any] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["valid"] = self.valid
        return result


def discover_split_files(processed_dir: Path) -> list[Path]:
    """각 물성 폴더의 splits.csv를 찾는다."""
    processed_dir = Path(processed_dir)
    return sorted(
        path
        for path in processed_dir.glob("*/splits.csv")
        if path.parent.name not in {"_cache", "reports"}
    )


def load_splits(path: Path) -> pd.DataFrame:
    """식별자와 라벨을 변경하지 않고 분할 파일을 읽는다."""
    return pd.read_csv(path, low_memory=False)


def _string_values(series: pd.Series) -> set[str]:
    return set(series.dropna().astype(str).str.strip())


def validate_split_file(path: Path, *, check_smiles: bool = True) -> DatasetValidation:
    path = Path(path)
    report = DatasetValidation(dataset=path.parent.name, path=str(path))
    try:
        frame = load_splits(path)
    except Exception as exc:  # 파일 또는 CSV 자체가 손상된 경우
        report.errors.append(f"CSV load failed: {exc}")
        return report

    report.rows = len(frame)
    report.missing_required_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if report.missing_required_columns:
        report.errors.append(
            "Missing required columns: " + ", ".join(report.missing_required_columns)
        )
        return report

    for column in sorted(REQUIRED_COLUMNS):
        count = int(frame[column].isna().sum())
        if count:
            report.missing_values[column] = count
    if report.missing_values:
        report.errors.append("Required columns contain missing values")

    report.duplicate_row_uids = int(frame["row_uid"].duplicated(keep=False).sum())
    if report.duplicate_row_uids:
        report.errors.append("row_uid is not unique")

    split_values = _string_values(frame["split"])
    report.invalid_split_values = sorted(split_values - ALLOWED_SPLITS)
    report.missing_splits = sorted(ALLOWED_SPLITS - split_values)
    report.split_counts = {
        name: int(count) for name, count in frame["split"].value_counts().items()
    }
    if report.invalid_split_values:
        report.errors.append("Invalid split values found")
    if report.missing_splits:
        report.errors.append("One or more required splits are absent")

    task_values = _string_values(frame["task_type"])
    report.invalid_task_types = sorted(task_values - ALLOWED_TASK_TYPES)
    if len(task_values) == 1:
        report.task_type = next(iter(task_values))
    else:
        report.errors.append("Each dataset must contain exactly one task_type")
    if report.invalid_task_types:
        report.errors.append("Invalid task_type values found")

    dataset_values = _string_values(frame["dataset"])
    report.dataset_name_mismatches = int((frame["dataset"].astype(str) != path.parent.name).sum())
    if dataset_values != {path.parent.name}:
        report.errors.append("dataset column does not match its directory name")

    uid_split_counts = frame.groupby("row_uid", dropna=False)["split"].nunique()
    report.split_leakage_row_uids = int((uid_split_counts > 1).sum())
    if report.split_leakage_row_uids:
        report.errors.append("row_uid appears in more than one split")

    if check_smiles:
        report.invalid_smiles = sum(
            Chem.MolFromSmiles(str(value)) is None
            for value in frame["parent_smiles"].dropna()
        )
        if report.invalid_smiles:
            report.errors.append("Invalid parent_smiles values found")

    if report.task_type == "classification":
        labels = sorted(frame["Y_final"].dropna().unique().tolist())
        report.classification_labels = labels
        if not set(labels).issubset({0, 0.0, 1, 1.0}):
            report.errors.append("Classification labels must be binary 0/1")
        if len(labels) < 2:
            report.warnings.append("Classification dataset contains only one label")

    for split_name in ALLOWED_SPLITS:
        if 0 < report.split_counts.get(split_name, 0) < 30:
            report.warnings.append(f"Very small {split_name} split (<30 rows)")

    return report


def validate_processed_dir(
    processed_dir: Path, *, check_smiles: bool = True
) -> list[DatasetValidation]:
    files = discover_split_files(processed_dir)
    if not files:
        raise FileNotFoundError(f"No dataset splits.csv files found under {processed_dir}")
    return [validate_split_file(path, check_smiles=check_smiles) for path in files]
