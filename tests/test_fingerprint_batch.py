import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_fingerprint_batch import result_is_complete


def test_complete_result_requires_matching_unique_rows(tmp_path):
    split_path = tmp_path / "processed" / "demo" / "splits.csv"
    split_path.parent.mkdir(parents=True)
    pd.DataFrame({"row_uid": ["a", "b"]}).to_csv(split_path, index=False)

    output_dir = tmp_path / "outputs"
    dataset_dir = output_dir / "demo"
    dataset_dir.mkdir(parents=True)
    pd.DataFrame({"row_uid": ["a", "b"]}).to_csv(
        dataset_dir / "fingerprint_predictions.csv", index=False
    )
    (dataset_dir / "fingerprint_metrics.json").write_text(
        json.dumps({"dataset": "demo", "rows": 2}), encoding="utf-8"
    )

    assert result_is_complete("demo", split_path, output_dir)


def test_duplicate_prediction_uid_is_incomplete(tmp_path):
    split_path = tmp_path / "processed" / "demo" / "splits.csv"
    split_path.parent.mkdir(parents=True)
    pd.DataFrame({"row_uid": ["a", "b"]}).to_csv(split_path, index=False)

    output_dir = tmp_path / "outputs"
    dataset_dir = output_dir / "demo"
    dataset_dir.mkdir(parents=True)
    pd.DataFrame({"row_uid": ["a", "a"]}).to_csv(
        dataset_dir / "fingerprint_predictions.csv", index=False
    )
    (dataset_dir / "fingerprint_metrics.json").write_text(
        json.dumps({"dataset": "demo", "rows": 2}), encoding="utf-8"
    )

    assert not result_is_complete("demo", split_path, output_dir)
