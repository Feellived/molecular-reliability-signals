import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from recompute_regression_conformal_ensemble import (
    build_ensemble_conformal,
    load_multiseed_predictions,
    replace_conformal_columns,
)


def test_ensemble_center_scale_and_metadata_are_reproducible():
    source = pd.DataFrame(
        {
            "row_uid": ["a", "b", "c", "d"],
            "Y_final": [1.0, 2.0, 3.0, 4.0],
            "split": ["calib", "calib", "test", "test"],
        }
    )
    multiseed = pd.DataFrame(
        {
            "row_uid": ["d", "b", "a", "c"],
            "pred_chemberta_augmented_seed_42": [4.0, 2.0, 1.0, 3.0],
            "pred_chemberta_augmented_seed_43": [5.0, 3.0, 2.0, 4.0],
            "pred_chemberta_augmented_seed_44": [3.0, 1.0, 0.0, 2.0],
        }
    )

    result, metadata = build_ensemble_conformal(source, multiseed)

    np.testing.assert_allclose(result["conformal_center"], [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(result["conformal_ensemble_std"], np.sqrt(2 / 3))
    assert (result["conformal_scale"] > result["conformal_ensemble_std"]).all()
    assert metadata["ensemble_seeds"] == [42, 43, 44]
    assert metadata["ensemble_ddof"] == 0


def test_replace_conformal_columns_aligns_by_row_uid_and_keeps_other_signals():
    signals = pd.DataFrame(
        {
            "row_uid": ["b", "a"],
            "pred_rf": [0.2, 0.1],
            "conformal_lower": [-1.0, -1.0],
            "conformal_upper": [1.0, 1.0],
            "ad_score": [0.8, 0.9],
        }
    )
    conformal = pd.DataFrame(
        {
            "row_uid": ["a", "b"],
            "conformal_center": [10.0, 20.0],
            "conformal_lower": [9.0, 19.0],
            "conformal_upper": [11.0, 21.0],
        }
    )

    result = replace_conformal_columns(signals, conformal)

    assert result["row_uid"].tolist() == ["b", "a"]
    assert result["conformal_center"].tolist() == [20.0, 10.0]
    assert result["pred_rf"].tolist() == signals["pred_rf"].tolist()
    assert result["ad_score"].tolist() == signals["ad_score"].tolist()


def test_load_multiseed_predictions_supports_raw_seed_directories(tmp_path):
    for seed, values in [(42, [0.1, 0.2]), (43, [0.3, 0.4]), (44, [0.5, 0.6])]:
        output = tmp_path / "outputs" / f"seed_{seed}" / "sample"
        output.mkdir(parents=True)
        pd.DataFrame(
            {
                "row_uid": ["a", "b"],
                "pred_chemberta_augmented": values,
            }
        ).to_csv(output / "chemberta_augmented_predictions.csv", index=False)

    result = load_multiseed_predictions(tmp_path, "sample")

    assert result["row_uid"].tolist() == ["a", "b"]
    assert result["pred_chemberta_augmented_seed_42"].tolist() == [0.1, 0.2]
    assert result["pred_chemberta_augmented_seed_43"].tolist() == [0.3, 0.4]
    assert result["pred_chemberta_augmented_seed_44"].tolist() == [0.5, 0.6]
