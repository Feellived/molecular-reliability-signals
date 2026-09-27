import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from role2_meta_oof import HANDOFF_COLUMNS, make_handoff_oof


def test_handoff_oof_has_exact_safe_columns():
    base = pd.DataFrame(
        {
            "row_uid": ["a", "b"],
            "cv_fold": [0.0, 1.0],
            "Y_final": [1, 0],
            "pred_rf_meta_fold_excluded": [0.8, 0.2],
            "pred_xgb_meta_fold_excluded": [0.7, 0.3],
            "std_rf_meta_fold_excluded": [0.1, 0.1],
        }
    )

    result = make_handoff_oof(base)

    assert result.columns.tolist() == HANDOFF_COLUMNS
    assert result["cv_fold"].tolist() == [0, 1]
    assert "Y_final" not in result


def test_handoff_oof_rejects_duplicate_uid():
    base = pd.DataFrame(
        {
            "row_uid": ["a", "a"],
            "cv_fold": [0, 1],
            "pred_rf_meta_fold_excluded": [0.8, 0.2],
            "pred_xgb_meta_fold_excluded": [0.7, 0.3],
        }
    )

    with pytest.raises(ValueError, match="row_uid"):
        make_handoff_oof(base)
