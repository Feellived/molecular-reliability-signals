import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fingerprint_models import MorganConfig, make_morgan_matrix
from run_fingerprint_dataset import metrics


def test_morgan_matrix_is_binary_and_fixed_width():
    matrix = make_morgan_matrix(["CCO", "c1ccccc1"], MorganConfig(n_bits=256))
    assert matrix.shape == (2, 256)
    assert matrix.dtype == np.uint8
    assert set(np.unique(matrix)).issubset({0, 1})
    assert not np.array_equal(matrix[0], matrix[1])


def test_invalid_smiles_fails_with_position():
    with pytest.raises(ValueError, match="position 1"):
        make_morgan_matrix(["CCO", "not-a-smiles"])


def test_regression_metrics_are_available():
    result = metrics([1.0, 2.0, 3.0], [1.0, 2.5, 2.5], "regression")
    assert set(result) == {"rmse", "mae", "r2"}
    assert result["rmse"] > 0
