"""create_anndata_object's dtype validation and output contents."""

import numpy as np
import pandas as pd
import pytest

from pyaging.logger._live import DisplayLogger
from pyaging.preprocess import create_anndata_object

LOGGER = DisplayLogger(lambda message: None)


def test_nonnumeric_columns_are_listed_in_column_order():
    frame = pd.DataFrame(
        {
            "cg1": [0.1],
            "tissue": ["blood"],
            "cg2": [0.2],
            "batch": pd.Categorical(["a"]),
            "when": pd.to_datetime(["2025-01-01"]),
            "sex": ["F"],
        }
    )
    with pytest.raises(TypeError) as error:
        create_anndata_object(frame, LOGGER)
    assert str(error.value) == (
        "Feature columns must be numeric; move nonnumeric columns to metadata_cols: ['tissue', 'batch', 'when']"
    )


def test_nonnumeric_columns_are_reported_before_complex_ones():
    frame = pd.DataFrame({"z": [1 + 2j], "tissue": ["blood"]})
    with pytest.raises(TypeError, match=r"metadata_cols: \['tissue'\]"):
        create_anndata_object(frame, LOGGER)


def test_complex_columns_are_rejected():
    frame = pd.DataFrame({"cg1": [0.1], "z": [1 + 2j]})
    with pytest.raises(ValueError, match="^Feature values must be real numbers, not complex values.$"):
        create_anndata_object(frame, LOGGER)


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, bool])
def test_homogeneous_frames_keep_values_and_dtype(dtype):
    values = np.arange(12).reshape(3, 4) % 2
    frame = pd.DataFrame(values.astype(dtype), columns=list("abcd"))
    adata = create_anndata_object(frame, LOGGER)
    assert adata.X.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(adata.X, frame.to_numpy())
    np.testing.assert_array_equal(adata.layers["X_original"], frame.to_numpy())
    assert not np.shares_memory(adata.X, adata.layers["X_original"])
