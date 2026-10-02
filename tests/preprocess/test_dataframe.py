"""DataFrame preprocessing must preserve sample identity and original values."""

import warnings

import numpy as np
import pandas as pd
import pytest

from pyaging.preprocess import df_to_adata, epicv2_probe_aggregation


def test_dataframe_stays_on_host_when_cupy_is_available(monkeypatch):
    from types import SimpleNamespace

    import pyaging.preprocess._preprocess as preprocessing

    def reject_full_cohort_transfer(*args, **kwargs):
        raise AssertionError("Only inference batches should be transferred to the GPU")

    monkeypatch.setattr(preprocessing, "CUPY_AVAILABLE", True, raising=False)
    monkeypatch.setattr(preprocessing, "cp", SimpleNamespace(array=reject_full_cohort_transfer), raising=False)
    adata = df_to_adata(pd.DataFrame({"cg1": [0.1, 0.2]}), verbose=False)
    assert isinstance(adata.X, np.ndarray)
    np.testing.assert_array_equal(adata.X[:, 0], [0.1, 0.2])


@pytest.mark.parametrize("index", [[20, 10], pd.to_datetime(["2025-01-02", "2025-01-01"])])
def test_metadata_preserves_nonstring_sample_identifiers(index):
    frame = pd.DataFrame({"gene": [1.0, 2.0], "group": ["case", "control"]}, index=index)
    adata = df_to_adata(frame, metadata_cols=["group"], verbose=False)
    assert adata.obs_names.tolist() == frame.index.astype(str).tolist()
    assert adata.obs["group"].tolist() == ["case", "control"]
    pd.testing.assert_index_equal(frame.index, pd.Index(index))


def test_multiindex_sample_identifiers_preserve_metadata():
    index = pd.MultiIndex.from_tuples([("donor2", "t1"), ("donor1", "t2")], names=["donor", "time"])
    frame = pd.DataFrame({"gene": [1.0, 2.0], "group": ["case", "control"]}, index=index)
    adata = df_to_adata(frame, metadata_cols=["group"], verbose=False)
    assert adata.obs_names.tolist() == ["('donor2', 't1')", "('donor1', 't2')"]
    assert adata.obs["group"].tolist() == ["case", "control"]
    pd.testing.assert_index_equal(frame.index, index)


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean"])
def test_nullable_numeric_columns_are_imputed(dtype):
    frame = pd.DataFrame({"gene": pd.Series([1, None, 0], dtype=dtype)})
    adata = df_to_adata(frame, imputer_strategy="constant", verbose=False)
    np.testing.assert_array_equal(adata.X[:, 0], [1.0, 0.0, 0.0])
    assert np.isnan(adata.layers["X_original"][1, 0])
    assert adata.var.loc["gene", "percent_na"] == pytest.approx(1 / 3)


def test_mixed_numeric_and_boolean_columns_remain_numeric():
    frame = pd.DataFrame({"age": [30.0, 40.0], "female": [True, False]})
    adata = df_to_adata(frame, verbose=False)
    np.testing.assert_array_equal(adata.X, [[30.0, 1.0], [40.0, 0.0]])


def test_invalid_imputer_strategy_is_rejected_without_missing_values():
    with pytest.raises(ValueError, match="Invalid imputer strategy"):
        df_to_adata(pd.DataFrame({"gene": [1.0]}), imputer_strategy="typo", verbose=False)


def test_nonnumeric_features_explain_metadata_columns():
    with pytest.raises(TypeError, match="metadata_cols"):
        df_to_adata(pd.DataFrame({"gene": [1.0], "tissue": ["blood"]}), verbose=False)


@pytest.mark.parametrize("value", [np.inf, -np.inf, 1 + 2j])
def test_invalid_numeric_values_are_rejected(value):
    with pytest.raises(ValueError, match="finite|real"):
        df_to_adata(pd.DataFrame({"gene": [value]}), verbose=False)


def test_duplicate_feature_names_are_checked_before_dropping_empty_columns():
    frame = pd.DataFrame([[np.nan, 1.0]], columns=["gene", "gene"])
    with pytest.raises(ValueError, match="duplicate feature"):
        df_to_adata(frame, verbose=False)


@pytest.mark.parametrize("missing", [False, True])
def test_original_layer_is_a_snapshot_and_input_is_not_aliased(missing):
    values = [1.0, np.nan if missing else 3.0]
    frame = pd.DataFrame({"gene": values})
    adata = df_to_adata(frame, imputer_strategy="constant", verbose=False)
    adata.X[0, 0] = 9.0
    assert adata.layers["X_original"][0, 0] == 1.0
    assert frame.iloc[0, 0] == 1.0
    if missing:
        assert adata.layers["X_imputed"][0, 0] == 1.0


@pytest.mark.parametrize("shape", [(0, 2), (2, 0), (0, 0)])
def test_empty_frames_preserve_axes_without_runtime_warnings(shape):
    frame = pd.DataFrame(np.empty(shape), columns=[f"g{i}" for i in range(shape[1])])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        adata = df_to_adata(frame, verbose=False)
    assert adata.shape == shape
    assert adata.var["percent_na"].tolist() == [0.0] * shape[1]


def test_all_missing_features_are_dropped_without_runtime_warnings():
    frame = pd.DataFrame({"gene": [np.nan, np.nan]}, index=["s1", "s2"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        adata = df_to_adata(frame, verbose=False)
    assert adata.shape == (2, 0)
    assert adata.obs_names.tolist() == ["s1", "s2"]


def test_epic_suffixes_are_removed_even_without_replicate_probes():
    frame = pd.DataFrame({"cg00000001_BC11": [0.2], "cg00000002_TC11": [0.5]}, index=["s"])
    result = epicv2_probe_aggregation(frame, verbose=False)
    assert result.columns.tolist() == ["cg00000001", "cg00000002"]
    assert frame.columns.tolist() == ["cg00000001_BC11", "cg00000002_TC11"]
    np.testing.assert_array_equal(result, [[0.2, 0.5]])


def test_epic_duplicate_column_labels_are_averaged_once_each():
    frame = pd.DataFrame(
        [[0.1, 0.2, 0.8, 0.2], [np.nan, 0.4, 0.6, 0.8]],
        index=pd.Index(["s2", "s1"], name="sample"),
        columns=["cg2_TC11", "cg1_BC11", "cg1_BC11", "cg1_BC12"],
    )
    result = epicv2_probe_aggregation(frame, verbose=False)
    expected = pd.DataFrame({"cg2": [0.1, np.nan], "cg1": [0.4, 0.6]}, index=frame.index)
    pd.testing.assert_frame_equal(result, expected)


def test_knn_helper_retains_all_missing_columns():
    import anndata

    from pyaging.logger._live import DisplayLogger
    from pyaging.preprocess import impute_missing_values

    adata = anndata.AnnData(np.array([[1.0, np.nan], [2.0, np.nan]]))
    impute_missing_values(adata, "knn", DisplayLogger(lambda message: None))
    np.testing.assert_array_equal(adata.X, [[1.0, 0.0], [2.0, 0.0]])
