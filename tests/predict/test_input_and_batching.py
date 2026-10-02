"""Prediction boundaries retain sample/feature identity and bounded batches."""

from unittest.mock import Mock

import anndata
import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse

import pyaging as pya
from pyaging.predict._pred_utils import check_features_in_adata, predict_ages_with_model
from pyaging.predict._transforms import quantile_normalize_with_gold_standard


def _clock():
    model = Mock()
    model.features = ["b", "missing", "a"]
    model.reference_values = [0.0, 7.0, 0.0]
    model.metadata = {"clock_name": "test"}
    model.preprocess_name = None
    model.postprocess_name = None
    return model


@pytest.mark.parametrize("matrix_type", [np.array, sparse.csr_matrix, sparse.csc_matrix])
def test_feature_alignment_accepts_sparse_without_changing_input(matrix_type):
    values = np.array([[1.0, 2.0, 99.0], [3.0, 4.0, 98.0]])
    adata = anndata.AnnData(matrix_type(values), var=pd.DataFrame(index=["a", "b", "unused"]))

    check_features_in_adata(adata, _clock(), Mock())

    np.testing.assert_array_equal(adata.obsm["X_test"], [[2.0, 7.0, 1.0], [4.0, 7.0, 3.0]])
    np.testing.assert_array_equal(adata.X.toarray() if sparse.issparse(adata.X) else adata.X, values)
    assert adata.uns["test_missing_features"] == ["missing"]
    assert adata.uns["test_percent_na"] == pytest.approx(100 / 3)


def test_duplicate_feature_names_do_not_silently_choose_a_measurement():
    adata = anndata.AnnData(np.array([[1.0, 9.0]]))
    adata.var_names = ["a", "a"]

    with pytest.raises(ValueError, match="unique"):
        check_features_in_adata(adata, _clock(), Mock())

    assert "X_test" not in adata.obsm


class _InPlaceClock(torch.nn.Module):
    metadata = {"clock_name": "test"}
    preprocess_name = "in-place transform"
    postprocess_name = None

    def forward(self, x):
        x.add_(10)
        return x.sum(dim=1, keepdim=True)


def test_in_place_preprocessing_does_not_mutate_retained_features():
    values = np.arange(15, dtype=np.float64).reshape(5, 3)
    adata = anndata.AnnData(values.copy())
    adata.obsm["X_test"] = values.copy()
    progress = Mock()

    result = predict_ages_with_model(adata, _InPlaceClock(), "cpu", 2, Mock(), progress_callback=progress)
    repeated = predict_ages_with_model(adata, _InPlaceClock(), "cpu", 5, Mock())

    np.testing.assert_array_equal(adata.obsm["X_test"], values)
    np.testing.assert_array_equal(result.numpy().ravel(), values.sum(axis=1) + 30)
    torch.testing.assert_close(repeated, result)
    assert [call.args for call in progress.call_args_list] == [(1, 3), (2, 3), (3, 3)]


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_invalid_batch_size_fails_before_loading_or_mutating(monkeypatch, batch_size):
    load = Mock()
    monkeypatch.setattr("pyaging.predict._pred.load_clock", load)
    adata = anndata.AnnData(np.ones((2, 1)))

    with pytest.raises(ValueError, match="batch_size.*positive integer"):
        pya.pred.predict_age(adata, "test", batch_size=batch_size, verbose=False)

    load.assert_not_called()
    assert not adata.obsm


def test_empty_cohort_fails_before_loading(monkeypatch):
    load = Mock()
    monkeypatch.setattr("pyaging.predict._pred.load_clock", load)
    adata = anndata.AnnData(np.empty((0, 1)))

    with pytest.raises(ValueError, match="at least one sample"):
        pya.pred.predict_age(adata, "test", verbose=False)

    load.assert_not_called()


def test_quantile_normalization_retains_fractional_reference_for_integer_counts():
    values = np.array([[8, 1, 3], [2, 9, 5]])
    reference = np.array([0.1, 1.5, 2.3])

    result = quantile_normalize_with_gold_standard(values, reference)

    np.testing.assert_allclose(result, [[2.3, 0.1, 1.5], [0.1, 2.3, 1.5]])
    np.testing.assert_array_equal(values, [[8, 1, 3], [2, 9, 5]])


def test_sparse_alignment_only_densifies_the_selected_features(monkeypatch):
    adata = anndata.AnnData(
        sparse.csr_matrix([[1.0, 2.0, 99.0], [3.0, 4.0, 98.0]]),
        var=pd.DataFrame(index=["a", "b", "unused"]),
    )
    shapes = []
    toarray = sparse.csr_matrix.toarray

    def record_toarray(matrix, *args, **kwargs):
        shapes.append(matrix.shape)
        return toarray(matrix, *args, **kwargs)

    monkeypatch.setattr(sparse.csr_matrix, "toarray", record_toarray)

    check_features_in_adata(adata, _clock(), Mock())

    assert shapes == [(2, 2)]


def test_cupy_input_uses_cpu_alignment_and_transfers_only_selected_features(monkeypatch):
    # Exercise device selection without requiring a CUDA runner. Real CuPy arrays
    # need an explicit asnumpy conversion; mere NumPy assignment cannot copy them.
    from types import SimpleNamespace

    from pyaging.predict import _pred_utils

    class GpuArray(np.ndarray):
        pass

    values = np.array([[1.0, 2.0, 99.0], [3.0, 4.0, 98.0]]).view(GpuArray)
    adata = anndata.AnnData(values, var=pd.DataFrame(index=["a", "b", "unused"]))
    asnumpy = Mock(side_effect=lambda value: np.asarray(value))
    allocate = Mock(side_effect=AssertionError("Full cohort must not be allocated on the GPU"))
    monkeypatch.setattr(_pred_utils, "CUPY_AVAILABLE", True)
    monkeypatch.setattr(
        _pred_utils, "cp", SimpleNamespace(ndarray=GpuArray, asnumpy=asnumpy, empty=allocate), raising=False
    )

    check_features_in_adata(adata, _clock(), Mock())

    allocate.assert_not_called()
    assert type(adata.obsm["X_test"]) is np.ndarray
    asnumpy.assert_called_once()
    assert asnumpy.call_args.args[0].shape == (2, 2)
    np.testing.assert_array_equal(adata.obsm["X_test"], [[2.0, 7.0, 1.0], [4.0, 7.0, 3.0]])


def test_previous_clock_is_released_before_loading_the_next(monkeypatch):
    import weakref

    references = []

    def load(name, *args, **kwargs):
        assert all(reference() is None for reference in references)
        model = _InPlaceClock()
        model.metadata = {"clock_name": name, "data_type": "transcriptomics"}
        model.features = ["a"]
        model.reference_values = None
        references.append(weakref.ref(model))
        return model

    monkeypatch.setattr("pyaging.predict._pred.load_clock", load)
    adata = anndata.AnnData(np.ones((2, 1)), var=pd.DataFrame(index=["a"]))

    pya.pred.predict_age(adata, ["first", "second"], verbose=False)

    assert all(reference() is None for reference in references)
    np.testing.assert_array_equal(adata.obs[["first", "second"]], [[11.0, 11.0], [11.0, 11.0]])
