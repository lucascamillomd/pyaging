"""KNN imputation must reproduce sklearn's KNNImputer(keep_empty_features=True)."""

import anndata
import numpy as np
import pytest

from pyaging.logger._live import DisplayLogger
from pyaging.preprocess import impute_missing_values

sklearn_impute = pytest.importorskip("sklearn.impute")


def _reference(X):
    return sklearn_impute.KNNImputer(keep_empty_features=True).fit_transform(X)


def _impute(X):
    adata = anndata.AnnData(np.array(X, copy=True))
    impute_missing_values(adata, "knn", DisplayLogger(lambda message: None))
    return adata.X


def _correlated(rng, n, p):
    base = rng.random((1, p))
    return np.clip(base + 0.1 * rng.standard_normal((n, p)) + 0.05 * rng.standard_normal((n, 1)), 0, 1)


def _random_missing(n, p, fraction, seed):
    rng = np.random.default_rng(seed)
    X = _correlated(rng, n, p)
    X[rng.random((n, p)) < fraction] = np.nan
    return X


def _block_missing():
    X = _correlated(np.random.default_rng(1), 30, 120)
    X[:10, :40] = np.nan
    X[10:20, 40:80] = np.nan
    X[20:, 80:100] = np.nan
    return X


def _few_donors():
    X = _correlated(np.random.default_rng(2), 12, 20)
    X[2:, 0] = np.nan  # two donors
    X[4:, 1] = np.nan  # four donors
    X[::3, 2:6] = np.nan
    return X


def _disjoint_halves():
    # The halves share no observed feature, so cross-half distances are NaN and receivers
    # whose only donors sit in the other half fall back to the column mean.
    X = _correlated(np.random.default_rng(3), 10, 20)
    X[:5, 10:] = np.nan
    X[5:, :10] = np.nan
    X[0, 3] = np.nan
    X[7, 15] = np.nan
    X[8, 2] = 0.25
    return X


def _all_missing_column():
    X = _random_missing(15, 30, 0.1, seed=4)
    X[:, 7] = np.nan
    return X


def _duplicated_samples():
    # Twins are at exactly tied distances from every receiver and agree on every value, so
    # any tie-breaking order gives the same mean.
    half = _random_missing(12, 40, 0.15, seed=5)
    return np.vstack([half, half])


def _sparse_columns_beyond_window():
    # More samples than the nearest-donor window, with columns observed by only a few
    # (often distant) samples, so the full donor list must be searched.
    X = _random_missing(120, 60, 0.05, seed=6)
    rng = np.random.default_rng(7)
    X[:, :20][rng.random((120, 20)) < 0.9] = np.nan
    return X


CASES = {
    "random_1pct": lambda: _random_missing(60, 300, 0.01, seed=8),
    "random_30pct": lambda: _random_missing(40, 200, 0.30, seed=9),
    "block": _block_missing,
    "fewer_than_k_donors": _few_donors,
    "four_samples": lambda: _random_missing(4, 25, 0.2, seed=10),
    "one_sample": lambda: _random_missing(1, 25, 0.2, seed=11),
    "disjoint_halves": _disjoint_halves,
    "all_missing_column": _all_missing_column,
    "duplicated_samples": _duplicated_samples,
    "sparse_columns_beyond_window": _sparse_columns_beyond_window,
}


@pytest.mark.parametrize("case", CASES)
def test_knn_matches_sklearn(case):
    X = CASES[case]()
    assert np.isnan(X).any()
    result = _impute(X)
    expected = _reference(X)
    assert result.dtype == expected.dtype
    assert not np.isnan(result).any()
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("case", ["random_30pct", "disjoint_halves", "sparse_columns_beyond_window"])
def test_knn_matches_sklearn_when_distances_are_computed_in_small_blocks(monkeypatch, case):
    from pyaging.preprocess import _preprocess_utils

    # Forces several feature blocks and several receiver blocks.
    monkeypatch.setattr(_preprocess_utils, "_KNN_BLOCK_ELEMENTS", 97)
    X = CASES[case]()
    np.testing.assert_allclose(_impute(X), _reference(X), rtol=0, atol=1e-12)


def test_knn_preserves_float32_and_matches_sklearn_to_float32_precision():
    X = _random_missing(30, 80, 0.1, seed=12).astype(np.float32)
    result = _impute(X)
    expected = _reference(X)
    assert result.dtype == np.float32 == expected.dtype
    # sklearn accumulates in float32; we accumulate in float64 and round once, so the two
    # agree to float32 resolution rather than to 1e-12.
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_knn_breaks_exact_distance_ties_by_lower_sample_index():
    # Six donors at the same distance from sample 0 disagree on the missing value. sklearn's
    # argpartition picks an arbitrary five; we take the five lowest-indexed donors.
    X = np.array([[0.0, np.nan]] + [[1.0, 10.0 * (i + 1)] for i in range(6)])
    result = _impute(X)
    assert result[0, 1] == pytest.approx(np.mean([10.0, 20.0, 30.0, 40.0, 50.0]))


def test_knn_does_not_modify_the_input_array():
    X = _random_missing(10, 20, 0.2, seed=13)
    original = X.copy()
    adata = anndata.AnnData(X)
    impute_missing_values(adata, "knn", DisplayLogger(lambda message: None))
    np.testing.assert_array_equal(X, original)
