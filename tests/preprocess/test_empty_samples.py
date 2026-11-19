"""Samples with no observed feature must not be imputed into plausible-looking samples."""

import warnings

import anndata
import numpy as np
import pandas as pd
import pytest

from pyaging.logger._live import DisplayLogger
from pyaging.preprocess import df_to_adata, impute_missing_values

STRATEGIES = ["knn", "mean", "median", "constant"]


def _frame():
    rng = np.random.default_rng(0)
    values = rng.random((8, 12))
    values[rng.random(values.shape) < 0.15] = np.nan
    values[:, 0] = np.linspace(0.1, 0.8, 8)  # keep every observed sample non-empty
    return pd.DataFrame(values, index=[f"s{i}" for i in range(8)], columns=[f"cg{i}" for i in range(12)])


def _with_empty(frame, names, position=3):
    empty = pd.DataFrame(np.nan, index=names, columns=frame.columns)
    return pd.concat([frame.iloc[:position], empty, frame.iloc[position:]])


@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_empty_samples_stay_missing_and_do_not_inform_the_fit(strategy, verbose):
    frame = _frame()
    reference = df_to_adata(frame, imputer_strategy=strategy, verbose=False)

    with pytest.warns(UserWarning, match=r"1 sample.*'dead'.*NaN predictions"):
        adata = df_to_adata(_with_empty(frame, ["dead"]), imputer_strategy=strategy, verbose=verbose)

    assert np.isnan(adata.X[3]).all()
    assert np.isnan(adata.layers["X_imputed"][3]).all()
    assert np.isnan(adata.layers["X_original"][3]).all()
    # Equal up to summation-order rounding: the fit sees the observed rows as a copy.
    observed = np.r_[0:3, 4:9]
    np.testing.assert_allclose(adata.X[observed], reference.X, rtol=0, atol=1e-12)
    np.testing.assert_allclose(adata.layers["X_imputed"][observed], reference.X, rtol=0, atol=1e-12)


def test_empty_samples_are_reported_once_with_at_most_five_names():
    names = [f"empty{i}" for i in range(7)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df_to_adata(_with_empty(_frame(), names), verbose=False)
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert len(messages) == 1
    assert "7 samples" in messages[0]
    assert all(name in messages[0] for name in names[:5])
    assert "empty5" not in messages[0] and "empty6" not in messages[0]


def test_empty_samples_alone_leave_other_samples_untouched():
    frame = _frame().fillna(0.5)
    with pytest.warns(UserWarning, match="NaN predictions"):
        adata = df_to_adata(_with_empty(frame, ["dead"]), verbose=False)
    np.testing.assert_array_equal(np.delete(adata.X, 3, axis=0), frame.to_numpy())
    assert np.isnan(adata.X[3]).all()
    assert "X_imputed" in adata.layers


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_imputation_helper_leaves_an_all_missing_matrix_missing(strategy):
    adata = anndata.AnnData(np.full((2, 3), np.nan))
    with pytest.warns(UserWarning, match="NaN predictions"):
        impute_missing_values(adata, strategy, DisplayLogger(lambda message: None))
    assert np.isnan(adata.X).all()
    assert np.isnan(adata.layers["X_imputed"]).all()


def test_knn_with_empty_samples_matches_sklearn_on_the_observed_samples():
    sklearn_impute = pytest.importorskip("sklearn.impute")
    rng = np.random.default_rng(1)
    X = rng.random((30, 50))
    X[rng.random(X.shape) < 0.2] = np.nan
    X[[4, 17]] = np.nan
    adata = anndata.AnnData(X.copy())
    with pytest.warns(UserWarning, match="NaN predictions"):
        impute_missing_values(adata, "knn", DisplayLogger(lambda message: None))
    keep = np.ones(30, dtype=bool)
    keep[[4, 17]] = False
    expected = sklearn_impute.KNNImputer(keep_empty_features=True).fit_transform(X[keep])
    np.testing.assert_allclose(adata.X[keep], expected, rtol=0, atol=1e-12)
    assert np.isnan(adata.X[~keep]).all()
