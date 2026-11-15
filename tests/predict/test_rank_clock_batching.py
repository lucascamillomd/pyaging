"""Rank clocks use one cohort median, regardless of inference batch size."""

from unittest.mock import Mock

import anndata
import numpy as np
import pandas as pd
import pytest
import torch

import pyaging as pya
from pyaging.predict._pred_utils import check_features_in_adata, predict_ages_with_model


def _model(clock_class):
    model = clock_class()
    model.metadata["clock_name"] = "rank"
    model.features = ["a", "b", "c"]
    model.reference_values = [np.nan] * 3
    if clock_class is pya.models.PastaMouse:
        model.set_mouse_features(["ENSMUSG1", "human1", "ENSMUSG2"], [np.nan, np.nan, np.nan])
    model.base_model = torch.nn.Linear(3, 1, bias=False).double()
    model.base_model.weight.data[:] = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float64)
    model.postprocess_dependencies = [1.0, 0.0] if isinstance(model, pya.models.Pasta) else [0.0]
    return model.eval()


@pytest.mark.parametrize("clock_class", [pya.models.Pasta, pya.models.Reg, pya.models.PastaMouse])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_rank_clock_batches_match_original_whole_cohort_prediction(clock_class, batch_size):
    model = _model(clock_class)
    if clock_class is pya.models.PastaMouse:
        # Complete mouse data still inserts a missing human reference column.
        values = np.array([[1.0, 4.0], [10.0, 12.0], [6.0, 7.0]])
    else:
        values = np.array([[1.0, np.nan, 4.0], [10.0, np.nan, 12.0], [6.0, np.nan, 7.0]])
    adata = anndata.AnnData(values.copy(), var=pd.DataFrame(index=model.features))
    check_features_in_adata(adata, model, Mock())
    with torch.inference_mode():
        original = model(torch.tensor(values, dtype=torch.float64))

    result = predict_ages_with_model(adata, model, "cpu", batch_size, Mock())

    torch.testing.assert_close(result, original, rtol=0, atol=0)
    np.testing.assert_array_equal(adata.obsm["X_rank"], values)


@pytest.mark.parametrize("clock_class", [pya.models.Pasta, pya.models.Reg, pya.models.PastaMouse])
def test_cached_rank_clock_recomputes_context_for_each_dataset(monkeypatch, clock_class):
    model = _model(clock_class)
    monkeypatch.setattr("pyaging.predict._pred.load_clock", lambda *args: model)
    cache = pya.pred.ClockCache()
    values = np.array([[1.0, np.nan, 4.0], [10.0, np.nan, 12.0], [6.0, np.nan, 7.0]])
    if clock_class is pya.models.PastaMouse:
        values = values[:, [0, 2]]

    for subset in (values, values[:1], values[1:]):
        adata = anndata.AnnData(subset.copy(), var=pd.DataFrame(index=model.features))
        with torch.inference_mode():
            expected = model(torch.tensor(subset, dtype=torch.float64)).numpy().ravel()
        pya.pred.predict_age(adata, "rank", batch_size=1, clock_cache=cache, verbose=False)
        np.testing.assert_array_equal(adata.obs["rank"], expected)
