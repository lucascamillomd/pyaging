"""Opt-in prepared-clock reuse across independent datasets."""

import json
import weakref
from pathlib import Path
from unittest.mock import Mock

import anndata
import numpy as np
import pandas as pd
import pytest
import torch

import pyaging as pya

PARAMS_DIR = Path(__file__).resolve().parents[1] / "data" / "bioage_params"


def _phenoage_sao_paulo():
    params = json.loads((PARAMS_DIR / "phenoagesaopaulo.json").read_text())
    model = pya.models.PhenoAgeSaoPaulo()
    model.features = params["features"]
    model.metadata["clock_name"] = "phenoagesaopaulo"
    model.metadata["nested"] = {"source": ["original"]}
    model.base_model = pya.models.LinearModel(input_dim=len(model.features))
    model.base_model.linear.weight.data = torch.tensor([params["coefficients"]], dtype=torch.float64)
    model.base_model.linear.bias.data = torch.tensor([params["intercept"]], dtype=torch.float64)
    for name in ("m_n", "m_d", "ba_n", "ba_d", "ba_i"):
        setattr(model, name, torch.tensor(params[name], dtype=torch.float64))
    return model.eval().double()


def _cohort(rows):
    frame = pd.DataFrame(rows)
    return anndata.AnnData(frame.to_numpy(), var=pd.DataFrame(index=frame.columns))


def test_cache_reuses_prepared_model_across_datasets_and_preserves_reference_predictions(monkeypatch):
    reference = json.loads((PARAMS_DIR / "reference_predictions.json").read_text())
    load = Mock(side_effect=lambda *args: _phenoage_sao_paulo())
    monkeypatch.setattr("pyaging.predict._pred.load_clock", load)
    cache = pya.pred.ClockCache(maxsize=2)

    for rows, expected, batch_size in (
        (reference["rows"][:7], reference["expected"]["phenoagesaopaulo"][:7], 2),
        (reference["rows"][7:], reference["expected"]["phenoagesaopaulo"][7:], 3),
    ):
        adata = _cohort(rows)
        pya.pred.predict_age(adata, "PhenoAgeSaoPaulo", batch_size=batch_size, verbose=False, clock_cache=cache)
        np.testing.assert_allclose(adata.obs["phenoagesaopaulo"], expected, rtol=0, atol=1e-6)

    load.assert_called_once()


def test_predictions_do_not_share_mutable_metadata_with_cached_clock_or_other_datasets(monkeypatch):
    model = _phenoage_sao_paulo()
    monkeypatch.setattr("pyaging.predict._pred.load_clock", lambda *args: model)
    cache = pya.pred.ClockCache()
    rows = json.loads((PARAMS_DIR / "reference_predictions.json").read_text())["rows"][:1]
    first, second = _cohort(rows), _cohort(rows)

    pya.pred.predict_age(first, "phenoagesaopaulo", verbose=False, clock_cache=cache)
    first.uns["phenoagesaopaulo_metadata"]["nested"]["source"].append("caller change")
    pya.pred.predict_age(second, "phenoagesaopaulo", verbose=False, clock_cache=cache)

    assert model.metadata["nested"]["source"] == ["original"]
    assert second.uns["phenoagesaopaulo_metadata"]["nested"]["source"] == ["original"]


def test_repeated_predictions_load_separately_without_an_explicit_cache(monkeypatch):
    rows = json.loads((PARAMS_DIR / "reference_predictions.json").read_text())["rows"][:1]
    load = Mock(side_effect=lambda *args: _phenoage_sao_paulo())
    monkeypatch.setattr("pyaging.predict._pred.load_clock", load)

    for _ in range(2):
        pya.pred.predict_age(_cohort(rows), "phenoagesaopaulo", verbose=False)

    assert load.call_count == 2


@pytest.mark.parametrize("maxsize", [0, -1, 1.5, True])
def test_cache_size_must_be_a_positive_integer(maxsize):
    with pytest.raises(ValueError, match="maxsize.*positive integer"):
        pya.pred.ClockCache(maxsize=maxsize)


def test_cache_keys_normalize_names_and_separate_revisions_and_devices(monkeypatch):
    load = Mock(side_effect=lambda: torch.nn.Linear(1, 1))
    cache = pya.pred.ClockCache(maxsize=4)
    monkeypatch.setenv("PYAGING_DATA_REVISION", "v1")
    first = cache._get_or_load("Example", "cpu", load)
    assert cache._get_or_load("example", torch.device("cpu"), load) is first
    assert cache._get_or_load("example", "cuda:1", load) is not first
    monkeypatch.setenv("PYAGING_DATA_REVISION", "v2")
    assert cache._get_or_load("example", "cpu", load) is not first
    assert load.call_count == 3


def test_cache_evicts_the_least_recently_used_model_before_loading_replacement():
    cache = pya.pred.ClockCache(maxsize=2)
    load = Mock(side_effect=lambda: torch.nn.Linear(1, 1))
    first = weakref.ref(cache._get_or_load("first", "cpu", load))
    second = weakref.ref(cache._get_or_load("second", "cpu", load))
    assert cache._get_or_load("first", "cpu", load) is first()

    def replacement():
        assert second() is None
        return torch.nn.Linear(1, 1)

    cache._get_or_load("third", "cpu", replacement)
    assert first() is not None
    assert second() is None
    assert load.call_count == 2


def test_clear_releases_models_and_next_prediction_loads_again():
    cache = pya.pred.ClockCache()
    load = Mock(side_effect=lambda: torch.nn.Linear(1, 1))
    model = weakref.ref(cache._get_or_load("first", "cpu", load))

    cache.clear()

    assert model() is None
    cache._get_or_load("first", "cpu", load)
    assert load.call_count == 2


def test_failed_load_is_not_cached():
    cache = pya.pred.ClockCache()
    load = Mock(side_effect=[RuntimeError("unavailable"), torch.nn.Linear(1, 1)])
    with pytest.raises(RuntimeError, match="unavailable"):
        cache._get_or_load("first", "cpu", load)

    model = cache._get_or_load("first", "cpu", load)

    assert cache._get_or_load("first", "cpu", load) is model
    assert load.call_count == 2


def test_model_cache_does_not_reuse_cohort_preprocessing_between_datasets(monkeypatch):
    model = pya.models.TAge()
    model.metadata["clock_name"] = "cohort"
    model.features = ["g"]
    model.cohort_transform = "cache_test"
    model.base_model = torch.nn.Linear(1, 1, bias=False).double()
    model.base_model.weight.data.fill_(1)
    load = Mock(return_value=model)
    monkeypatch.setattr("pyaging.predict._pred.load_clock", load)

    def transform(adata, **kwargs):
        return pd.DataFrame(2 * adata.X, index=adata.obs_names, columns=["g"])

    transform = Mock(side_effect=transform)
    monkeypatch.setitem(pya.predict._pred_utils.COHORT_TRANSFORMS, "cache_test", transform)
    cache = pya.pred.ClockCache()
    for values in ([1.0, 3.0], [2.0, 4.0]):
        adata = anndata.AnnData(np.array(values).reshape(-1, 1))
        pya.pred.predict_age(adata, "cohort", clock_cache=cache, verbose=False)
        np.testing.assert_array_equal(adata.obs["cohort"], 2 * np.array(values))

    assert load.call_count == 1
    assert transform.call_count == 2
