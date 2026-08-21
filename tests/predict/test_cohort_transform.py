"""``predict_age`` runs a clock's declared cohort transform for it.

A cohort-relative clock cannot read a sample in isolation, so its feature
matrix is not sliced out of ``adata.X`` but computed from the whole cohort
first. The model declares ``cohort_transform``; ``predict_age`` resolves that
name to a transform, runs it once per call however many clocks share it, and
aligns each clock's features against the transformed frame.
"""

import anndata
import numpy as np
import pandas as pd
import pytest
import torch

import pyaging as pya
from pyaging.models._base_models import pyagingModel
from pyaging.predict import _pred_utils

TRANSFORMED = pd.DataFrame(
    {
        "101": [1.0, 2.0, 3.0],
        "102": [4.0, 5.0, 6.0],
        "103": [7.0, 8.0, 9.0],
    },
    index=["s0", "s1", "s2"],
)


class _CohortClock(pyagingModel):
    def __init__(self):
        super().__init__()
        self.cohort_transform = "fake"

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


def _raw_adata():
    a = anndata.AnnData(X=np.zeros((3, 2), dtype=np.float64))
    a.var_names = ["G1", "G2"]
    a.obs_names = ["s0", "s1", "s2"]
    return a


def _stub_clock(clock_name, features, reference_values=None):
    model = _CohortClock()
    model.metadata["clock_name"] = clock_name
    model.metadata["data_type"] = "transcriptomics (relative)"
    model.features = features
    model.reference_values = reference_values
    model.base_model = torch.nn.Linear(len(features), 1).double()
    return model


@pytest.fixture
def fake_transform(monkeypatch):
    """Register a counting stand-in under the transform name the stubs declare."""
    calls = []

    def transform(adata, dir="pyaging_data", logger=None):
        calls.append(adata)
        return TRANSFORMED

    monkeypatch.setitem(_pred_utils.COHORT_TRANSFORMS, "fake", transform)
    return calls


def _patch_clocks(monkeypatch, models):
    lookup = {model.metadata["clock_name"]: model for model in models}
    monkeypatch.setattr("pyaging.predict._pred.load_clock", lambda name, *a, **k: lookup[name])


def test_base_model_defaults_to_no_cohort_transform():
    class _Plain(pyagingModel):
        def preprocess(self, x):
            return x

        def postprocess(self, x):
            return x

    assert _Plain().cohort_transform is None


@pytest.mark.parametrize("clock_class", [pya.models.TAge, pya.models.TAgeMortality])
def test_tage_models_declare_the_tage_transform(clock_class):
    model = clock_class()
    assert model.cohort_transform == "tage"
    assert model.required_uns_flag is None


def test_tage_transform_name_is_registered():
    assert "tage" in _pred_utils.COHORT_TRANSFORMS


def test_feature_matrix_comes_from_the_transformed_frame(monkeypatch, fake_transform):
    model = _stub_clock("cohort", ["101", "103"])
    _patch_clocks(monkeypatch, [model])
    adata = _raw_adata()

    pya.pred.predict_age(adata, ["cohort"], clean=False, verbose=False)

    np.testing.assert_array_equal(
        adata.obsm["X_cohort"],
        TRANSFORMED[["101", "103"]].to_numpy(),
    )
    assert "cohort" in adata.obs.columns


def test_missing_features_take_the_reference_values(monkeypatch, fake_transform):
    model = _stub_clock("cohort", ["101", "999"], reference_values=[0.0, -7.0])
    _patch_clocks(monkeypatch, [model])
    adata = _raw_adata()

    pya.pred.predict_age(adata, ["cohort"], clean=False, verbose=False)

    np.testing.assert_array_equal(adata.obsm["X_cohort"][:, 0], TRANSFORMED["101"].to_numpy())
    np.testing.assert_array_equal(adata.obsm["X_cohort"][:, 1], np.full(3, -7.0))


def test_bookkeeping_reflects_the_transformed_gene_set(monkeypatch, fake_transform):
    model = _stub_clock("cohort", ["101", "999"], reference_values=[0.0, -7.0])
    _patch_clocks(monkeypatch, [model])
    adata = _raw_adata()

    pya.pred.predict_age(adata, ["cohort"], clean=False, verbose=False)

    assert adata.uns["cohort_missing_features"] == ["999"]
    assert adata.uns["cohort_percent_na"] == 50.0
    np.testing.assert_array_equal(
        adata.uns["cohort_supplied_features_mask"],
        np.array([True, False]),
    )


def test_transform_runs_once_for_two_clocks_sharing_it(monkeypatch, fake_transform):
    models = [_stub_clock("cohort_a", ["101"]), _stub_clock("cohort_b", ["102"])]
    _patch_clocks(monkeypatch, models)
    adata = _raw_adata()

    pya.pred.predict_age(adata, ["cohort_a", "cohort_b"], verbose=False)

    assert len(fake_transform) == 1
    assert set(adata.obs.columns) >= {"cohort_a", "cohort_b"}


def test_the_raw_matrix_is_left_alone(monkeypatch, fake_transform):
    model = _stub_clock("cohort", ["101"])
    _patch_clocks(monkeypatch, [model])
    adata = _raw_adata()
    original = adata.X.copy()

    pya.pred.predict_age(adata, ["cohort"], verbose=False)

    np.testing.assert_array_equal(adata.X, original)
    assert list(adata.var_names) == ["G1", "G2"]


def test_a_declared_transform_supersedes_a_stale_required_flag(monkeypatch, fake_transform):
    # Weights built before the transform existed still carry the guard flag; the
    # transform now supplies what the guard used to demand, so it must not fire.
    model = _stub_clock("cohort", ["101"])
    model.required_uns_flag = "tage_prepared"
    _patch_clocks(monkeypatch, [model])
    adata = _raw_adata()

    pya.pred.predict_age(adata, ["cohort"], verbose=False)

    assert "cohort" in adata.obs.columns


@pytest.mark.parametrize("clock_class", [pya.models.TAge, pya.models.TAgeMortality])
def test_tage_preprocess_substitutes_reference_values_for_absent_genes(clock_class):
    model = clock_class()
    model.reference_values = [1.0, 2.0, 3.0]
    x = torch.tensor([[float("nan"), 5.0, float("nan")]], dtype=torch.float64)
    assert torch.equal(model.preprocess(x), torch.tensor([[1.0, 5.0, 3.0]], dtype=torch.float64))


@pytest.mark.parametrize("clock_class", [pya.models.TAge, pya.models.TAgeMortality])
def test_tage_preprocess_is_a_no_op_without_reference_values(clock_class):
    model = clock_class()
    x = torch.tensor([[float("nan"), 5.0]], dtype=torch.float64)
    assert model.preprocess(x) is x


@pytest.mark.parametrize("clock_class", [pya.models.TAge, pya.models.TAgeMortality])
def test_tage_postprocess_is_the_identity(clock_class):
    x = torch.ones(1, 3, dtype=torch.float64)
    assert torch.equal(clock_class().postprocess(x), x)


def test_an_unknown_transform_name_raises(monkeypatch, fake_transform):
    model = _stub_clock("cohort", ["101"])
    model.cohort_transform = "nope"
    _patch_clocks(monkeypatch, [model])

    with pytest.raises(ValueError, match="nope"):
        pya.pred.predict_age(_raw_adata(), ["cohort"], verbose=False)
