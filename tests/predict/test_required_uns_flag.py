"""A clock may demand a cohort-preprocessing marker in ``adata.uns``."""

import anndata
import numpy as np
import pytest
import torch

import pyaging as pya
from pyaging.models._base_models import pyagingModel


class _GuardedClock(pyagingModel):
    def __init__(self):
        super().__init__()
        self.required_uns_flag = "tage_prepared"

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


def _minimal_adata():
    return anndata.AnnData(X=np.zeros((3, 2)), var={"var_names": ["g1", "g2"]})


def _stub_clock(base_model):
    model = _GuardedClock()
    model.metadata["clock_name"] = "guarded"
    model.metadata["data_type"] = "transcriptomics"
    model.features = ["g1", "g2"]
    model.base_model = base_model
    return model


def test_base_model_defaults_to_no_flag():
    class _Plain(pyagingModel):
        def preprocess(self, x):
            return x

        def postprocess(self, x):
            return x

    assert _Plain().required_uns_flag is None


def test_guard_raises_without_uns_marker(monkeypatch):
    model = _stub_clock(torch.nn.Identity())
    monkeypatch.setattr("pyaging.predict._pred.load_clock", lambda *a, **k: model)
    adata = _minimal_adata()
    with pytest.raises(ValueError, match="prepare_tage"):
        pya.pred.predict_age(adata, ["guarded"], verbose=False)


def test_guard_passes_with_uns_marker(monkeypatch):
    model = _stub_clock(torch.nn.Linear(2, 1).double())
    monkeypatch.setattr("pyaging.predict._pred.load_clock", lambda *a, **k: model)
    adata = _minimal_adata()
    adata.uns["tage_prepared"] = True
    pya.pred.predict_age(adata, ["guarded"], verbose=False)
    assert "guarded" in adata.obs.columns


@pytest.mark.parametrize("clock_class", [pya.models.TAge, pya.models.TAgeMortality])
def test_tage_models_declare_the_guard(clock_class):
    model = clock_class()
    assert model.required_uns_flag == "tage_prepared"
    x = torch.ones(1, 3, dtype=torch.float64)
    assert torch.equal(model.postprocess(x), x)


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
