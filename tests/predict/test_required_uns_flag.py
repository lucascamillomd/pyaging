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
