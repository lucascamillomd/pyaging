"""Replication clocks checked against original-author R reference predictions."""

import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData

from pyaging.logger import Logger
from pyaging.predict._pred_utils import check_features_in_adata


def _replication_models():
    # Keep an absent implementation a clear test failure during the red stage.
    assert importlib.util.find_spec("pyaging.models._replication"), "Replication clocks are not implemented"
    return importlib.import_module("pyaging.models._replication")


def test_celldrift_distinguishes_missing_probes_from_supplied_nan():
    model = _replication_models().CellDRIFT()
    model.metadata["clock_name"] = "celldrift"
    model.features = ["cg00000001", "cg00000002"]
    model.reference_values = [0.3, 0.7]
    model.base_model = torch.nn.Linear(2, 1).double()
    with torch.no_grad():
        model.base_model.weight.copy_(torch.tensor([[2.0, 3.0]], dtype=torch.float64))
        model.base_model.bias.fill_(5.0)
    adata = AnnData(pd.DataFrame([[np.nan], [0.2]], columns=["cg00000001"], index=["sample1", "sample2"]))
    check_features_in_adata(adata, model, Logger(level="ERROR"))
    matrix = torch.tensor(adata.obsm["X_celldrift"], dtype=torch.float64)
    original = matrix.clone()
    # The authors replace present NAs with zero, but missing probes with means.
    assert model(matrix).flatten().tolist() == pytest.approx([7.1, 7.5])
    torch.testing.assert_close(matrix, original, equal_nan=True)


FIXTURES = Path(__file__).resolve().parents[1] / "data/replication"


@pytest.fixture
def celldrift_model():
    from pyaging.models._base_models import LinearModel

    params = np.load(FIXTURES / "celldrift_parameters.npz")
    model = _replication_models().CellDRIFT().double()
    model.features = params["features"].tolist()
    model.reference_values = params["reference"].tolist()
    model.metadata.update(clock_name="celldrift", data_type="DNA methylation")
    model.base_model = LinearModel(len(model.features)).double()
    with torch.no_grad():
        model.base_model.linear.weight.copy_(torch.from_numpy(params["weight"]))
        model.base_model.linear.bias.copy_(torch.from_numpy(params["bias"]))
    return model.eval()


@pytest.mark.parametrize("batch_size", [1, 3, 64])
@pytest.mark.parametrize("partial", [False, True])
def test_celldrift_pipeline_matches_original_author_r(celldrift_model, batch_size, partial, monkeypatch, tmp_path):
    from pyaging.predict import _pred_utils, predict_age

    fixture = np.load(FIXTURES / "celldrift_input.npz")
    expected = pd.read_csv(FIXTURES / "celldrift_expected.csv", index_col=0)
    frame = pd.DataFrame(fixture["values"], index=fixture["samples"], columns=fixture["features"])
    dropped = frame.columns[::13].tolist() if partial else []
    # Reversed input order tests name-based alignment independently of weights.
    frame = frame.drop(columns=dropped).iloc[:, ::-1]
    adata = AnnData(frame)
    original = adata.X.copy()
    weights = tmp_path / "celldrift.pt"
    torch.save(celldrift_model, weights)
    monkeypatch.setattr(_pred_utils, "download_clock_weights", lambda *args, **kwargs: weights)
    predict_age(adata, "CellDRIFT", batch_size=batch_size, verbose=False)
    np.testing.assert_allclose(adata.obs["celldrift"], expected["partial" if partial else "full"], rtol=0, atol=1e-10)
    assert set(adata.uns["celldrift_missing_features"]) == set(dropped)
    np.testing.assert_array_equal(adata.X, original)


def test_celldrift_sparse_beta_boundaries_and_serialization(celldrift_model, tmp_path):
    from scipy.sparse import csr_matrix

    frame = pd.DataFrame(
        np.vstack([np.zeros(2322), np.ones(2322)]), index=["zero", "one"], columns=celldrift_model.features
    )
    adata = AnnData(
        csr_matrix(frame.to_numpy()), obs=pd.DataFrame(index=frame.index), var=pd.DataFrame(index=frame.columns)
    )
    check_features_in_adata(adata, celldrift_model, Logger(level="ERROR"))
    weights = tmp_path / "celldrift.pt"
    torch.save(celldrift_model, weights)
    restored = torch.load(weights, weights_only=False, map_location="cpu")
    result = restored(torch.tensor(adata.obsm["X_celldrift"], dtype=torch.float64)).flatten().detach().numpy()
    expected = pd.read_csv(FIXTURES / "celldrift_expected.csv", index_col=0).loc[["zero", "one"], "full"]
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-10)


@pytest.fixture
def miage_model():
    params = pd.read_csv(FIXTURES / "miage_parameters.csv", index_col=0, float_precision="round_trip")
    assert hasattr(_replication_models(), "MiAge"), "MiAge is not implemented"
    model = _replication_models().MiAge().double()
    model.features = params.index.tolist()
    model.reference_values = [float("nan")] * len(params)
    model.metadata.update(clock_name="miage", data_type="DNA methylation")
    model.set_parameters(params.b.to_numpy(), params.c.to_numpy(), params.d.to_numpy())
    return model.eval()


@pytest.mark.parametrize("batch_size", [1, 7, 64])
@pytest.mark.parametrize("partial", [False, True])
def test_miage_pipeline_matches_original_author_r(miage_model, batch_size, partial, monkeypatch, tmp_path):
    from pyaging.predict import _pred_utils, predict_age

    fixture = np.load(FIXTURES / "miage_input.npz")
    expected = pd.read_csv(FIXTURES / "miage_expected.csv", index_col=0)
    frame = pd.DataFrame(fixture["values"], index=fixture["samples"], columns=fixture["features"])
    dropped = frame.columns[::13].tolist() if partial else []
    frame = frame.drop(columns=dropped).iloc[:, ::-1]
    adata = AnnData(frame)
    original = adata.X.copy()
    weights = tmp_path / "miage.pt"
    torch.save(miage_model, weights)
    monkeypatch.setattr(_pred_utils, "download_clock_weights", lambda *args, **kwargs: weights)
    predict_age(adata, "MiAge", batch_size=batch_size, verbose=False)
    np.testing.assert_allclose(adata.obs["miage"], expected["partial" if partial else "full"], rtol=0, atol=1e-3)
    assert set(adata.uns["miage_missing_features"]) == set(dropped)
    np.testing.assert_array_equal(adata.X, original)


def test_miage_preserves_author_all_nan_start_and_bounded_solution(miage_model):
    values = torch.tensor([[float("nan")] * 268, [0.0] * 268, [1.0] * 268], dtype=torch.float64)
    result = miage_model(values)
    # The original R optimizer chooses its first start on an all-missing tie.
    assert result[0].item() == 2008.0
    assert torch.all((result >= 10.0) & (result <= 10000.0))
    assert result.dtype == values.dtype
    assert result.device == values.device


@pytest.mark.parametrize("invalid", [float("inf"), -float("inf")])
def test_miage_rejects_infinite_beta_instead_of_returning_initial_start(miage_model, invalid):
    values = torch.full((1, 268), 0.5, dtype=torch.float64)
    values[0, 7] = invalid
    with pytest.raises(ValueError, match="finite"):
        miage_model(values)
