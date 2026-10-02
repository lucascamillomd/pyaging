"""Original-author oracle regressions for PCBrainAge and PCGrimAge proxies."""

import importlib
import importlib.util

import pytest
import torch

MODEL_NAMES = [
    "PCBrainAge",
    "PCGrimAgePackYrs",
    "PCGrimAgeADM",
    "PCGrimAgeB2M",
    "PCGrimAgeCystatinC",
    "PCGrimAgeGDF15",
    "PCGrimAgeLeptin",
    "PCGrimAgePAI1",
    "PCGrimAgeTIMP1",
]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_pc_clock_reference_imputation_and_batch_predictions(name):
    # Catches absent runtime classes, changed imputation, and nonlinear transforms.
    assert importlib.util.find_spec("pyaging.models._pc_extensions") is not None
    cls = getattr(importlib.import_module("pyaging.models._pc_extensions"), name)
    model = cls().double()
    model.reference_values = [0.2, 0.7]
    model.base_model = torch.nn.Linear(2, 1).double()
    with torch.no_grad():
        model.base_model.weight.copy_(torch.tensor([[2.0, -3.0]], dtype=torch.float64))
        model.base_model.bias.fill_(5.0)
    inputs = torch.tensor([[0.5, 0.1], [float("nan"), 0.4]], dtype=torch.float64)
    expected = torch.tensor([[5.7], [4.2]], dtype=torch.float64)
    torch.testing.assert_close(model(inputs), expected)
    torch.testing.assert_close(torch.cat([model(row[None]) for row in inputs]), expected)
    assert torch.isnan(inputs[1, 0])


def _builder():
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "clocks/build_pc_extensions.py"
    spec = importlib.util.spec_from_file_location("build_pc_extensions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _oracle_inputs(model):
    import numpy as np

    features = model.features
    n_cpgs = sum(feature not in {"female", "age"} for feature in features)
    ref = np.asarray(model.reference_values[:n_cpgs])
    j = np.arange(n_cpgs, dtype=np.int64)
    x = np.stack(
        [
            ref,
            np.full(n_cpgs, 0.5),
            np.zeros(n_cpgs),
            np.ones(n_cpgs),
            ((j * 37) % 1000) / 1000,
            ((j * 91 + 17) % 1000) / 1000,
            ref.copy(),
            ((j * 37) % 1000) / 1000,
        ]
    )
    x[6, [0, n_cpgs // 2, n_cpgs - 1]] = [0.1, 0.9, 0.3]
    x[7, ::97] = float("nan")
    covariates = {
        "female": [0, 1, 0, 1, 0, 1, 0, 1],
        "age": [20, 35, 50, 65, 80, 95, 45, 70],
    }
    if len(features) > n_cpgs:
        x = np.column_stack([x] + [covariates[f] for f in features[n_cpgs:]])
    return torch.from_numpy(x)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_pc_extension_matches_original_r_predictions_and_roundtrip(name, tmp_path):
    # A lost centering term, shifted CpG, rounded coefficient, or ignored sex/age
    # fails fixed predictions executed by the original authors' R expressions.
    import json

    builder = _builder()
    assert (builder.DATA / "coefficients.npz").exists(), "Author source export has not been generated"
    model = builder.build_model(name.lower())
    expected = json.loads((builder.DATA / "oracle.json").read_text())[name.lower()]
    inputs = _oracle_inputs(model)
    actual = model(inputs).flatten()
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64), rtol=2e-11, atol=2e-8)
    batched = torch.cat([model(row[None]) for row in inputs]).flatten()
    torch.testing.assert_close(actual, batched, rtol=2e-11, atol=2e-8)
    path = tmp_path / "model.pt"
    torch.save(model, path)
    reloaded = torch.load(path, weights_only=False)
    torch.testing.assert_close(reloaded(inputs), model(inputs), rtol=0, atol=0)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_pc_extension_pipeline_aligns_and_reports_missing_cpgs(name, tmp_path, monkeypatch):
    # Catches a mismatch between package feature alignment and original reference
    # substitution. Only the download boundary is replaced with a local artifact.
    import json

    import anndata
    import numpy as np

    import pyaging as pya
    from pyaging.predict import _pred_utils

    builder = _builder()
    model = builder.build_model(name.lower())
    x = _oracle_inputs(model)[7].numpy()
    keep = np.flatnonzero(~np.isnan(x))[::-1]
    adata = anndata.AnnData(X=x[None, keep])
    adata.var_names = [model.features[i] for i in keep]
    adata.obs_names = ["sample-with-missing-probes"]
    path = tmp_path / f"{name.lower()}.pt"
    torch.save(model, path)
    monkeypatch.setattr(_pred_utils, "download_clock_weights", lambda *args, **kwargs: str(path))
    assert pya.pred.predict_age(adata, name, batch_size=1, verbose=False) is None
    expected = json.loads((builder.DATA / "oracle.json").read_text())[name.lower()][7]
    assert adata.obs[name.lower()].iloc[0] == pytest.approx(expected, rel=2e-11, abs=2e-8)
    missing = [f for i, f in enumerate(model.features) if np.isnan(x[i])]
    assert set(adata.uns[f"{name.lower()}_missing_features"]) == set(missing)
    assert adata.uns[f"{name.lower()}_percent_na"] == pytest.approx(100 * len(missing) / len(model.features))
    assert adata.obs_names.tolist() == ["sample-with-missing-probes"]


@pytest.mark.parametrize("name", MODEL_NAMES[1:])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_pc_proxy_rejects_missing_selected_demographics(name, bad_value):
    cls = getattr(importlib.import_module("pyaging.models._pc_extensions"), name)
    model = cls().double()
    model.features = ["cg00000001", "female", "age"]
    model.required_covariates = ["female", "age"]
    model.required_covariate_indices = torch.tensor([1, 2])
    model.reference_values = [0.2, float("nan"), float("nan")]
    model.base_model = torch.nn.Linear(3, 1).double()
    for column, required in [(1, "female"), (2, "age")]:
        row = torch.tensor([[0.5, 1.0, 65.0]], dtype=torch.float64)
        row[0, column] = bad_value
        with pytest.raises(ValueError, match=required):
            model(row)


@pytest.mark.parametrize("name", MODEL_NAMES[1:])
def test_pc_proxy_imputes_cpgs_without_changing_supplied_demographics(name):
    cls = getattr(importlib.import_module("pyaging.models._pc_extensions"), name)
    model = cls().double()
    model.features = ["cg00000001", "female", "age"]
    model.required_covariates = ["female", "age"]
    model.required_covariate_indices = torch.tensor([1, 2])
    model.reference_values = [0.2, float("nan"), float("nan")]
    model.base_model = torch.nn.Linear(3, 1).double()
    with torch.no_grad():
        model.base_model.weight.copy_(torch.tensor([[2.0, 3.0, 4.0]], dtype=torch.float64))
        model.base_model.bias.zero_()
    row = torch.tensor([[float("nan"), 0.0, 50.0]], dtype=torch.float64)
    assert model(row).item() == pytest.approx(200.4)


@pytest.mark.parametrize("name", MODEL_NAMES[1:])
def test_pc_proxy_pipeline_rejects_absent_original_demographic(name, tmp_path, monkeypatch):
    import anndata
    import numpy as np

    import pyaging as pya
    from pyaging.predict import _pred_utils

    model = _builder().build_model(name.lower())
    if not model.required_covariates:
        return  # The original component selected methylation PCs only.
    missing_name = model.required_covariates[0]
    x = _oracle_inputs(model)[4].numpy()
    keep = [i for i, feature in enumerate(model.features) if feature != missing_name]
    adata = anndata.AnnData(X=x[None, keep])
    adata.var_names = [model.features[i] for i in keep]
    path = tmp_path / f"{name.lower()}.pt"
    torch.save(model, path)
    monkeypatch.setattr(_pred_utils, "download_clock_weights", lambda *args, **kwargs: str(path))
    assert np.isnan(model.reference_values[model.features.index(missing_name)])
    with pytest.raises(ValueError, match=missing_name):
        pya.pred.predict_age(adata, name, verbose=False)
