"""HPS parity with pinned Kuo et al. R code, including the public input path."""

import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "hps"


def _fixture_model():
    assert importlib.util.find_spec("pyaging.models._proteoclock") is not None, "HPS implementation is missing"
    module = importlib.import_module("pyaging.models._proteoclock")
    model = module.HPS()
    parameters = json.loads((FIXTURES / "hps_author_parameters.json").read_text())
    model.features = parameters["features"]
    model.base_model_features = list(model.features)
    model.feature_units = ["years"] + ["NPX"] * 86
    model.base_model = torch.nn.Linear(87, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.base_model.weight.copy_(torch.tensor([parameters["coefficients"]], dtype=torch.float64))
        model.gompertz_parameters.copy_(torch.tensor([parameters[k] for k in ("shape", "rate")], dtype=torch.float64))
    model.metadata.update(clock_name="hps", data_type="proteomics", research_only=True)
    return model.eval()


@pytest.fixture(scope="module")
def npx():
    return pd.read_csv(FIXTURES / "input_npx.csv", index_col=0)


@pytest.fixture(scope="module")
def oracle():
    return pd.read_csv(FIXTURES / "expected_predictions.csv")


@pytest.mark.parametrize("case", ["complete", "present_na", "reversed_features"])
@pytest.mark.parametrize("source", ["author_fixture", "local_artifact"])
def test_hps_matches_original_r(npx, oracle, case, source):
    from tests.helpers import load_local_clock

    model = _fixture_model() if source == "author_fixture" else load_local_clock("hps")
    frame = npx.copy()
    if case == "present_na":
        frame.iloc[0, frame.columns.get_loc("ace2")] = np.nan
    if case == "reversed_features":
        frame = frame.iloc[:, ::-1]
    expected = oracle.loc[oracle.case.eq(case), "prediction"]
    with torch.inference_mode():
        actual = model(torch.tensor(frame[model.features].to_numpy(), dtype=torch.float64)).numpy().ravel()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10, equal_nan=True)


@pytest.mark.parametrize("batch_size", [1, 3, 1000])
def test_predict_age_preserves_samples_and_feature_order(npx, oracle, tmp_path, monkeypatch, batch_size):
    import anndata

    import pyaging as pya

    model = _fixture_model()
    artifact = tmp_path / "hps.pt"
    torch.save(model, artifact)
    monkeypatch.setattr("pyaging.predict._pred_utils.download_clock_weights", lambda *args, **kwargs: artifact)
    frame = npx.iloc[::-1, ::-1].copy()
    frame["irrelevant_protein"] = 17.0
    adata = anndata.AnnData(frame)
    pya.pred.predict_age(adata, "HPS", batch_size=batch_size, verbose=False)
    expected = oracle.loc[oracle.case.eq("complete")].set_index("sample_id").loc[frame.index, "prediction"]
    np.testing.assert_allclose(adata.obs["hps"], expected, rtol=0, atol=1e-10)
    assert adata.uns["hps_missing_features"] == []


@pytest.mark.parametrize("missing", ["age", "ace2", "nppb"])
def test_predict_age_rejects_missing_predictors(npx, tmp_path, monkeypatch, missing):
    import anndata

    import pyaging as pya

    artifact = tmp_path / "hps.pt"
    torch.save(_fixture_model(), artifact)
    monkeypatch.setattr("pyaging.predict._pred_utils.download_clock_weights", lambda *args, **kwargs: artifact)
    adata = anndata.AnnData(npx.drop(columns=missing))
    with pytest.raises(ValueError, match=f"HPS requires.*{missing}"):
        pya.pred.predict_age(adata, "hps", verbose=False)
    assert "hps" not in adata.obs


def test_predict_age_propagates_present_nan(npx, oracle, tmp_path, monkeypatch):
    import anndata

    import pyaging as pya

    artifact = tmp_path / "hps.pt"
    torch.save(_fixture_model(), artifact)
    monkeypatch.setattr("pyaging.predict._pred_utils.download_clock_weights", lambda *args, **kwargs: artifact)
    frame = npx.copy()
    frame.iloc[0, frame.columns.get_loc("ace2")] = np.nan
    adata = anndata.AnnData(frame)
    pya.pred.predict_age(adata, "hps", verbose=False)
    expected = oracle.loc[oracle.case.eq("present_na"), "prediction"]
    np.testing.assert_allclose(adata.obs["hps"], expected, rtol=0, atol=1e-10, equal_nan=True)


def test_predict_age_rejects_duplicate_assays(npx, tmp_path, monkeypatch):
    import anndata

    import pyaging as pya

    artifact = tmp_path / "hps.pt"
    torch.save(_fixture_model(), artifact)
    monkeypatch.setattr("pyaging.predict._pred_utils.download_clock_weights", lambda *args, **kwargs: artifact)
    frame = pd.concat([npx, npx[["ace2"]]], axis=1)
    with pytest.warns(UserWarning, match="Variable names are not unique"):
        adata = anndata.AnnData(frame)
    with pytest.raises(ValueError, match="HPS requires unique feature names"):
        pya.pred.predict_age(adata, "hps", verbose=False)


def test_feature_ranges_distinguish_age_from_npx(tmp_path, monkeypatch):
    import pyaging as pya

    artifact = tmp_path / "hps.pt"
    torch.save(_fixture_model(), artifact)
    monkeypatch.setattr("pyaging.predict._pred_utils.download_clock_weights", lambda *args, **kwargs: artifact)
    ranges = pya.utils.get_feature_ranges("HPS").set_index("feature")
    assert ranges.loc["age", "unit"] == "years"
    proteins = ranges.drop(index="age")
    assert proteins["unit"].eq("NPX").all()
    assert np.isneginf(proteins["low"]).all()
    assert np.isposinf(proteins["high"]).all()


def test_predict_age_respects_deliberate_upstream_imputation(npx, tmp_path, monkeypatch):
    import pyaging as pya

    model = _fixture_model()
    artifact = tmp_path / "hps.pt"
    torch.save(model, artifact)
    monkeypatch.setattr("pyaging.predict._pred_utils.download_clock_weights", lambda *args, **kwargs: artifact)
    frame = npx.copy()
    frame.iloc[0, frame.columns.get_loc("ace2")] = np.nan
    adata = pya.pp.df_to_adata(frame, imputer_strategy="mean", verbose=False)
    assert adata.var.loc["ace2", "percent_na"] > 0
    supplied = adata.to_df()[model.features].to_numpy()
    with torch.inference_mode():
        expected = model(torch.tensor(supplied, dtype=torch.float64)).numpy().ravel()
    pya.pred.predict_age(adata, "hps", verbose=False)
    np.testing.assert_allclose(adata.obs["hps"], expected, rtol=0, atol=1e-10)


def test_hps_uses_original_precision_and_protein_names():
    parameters = json.loads((FIXTURES / "hps_author_parameters.json").read_text())
    model = _fixture_model()
    assert len(model.features) == 87
    assert model.features[0] == "age"
    assert "nppb" in model.features
    assert "ntprobnp" in model.features
    assert parameters["rate"] == 0.002104934
    assert parameters["shape"] == 0.064438847
    assert model.reference_values is None


def _builder():
    path = Path(__file__).resolve().parents[2] / "clocks" / "build_hps.py"
    spec = importlib.util.spec_from_file_location("build_hps", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_rejects_tampered_original_source(tmp_path):
    builder = _builder()
    (tmp_path / "HPS.R").write_text("rate=0.000232\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        builder.build_clock(source_dir=tmp_path)


def test_built_artifact_preserves_original_parameters_and_provenance():
    from tests.helpers import load_local_clock

    model = load_local_clock("hps")
    expected = _fixture_model()
    torch.testing.assert_close(model.base_model.weight, expected.base_model.weight, rtol=0, atol=0)
    torch.testing.assert_close(model.gompertz_parameters, expected.gompertz_parameters, rtol=0, atol=0)
    assert model.features == expected.features
    assert model.feature_units == expected.feature_units
    assert model.version == "0.5.6"
    assert model.provenance["commit"] == "6eec9ac5a091e34fece69fa4652ebd9af22c1839"
    assert model.provenance["coefficient_sha256"] == _builder().SOURCE_SHA256["HPS.R"]
    assert model.reference_values is None
    assert model.metadata["n_features"] == 87
    assert "research use only" in model.license
    assert model.provenance["publication_license"] == "CC BY-NC-ND 4.0"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device is unavailable")
def test_hps_cuda_matches_original_r(npx, oracle):
    model = _fixture_model().cuda()
    with torch.inference_mode():
        actual = model(torch.tensor(npx[model.features].to_numpy(), dtype=torch.float64, device="cuda"))
    expected = oracle.loc[oracle.case.eq("complete"), "prediction"]
    np.testing.assert_allclose(actual.cpu().numpy().ravel(), expected, rtol=0, atol=1e-9)
