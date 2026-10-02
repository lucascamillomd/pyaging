"""PAOPAC parity against the original compiled Windows prediction interface."""

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest
import torch

from pyaging.models._paopac import PAOPAC

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "paopac"
CASES = json.loads((FIXTURES / "oracle_provenance.json").read_text())["cases"]
SPEC = importlib.util.spec_from_file_location(
    "build_paopac", Path(__file__).resolve().parents[2] / "clocks" / "build_paopac.py"
)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


@pytest.fixture(scope="module")
def model():
    text = gzip.decompress((FIXTURES / "author_conventional.txt.gz").read_bytes())
    assert hashlib.sha256(text).hexdigest() == BUILDER.MODEL_TEXT_SHA256
    features, trees = BUILDER.convert_trees(text.decode())
    result = PAOPAC()
    result.features = [*features, "age"]
    result.base_model_features = list(result.features)
    result.feature_units = ["NPX"] * 189 + ["Townsend deprivation index", "years"]
    result.reference_values = [np.nan] * 191
    result.base_model = trees
    result.metadata = BUILDER.clock_metadata()
    return result.eval()


@pytest.fixture(scope="module")
def oracle():
    with np.load(FIXTURES / "native_oracle.npz", allow_pickle=False) as archive:
        return dict(archive)


def input_frame(oracle, case):
    result = pd.DataFrame(oracle[case + "_input"], columns=oracle[case + "_columns"], index=oracle[case + "_index"])
    result["age"] = oracle[case + "_ages"]
    return result


@pytest.mark.parametrize("case", CASES)
def test_native_binary_normalization_raw_and_corrected_parity(model, oracle, case):
    adata = anndata.AnnData(input_frame(oracle, case))
    aligned = model.prepare_input_frame(adata).reindex(columns=model.features)
    matrix = torch.tensor(aligned.to_numpy(), dtype=torch.float64)
    with torch.inference_mode():
        context = model.prepare_cohort_context(matrix)
        normalized = (model._linear_abundance(matrix) - context[0].numpy()) / context[1].numpy()
        raw = model.predict_with_cohort_context(matrix, context)
        corrected = model.postprocess_cohort(raw, matrix)
    np.testing.assert_allclose(normalized, oracle[case + "_scaled"], rtol=0, atol=2e-14)
    np.testing.assert_allclose(raw.numpy().ravel(), oracle[case + "_raw"], rtol=0, atol=2e-12)
    np.testing.assert_allclose(
        corrected.numpy().ravel(), oracle[case + "_corrected"], rtol=0, atol=2e-10, equal_nan=True
    )


@pytest.mark.parametrize("batch_size", [1, 7, 1000])
@pytest.mark.parametrize("case", ["complete", "missing", "tdi", "age_nan"])
def test_public_pipeline_batches_cohort_and_preserves_input(model, oracle, case, batch_size, monkeypatch):
    import pyaging as pya

    monkeypatch.setattr("pyaging.predict._pred.load_clock", lambda *args, **kwargs: model)
    frame = input_frame(oracle, case).iloc[::-1, ::-1].copy()
    # Original interface accepts lowercase proteins and uppercase Age.
    frame.columns = ["Age" if name == "age" else name.lower() for name in frame.columns]
    adata = anndata.AnnData(frame)
    adata.obs["TDI"] = 20.0  # Metadata must not substitute the protein-matrix TDI.
    before = adata.to_df().copy()
    pya.pred.predict_age(adata, "PAOPAC", batch_size=batch_size, verbose=False)
    np.testing.assert_allclose(
        adata.obs["paopac"], oracle[case + "_corrected"][::-1], rtol=0, atol=2e-10, equal_nan=True
    )
    pd.testing.assert_frame_equal(adata.to_df(), before)
    expected_missing = set(model.features).difference(model.prepare_input_frame(adata).columns)
    assert set(adata.uns["paopac_missing_features"]) == expected_missing


def test_case_collision_matches_author_rejection(model, oracle):
    frame = input_frame(oracle, "complete")
    frame.columns = frame.columns.str.lower()
    frame["ABO"] = 5.0
    adata = anndata.AnnData(frame)
    with pytest.raises(ValueError, match="collide after uppercasing"):
        model.prepare_input_frame(adata)


def test_age_is_required_before_alignment(model, oracle, monkeypatch):
    import pyaging as pya

    monkeypatch.setattr("pyaging.predict._pred.load_clock", lambda *args, **kwargs: model)
    adata = anndata.AnnData(input_frame(oracle, "complete").drop(columns="age"))
    with pytest.raises(ValueError, match="PAOPAC requires chronological age"):
        pya.pred.predict_age(adata, "paopac", verbose=False)


def test_cached_model_does_not_retain_cohort_state(model, oracle, monkeypatch):
    import pyaging as pya

    monkeypatch.setattr("pyaging.predict._pred.load_clock", lambda *args, **kwargs: model)
    cache = pya.pred.ClockCache()
    states = {name: value.clone() for name, value in model.state_dict().items()}
    for case in ["complete", "tdi", "missing", "complete"]:
        adata = anndata.AnnData(input_frame(oracle, case))
        pya.pred.predict_age(adata, "paopac", batch_size=5, clock_cache=cache, verbose=False)
        np.testing.assert_allclose(adata.obs["paopac"], oracle[case + "_corrected"], rtol=0, atol=2e-10)
    for name, value in model.state_dict().items():
        assert torch.equal(value, states[name])


def test_original_small_cohort_behavior_is_visible(oracle):
    # The author's cohort correction reduces a singleton exactly to its age.
    np.testing.assert_allclose(oracle["singleton_corrected"], oracle["singleton_ages"], rtol=0, atol=1e-12)
    assert np.isnan(oracle["age_nan_corrected"][3])


def test_extreme_infinite_abundance_is_rejected(model, oracle):
    frame = input_frame(oracle, "complete").reindex(columns=model.features)
    frame.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinity"):
        model.prepare_cohort_context(torch.tensor(frame.to_numpy(), dtype=torch.float64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_trees_match_native_binary(model, oracle):
    model = model.to("cuda")
    try:
        matrix = torch.tensor(
            input_frame(oracle, "complete").reindex(columns=model.features).to_numpy(),
            dtype=torch.float64,
            device="cuda",
        )
        with torch.inference_mode():
            actual = model(matrix).cpu().numpy().ravel()
        np.testing.assert_allclose(actual, oracle["complete_corrected"], rtol=0, atol=2e-10)
    finally:
        model.cpu()
