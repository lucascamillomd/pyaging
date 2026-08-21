"""End-to-end parity for the tAge clocks against the authors' own pipeline.

``tests/data/tage`` holds predictions produced by the reference R preprocessing
and the published sklearn models (see its README); nothing there came from a
pyaging code path, so a pass here means the whole chain -- ``prepare_tage``,
feature alignment, the imputer substitution, and the ported weights -- lands on
the reference numbers rather than merely agreeing with itself.

Two assets that production fetches from the Hub are read locally instead: the
gene mapping (``_tage._load_mapping``) and the clock weights
(``download_clock_weights``, patched where ``load_clock`` looks it up). The
weights are build artifacts of ``clocks/*.ipynb`` and are gitignored, so the
whole module skips when they are absent and CI without them stays green.
"""

import gzip
import json
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest

import pyaging as pya
from pyaging.predict import _pred_utils
from pyaging.preprocess import _tage

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "tests/data/tage"
WEIGHTS = REPO / "clocks/weights"
CLOCKS = ("tage", "tagemortality")
MAPPING_ASSET = WEIGHTS / "tage_gene_mapping.csv.gz"

# The fixtures are the authors' float64 predictions, so parity is judged in
# absolute terms only: rtol must be 0, or numpy's 1e-7 relative default would
# quietly dominate the comparison for the larger predictions.
TOL = 1e-6

pytestmark = pytest.mark.skipif(
    not all((WEIGHTS / f"{name}.pt").exists() for name in CLOCKS) or not MAPPING_ASSET.exists(),
    reason="local tAge weights or gene mapping not built; run the clocks/ notebooks first",
)


@pytest.fixture(scope="module")
def expected():
    return json.loads((DATA / "expected_predictions.json").read_text())


@pytest.fixture(scope="module")
def counts():
    """The example cohort as samples x genes, from the genes x samples fixture.

    Every stage CSV is written in R orientation (README, "Matrix orientation"),
    so the transpose is what makes ``var_names`` the mouse Ensembl gene IDs
    ``prepare_tage`` expects.
    """
    with gzip.open(DATA / "input_expression.csv.gz", "rt") as handle:
        frame = pd.read_csv(handle, index_col=0)
    return frame.T


@pytest.fixture(scope="module")
def mapping():
    return pd.read_csv(MAPPING_ASSET, dtype=str)


@pytest.fixture
def adata(counts):
    out = anndata.AnnData(X=counts.to_numpy(dtype=np.float64))
    out.obs_names = counts.index.astype(str)
    out.var_names = counts.columns.astype(str)
    return out


@pytest.fixture(autouse=True)
def local_assets(monkeypatch, mapping):
    """Serve the mapping and the clock weights from the working tree."""
    monkeypatch.setattr(_tage, "_load_mapping", lambda dir: mapping)

    def local_weights(clock_name, dir="pyaging_data", logger=None, indent_level=1):
        return str(WEIGHTS / f"{clock_name}.pt")

    monkeypatch.setattr(_pred_utils, "download_clock_weights", local_weights)


def test_input_fixture_is_mouse_ensembl_samples_by_genes(adata, expected):
    assert list(adata.obs_names) == expected["sample_ids"]
    assert adata.shape == (24, 57010)
    assert all(name.startswith("ENSMUSG") for name in adata.var_names)


@pytest.mark.parametrize("clock_name", CLOCKS)
def test_cohort_centered_predictions_match_reference(adata, expected, clock_name):
    prepared = pya.pp.prepare_tage(adata, species="mouse", verbose=False)
    pya.pred.predict_age(prepared, [clock_name], verbose=False)

    assert list(prepared.obs_names) == expected["sample_ids"]
    np.testing.assert_allclose(
        prepared.obs[clock_name].to_numpy(),
        np.asarray(expected[f"{clock_name}_center_all"]),
        rtol=0,
        atol=TOL,
    )


@pytest.mark.parametrize("clock_name", CLOCKS)
def test_reference_group_centered_predictions_match_reference(adata, expected, clock_name):
    prepared = pya.pp.prepare_tage(
        adata,
        species="mouse",
        reference_group=expected["reference_group_sample_ids"],
        verbose=False,
    )
    pya.pred.predict_age(prepared, [clock_name], verbose=False)

    np.testing.assert_allclose(
        prepared.obs[clock_name].to_numpy(),
        np.asarray(expected[f"{clock_name}_center_refgroup"]),
        rtol=0,
        atol=TOL,
    )


def test_prediction_writes_the_standard_clock_metadata(adata):
    prepared = pya.pp.prepare_tage(adata, species="mouse", verbose=False)
    pya.pred.predict_age(prepared, list(CLOCKS), verbose=False)

    for clock_name in CLOCKS:
        assert clock_name in prepared.obs.columns
        metadata = prepared.uns[f"{clock_name}_metadata"]
        assert metadata["clock_name"] == clock_name
        assert metadata["data_type"] == "transcriptomics (relative)"
        assert prepared.uns[f"{clock_name}_missing_features"] is not None
        assert 0 <= prepared.uns[f"{clock_name}_percent_na"] < 100


def test_batch_size_does_not_change_predictions(adata):
    prepared = pya.pp.prepare_tage(adata, species="mouse", verbose=False)
    batched = prepared.copy()
    pya.pred.predict_age(prepared, ["tage"], verbose=False)
    pya.pred.predict_age(batched, ["tage"], batch_size=2, verbose=False)

    np.testing.assert_allclose(
        prepared.obs["tage"].to_numpy(),
        batched.obs["tage"].to_numpy(),
        rtol=0,
        atol=1e-12,
    )


def test_predict_without_prepare_raises(adata):
    with pytest.raises(ValueError, match="prepare_tage"):
        pya.pred.predict_age(adata, ["tage"], verbose=False)
