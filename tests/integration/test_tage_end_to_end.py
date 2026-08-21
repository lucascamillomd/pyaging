"""End-to-end parity for the tAge clocks against the authors' own pipeline.

``tests/data/tage`` holds predictions produced by the reference R preprocessing
and the published sklearn models (see its README); nothing there came from a
pyaging code path, so a pass here means the whole chain -- the in-predict cohort
transform, feature alignment, the imputer substitution, and the ported weights
-- lands on the reference numbers rather than merely agreeing with itself.

The entry point is ``predict_age`` on raw counts: the cohort preprocessing is no
longer a separate call the user makes.

Three assets that production fetches from the Hub are read locally instead: the
gene mapping (``_tage._load_mapping``) and the clock weights
(``download_clock_weights``, patched where ``load_clock`` looks it up). The
weights are build artifacts of ``clocks/*.ipynb`` and are gitignored, so the
whole module skips when they are absent and CI without them stays green.

The committed ``.pt`` files predate the in-predict transform: they still carry
``required_uns_flag`` and no ``cohort_transform``. Until the notebooks are
re-executed the seam sets ``cohort_transform`` on the freshly loaded model, so
these tests exercise the new path against the real weights -- and, because the
stale flag is left in place, they also pin that a declared transform supersedes
it.
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
    so the transpose is what makes ``var_names`` the mouse Ensembl gene IDs the
    cohort transform expects.
    """
    with gzip.open(DATA / "input_expression.csv.gz", "rt") as handle:
        frame = pd.read_csv(handle, index_col=0)
    return frame.T


@pytest.fixture(scope="module")
def mapping():
    return pd.read_csv(MAPPING_ASSET, dtype=str)


def _adata_from(frame):
    out = anndata.AnnData(X=frame.to_numpy(dtype=np.float64))
    out.obs_names = frame.index.astype(str)
    out.var_names = frame.columns.astype(str)
    return out


@pytest.fixture
def adata(counts):
    """Raw counts with no species indicator: the default-mouse path."""
    return _adata_from(counts)


@pytest.fixture
def adata_mouse(counts):
    """The same counts with an explicit ``mouse`` indicator column."""
    return _adata_from(counts.assign(mouse=1.0))


@pytest.fixture(autouse=True)
def local_assets(monkeypatch, mapping):
    """Serve the mapping and the clock weights from the working tree."""
    monkeypatch.setattr(_tage, "_load_mapping", lambda dir: mapping)

    def local_weights(clock_name, dir="pyaging_data", logger=None, indent_level=1):
        return str(WEIGHTS / f"{clock_name}.pt")

    monkeypatch.setattr(_pred_utils, "download_clock_weights", local_weights)

    # The committed weights predate the attribute; Task B's notebook run bakes
    # it in and this shim goes away.
    real_load_clock = _pred_utils.load_clock

    def load_with_transform(*args, **kwargs):
        model = real_load_clock(*args, **kwargs)
        assert getattr(model, "cohort_transform", None) is None, "shim obsolete — remove it (Task B landed)"
        model.cohort_transform = "tage"
        return model

    monkeypatch.setattr("pyaging.predict._pred.load_clock", load_with_transform)


def test_input_fixture_is_mouse_ensembl_samples_by_genes(adata, expected):
    assert list(adata.obs_names) == expected["sample_ids"]
    assert adata.shape == (24, 57010)
    assert all(name.startswith("ENSMUSG") for name in adata.var_names)


@pytest.mark.parametrize("clock_name", CLOCKS)
def test_cohort_centered_predictions_match_reference(adata_mouse, expected, clock_name):
    pya.pred.predict_age(adata_mouse, [clock_name], verbose=False)

    assert adata_mouse.uns["tage_preparation"]["species"] == "mouse"
    assert list(adata_mouse.obs_names) == expected["sample_ids"]
    np.testing.assert_allclose(
        adata_mouse.obs[clock_name].to_numpy(),
        np.asarray(expected[f"{clock_name}_center_all"]),
        rtol=0,
        atol=TOL,
    )


@pytest.mark.parametrize("clock_name", CLOCKS)
def test_predictions_are_the_same_without_a_species_column(adata, expected, clock_name):
    # No indicator at all: the transform defaults to mouse, which is what this
    # cohort is, so the numbers must not move.
    pya.pred.predict_age(adata, [clock_name], verbose=False)

    assert adata.uns["tage_preparation"]["species"] == "mouse"
    np.testing.assert_allclose(
        adata.obs[clock_name].to_numpy(),
        np.asarray(expected[f"{clock_name}_center_all"]),
        rtol=0,
        atol=TOL,
    )


@pytest.mark.parametrize("clock_name", CLOCKS)
def test_reference_group_centered_predictions_match_reference(adata_mouse, expected, clock_name):
    adata_mouse.obs["tage_reference_group"] = [
        name in set(expected["reference_group_sample_ids"]) for name in adata_mouse.obs_names
    ]
    pya.pred.predict_age(adata_mouse, [clock_name], verbose=False)

    assert adata_mouse.uns["tage_preparation"]["n_reference_samples"] == len(expected["reference_group_sample_ids"])
    np.testing.assert_allclose(
        adata_mouse.obs[clock_name].to_numpy(),
        np.asarray(expected[f"{clock_name}_center_refgroup"]),
        rtol=0,
        atol=TOL,
    )


def test_prediction_writes_the_standard_clock_metadata(adata_mouse):
    pya.pred.predict_age(adata_mouse, list(CLOCKS), verbose=False)

    for clock_name in CLOCKS:
        assert clock_name in adata_mouse.obs.columns
        metadata = adata_mouse.uns[f"{clock_name}_metadata"]
        assert metadata["clock_name"] == clock_name
        assert metadata["data_type"] == "transcriptomics (relative)"
        assert adata_mouse.uns[f"{clock_name}_missing_features"] is not None
        assert 0 <= adata_mouse.uns[f"{clock_name}_percent_na"] < 100


def test_the_cohort_transform_runs_once_for_both_clocks(adata_mouse, monkeypatch):
    calls = []
    real = _pred_utils.COHORT_TRANSFORMS["tage"]

    def counting(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setitem(_pred_utils.COHORT_TRANSFORMS, "tage", counting)
    pya.pred.predict_age(adata_mouse, list(CLOCKS), verbose=False)

    assert len(calls) == 1


def test_the_raw_counts_are_not_mutated(adata_mouse):
    original = adata_mouse.X.copy()
    pya.pred.predict_age(adata_mouse, list(CLOCKS), verbose=False)

    np.testing.assert_array_equal(adata_mouse.X, original)
    assert list(adata_mouse.var_names)[-1] == "mouse"


def test_batch_size_does_not_change_predictions(adata_mouse):
    batched = adata_mouse.copy()
    pya.pred.predict_age(adata_mouse, ["tage"], verbose=False)
    pya.pred.predict_age(batched, ["tage"], batch_size=2, verbose=False)

    np.testing.assert_allclose(
        adata_mouse.obs["tage"].to_numpy(),
        batched.obs["tage"].to_numpy(),
        rtol=0,
        atol=1e-12,
    )
