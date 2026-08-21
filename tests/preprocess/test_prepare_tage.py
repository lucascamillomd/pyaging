"""Tests for the public ``prepare_tage`` entry point and its gene filter.

The numeric parity of each individual stage is covered by
``test_tage_transforms.py`` against the authors' R fixtures; what is checked
here is that ``prepare_tage`` composes those stages in the reference order
(filter -> map -> RLE -> log -> scale -> center), resolves the reference group,
and stamps the provenance the predict pipeline reads back.

The mapping asset is monkeypatched to a four-gene table, so nothing here
touches the network.
"""

import io
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest
from rich.console import Console

import pyaging as pya
import pyaging.logger._live as live_module
from pyaging.preprocess import _tage

FIXTURES = Path(__file__).resolve().parents[2] / "tests/data/tage"

MAPPING = pd.DataFrame(
    {
        "species": ["mouse"] * 4,
        "source_id": ["G1", "G2", "G3", "G4"],
        "source_type": ["symbol"] * 4,
        "mouse_entrez": ["101", "102", "103", "104"],
    }
)


@pytest.fixture(autouse=True)
def _local_mapping(monkeypatch):
    monkeypatch.setattr(_tage, "_load_mapping", lambda dir: MAPPING)


@pytest.fixture
def display_output(monkeypatch):
    """Point the live display at a buffer this test can read back.

    ``tests/conftest.py`` already swaps the module console for a silent one; the
    warning tests need the same treatment but with the buffer in hand. The width
    is set wide so a wrapped line cannot hide the asserted text.
    """
    buffer = io.StringIO()
    monkeypatch.setattr(
        live_module, "_console", Console(file=buffer, force_terminal=False, force_jupyter=False, width=200)
    )
    return buffer


def _adata(n_obs=4):
    rng = np.random.default_rng(0)
    X = rng.integers(10, 1000, size=(n_obs, 4)).astype(float)
    a = anndata.AnnData(X=X)
    a.var_names = ["G1", "G2", "G3", "G4"]
    a.obs_names = [f"s{i}" for i in range(n_obs)]
    return a


def test_prepare_stamps_uns_and_maps_names():
    out = pya.pp.prepare_tage(_adata(), species="mouse", verbose=False)
    assert out.uns["tage_prepared"] is True
    assert list(out.var_names) == ["101", "102", "103", "104"]
    assert out.uns["tage_preparation"]["reference_group"] == "all_samples"
    assert out.uns["tage_preparation"]["species"] == "mouse"
    assert out.uns["tage_preparation"]["n_input_genes"] == 4
    assert out.uns["tage_preparation"]["n_mapped_genes"] == 4
    assert out.uns["tage_preparation"]["n_reference_samples"] == 4


def test_prepare_pipeline_matches_composed_helpers():
    a = _adata()
    out = pya.pp.prepare_tage(a, species="mouse", verbose=False)
    frame = pd.DataFrame(a.X, index=a.obs_names, columns=list(a.var_names))
    filtered = _tage._filter_genes(frame)
    mapped = _tage._map_to_mouse_entrez(filtered, "mouse", MAPPING)
    expected = _tage._center_against_reference(_tage._scale_samples(_tage._log_transform(_tage._rle_normalize(mapped))))
    np.testing.assert_allclose(out.X, expected.values, atol=1e-12)


def test_prepare_applies_the_gene_filter_before_mapping():
    # G4 never reaches the count threshold, so filter_genes drops it and it must
    # not reach the mapper (and so not appear among the output columns).
    a = _adata()
    a.X[:, 3] = 0.0
    out = pya.pp.prepare_tage(a, species="mouse", verbose=False)
    assert list(out.var_names) == ["101", "102", "103"]
    assert out.uns["tage_preparation"]["n_input_genes"] == 4
    assert out.uns["tage_preparation"]["n_mapped_genes"] == 3


def test_obs_and_names_are_carried_through():
    a = _adata()
    a.obs["group"] = ["a", "a", "b", "b"]
    out = pya.pp.prepare_tage(a, species="mouse", verbose=False)
    assert list(out.obs_names) == list(a.obs_names)
    assert list(out.obs["group"]) == ["a", "a", "b", "b"]
    assert out.X.dtype == np.float64


def test_reference_group_by_name_and_mask_agree():
    a = _adata()
    by_name = pya.pp.prepare_tage(a, species="mouse", reference_group=["s0", "s1"], verbose=False)
    mask = np.array([True, True, False, False])
    by_mask = pya.pp.prepare_tage(a, species="mouse", reference_group=mask, verbose=False)
    np.testing.assert_allclose(by_name.X, by_mask.X)
    assert by_name.uns["tage_preparation"]["n_reference_samples"] == 2
    assert by_name.uns["tage_preparation"]["reference_group"] == ["s0", "s1"]


def test_reference_group_centres_on_that_group_only():
    a = _adata()
    out = pya.pp.prepare_tage(a, species="mouse", reference_group=["s0", "s1"], verbose=False)
    # The median of the two reference samples is their mean, so the two rows are
    # equal and opposite after centring.
    np.testing.assert_allclose(out.X[0], -out.X[1], atol=1e-12)


def test_single_sample_raises():
    with pytest.raises(ValueError, match="at least two samples"):
        pya.pp.prepare_tage(_adata(n_obs=1), species="mouse", verbose=False)


def test_empty_reference_group_raises():
    with pytest.raises(ValueError, match="reference_group"):
        pya.pp.prepare_tage(_adata(), species="mouse", reference_group=[], verbose=False)


def test_unknown_reference_group_name_raises():
    with pytest.raises(ValueError, match="reference_group"):
        pya.pp.prepare_tage(_adata(), species="mouse", reference_group=["nope"], verbose=False)


def test_wrong_length_boolean_mask_raises():
    with pytest.raises(ValueError, match="one entry per sample"):
        pya.pp.prepare_tage(_adata(), species="mouse", reference_group=np.array([True, False]), verbose=False)


def test_unknown_species_raises():
    with pytest.raises(ValueError, match="species"):
        pya.pp.prepare_tage(_adata(), species="ferret", verbose=False)


def test_no_mappable_gene_raises():
    a = _adata()
    a.var_names = ["X1", "X2", "X3", "X4"]
    with pytest.raises(ValueError, match="mapped"):
        pya.pp.prepare_tage(a, species="mouse", verbose=False)


def test_everything_filtered_out_raises():
    a = _adata()
    a.X[:] = 1.0
    with pytest.raises(ValueError, match="raw RNA-seq counts"):
        pya.pp.prepare_tage(a, species="mouse", verbose=False)


def test_low_overlap_warns(display_output):
    # One of six retained genes maps, well under the 50% threshold.
    rng = np.random.default_rng(1)
    a = anndata.AnnData(X=rng.integers(10, 1000, size=(4, 6)).astype(float))
    a.var_names = ["G1", "U1", "U2", "U3", "U4", "U5"]
    a.obs_names = [f"s{i}" for i in range(4)]
    pya.pp.prepare_tage(a, species="mouse", verbose=True)
    assert "1 of 6 expressed genes mapped" in display_output.getvalue()


def test_good_overlap_does_not_warn(display_output):
    pya.pp.prepare_tage(_adata(), species="mouse", verbose=True)
    assert "mouse Entrez" not in display_output.getvalue()


def test_public_export():
    assert "prepare_tage" in pya.pp.__all__


# --- gene filter semantics -------------------------------------------------
#
# ``filter_genes`` (tAge R/preprocessing.R:68-74) keeps a gene when
# ``sum(x >= count_threshold, na.rm = TRUE) >= ncol * (percent_threshold/100)``.
# Both comparisons are inclusive and NAs count as failures.


def _counts(rows):
    frame = pd.DataFrame(rows, dtype=float)
    frame.columns = [f"g{i}" for i in range(frame.shape[1])]
    return frame


def test_filter_threshold_is_inclusive_on_the_count():
    # Ten samples, one gene at exactly 10 in 2 samples (20%): kept.
    frame = _counts([[10.0]] * 2 + [[0.0]] * 8)
    assert list(_tage._filter_genes(frame).columns) == ["g0"]


def test_filter_drops_a_gene_one_below_the_count_threshold():
    frame = _counts([[9.0]] * 2 + [[0.0]] * 8)
    assert list(_tage._filter_genes(frame).columns) == []


def test_filter_percentage_is_inclusive_on_the_sample_count():
    # 20% of 10 samples is exactly 2, so 2 passing samples is enough and 1 is not.
    enough = _counts([[10.0, 10.0]] * 2 + [[0.0, 0.0]] * 8)
    assert list(_tage._filter_genes(enough).columns) == ["g0", "g1"]
    short = _counts([[10.0]] + [[0.0]] * 9)
    assert list(_tage._filter_genes(short).columns) == []


def test_filter_counts_nan_as_failing():
    # R's sum(..., na.rm = TRUE) ignores NA, so a NaN sample never counts toward
    # the quota; here only one of ten samples passes, one short of the 2 needed.
    frame = _counts([[10.0], [np.nan]] + [[0.0]] * 8)
    assert list(_tage._filter_genes(frame).columns) == []


def test_filter_thresholds_are_configurable():
    frame = _counts([[5.0]] * 10)
    assert list(_tage._filter_genes(frame, count_threshold=5).columns) == ["g0"]
    assert list(_tage._filter_genes(frame, count_threshold=6).columns) == []


def test_filter_matches_the_reference_fixture_gene_count():
    # Reproduces the filter_genes call in clocks/generate_tage_fixtures.R against
    # the committed raw counts; the R run retained 19 550 of 57 010 genes.
    counts = pd.read_csv(FIXTURES / "input_expression.csv.gz", index_col=0).T
    assert _tage._filter_genes(counts).shape == (24, 19550)
