"""Tests for the internal tAge cohort transform and its gene filter.

The numeric parity of each individual stage is covered by
``test_tage_transforms.py`` against the authors' R fixtures; what is checked
here is that ``_prepare_tage`` composes those stages in the reference order
(filter -> map -> RLE -> log -> scale -> center), reads the species indicator
and reference-group columns off the input, and records the provenance the
predict pipeline reports back.

``_prepare_tage`` is no longer a public entry point: ``predict_age`` calls it
for the cohort-relative clocks. It returns the transformed samples x
mouse-Entrez frame and never touches the caller's matrix.

The mapping asset is monkeypatched to a four-gene table, so nothing here
touches the network.
"""

from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest

import pyaging as pya
from pyaging.logger._live import DisplayLogger
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
def warnings():
    """Collect what the transform logs, the way the predict display would."""
    messages = []
    return messages


def _logger(messages):
    return DisplayLogger(messages.append)


def _adata(n_obs=4):
    rng = np.random.default_rng(0)
    X = rng.integers(10, 1000, size=(n_obs, 4)).astype(float)
    a = anndata.AnnData(X=X)
    a.var_names = ["G1", "G2", "G3", "G4"]
    a.obs_names = [f"s{i}" for i in range(n_obs)]
    return a


def _with_species(adata, species, values=None):
    """Append a 0/1 species indicator column to a samples x genes AnnData."""
    if values is None:
        values = np.ones(adata.n_obs)
    frame = pd.DataFrame(adata.X, index=adata.obs_names, columns=list(adata.var_names))
    frame[species] = np.asarray(values, dtype=float)
    out = anndata.AnnData(X=frame.to_numpy(dtype=np.float64), obs=adata.obs.copy())
    out.obs_names = adata.obs_names
    out.var_names = frame.columns
    return out


def _prepare(adata, messages=None, **kwargs):
    return _tage._prepare_tage(adata, logger=_logger(messages if messages is not None else []), **kwargs)


# --- composition and provenance --------------------------------------------


def test_prepare_returns_mapped_columns_and_records_provenance():
    a = _adata()
    out = _prepare(a)
    assert list(out.columns) == ["101", "102", "103", "104"]
    provenance = a.uns["tage_preparation"]
    assert provenance["reference_group"] == "all_samples"
    assert provenance["species"] == "mouse"
    assert provenance["n_input_genes"] == 4
    assert provenance["n_mapped_genes"] == 4
    assert provenance["n_reference_samples"] == 4


def test_prepare_pipeline_matches_composed_helpers():
    a = _adata()
    out = _prepare(a)
    frame = pd.DataFrame(a.X, index=a.obs_names, columns=list(a.var_names))
    filtered = _tage._filter_genes(frame)
    mapped = _tage._map_to_mouse_entrez(filtered, "mouse", MAPPING)
    expected = _tage._center_against_reference(_tage._scale_samples(_tage._log_transform(_tage._rle_normalize(mapped))))
    np.testing.assert_allclose(out.to_numpy(), expected.to_numpy(), atol=1e-12)


def test_prepare_applies_the_gene_filter_before_mapping():
    # G4 never reaches the count threshold, so filter_genes drops it and it must
    # not reach the mapper (and so not appear among the output columns).
    a = _adata()
    a.X[:, 3] = 0.0
    out = _prepare(a)
    assert list(out.columns) == ["101", "102", "103"]
    assert a.uns["tage_preparation"]["n_input_genes"] == 4
    assert a.uns["tage_preparation"]["n_mapped_genes"] == 3


def test_sample_names_are_carried_through_and_the_input_is_untouched():
    a = _adata()
    original = a.X.copy()
    out = _prepare(a)
    assert list(out.index) == list(a.obs_names)
    assert out.to_numpy().dtype == np.float64
    np.testing.assert_array_equal(a.X, original)
    assert list(a.var_names) == ["G1", "G2", "G3", "G4"]


# --- species indicator columns ---------------------------------------------


@pytest.mark.parametrize("species", ["mouse", "rat", "macaque", "human"])
def test_species_column_selects_the_mapping_species(species, monkeypatch):
    monkeypatch.setattr(_tage, "_load_mapping", lambda dir: MAPPING.assign(species=species))
    a = _with_species(_adata(), species)
    out = _prepare(a)
    assert a.uns["tage_preparation"]["species"] == species
    # The indicator is not a gene: it never reaches the mapped columns.
    assert list(out.columns) == ["101", "102", "103", "104"]


def test_species_column_is_case_insensitive(monkeypatch):
    monkeypatch.setattr(_tage, "_load_mapping", lambda dir: MAPPING.assign(species="human"))
    a = _with_species(_adata(), "Human")
    _prepare(a)
    assert a.uns["tage_preparation"]["species"] == "human"


def test_missing_species_column_defaults_to_mouse_with_a_warning(warnings):
    a = _adata()
    _prepare(a, messages=warnings)
    assert a.uns["tage_preparation"]["species"] == "mouse"
    assert any("defaulting to mouse" in message for message in warnings)


def test_all_zero_species_columns_default_to_mouse_with_a_warning(warnings):
    a = _with_species(_adata(), "human", values=np.zeros(4))
    _prepare(a, messages=warnings)
    assert a.uns["tage_preparation"]["species"] == "mouse"
    assert any("defaulting to mouse" in message for message in warnings)


def test_species_column_set_to_one_does_not_warn(warnings):
    a = _with_species(_adata(), "mouse")
    _prepare(a, messages=warnings)
    assert not any("defaulting to mouse" in message for message in warnings)


def test_two_species_columns_set_raises():
    a = _with_species(_with_species(_adata(), "mouse"), "human")
    with pytest.raises(ValueError, match="more than one species"):
        _prepare(a)


def test_species_column_that_varies_across_samples_raises():
    a = _with_species(_adata(), "mouse", values=[1.0, 1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="same value for every sample"):
        _prepare(a)


def test_species_column_with_a_value_other_than_zero_or_one_raises():
    a = _with_species(_adata(), "mouse", values=np.full(4, 2.0))
    with pytest.raises(ValueError, match="0 or 1"):
        _prepare(a)


def test_species_indicator_never_reaches_the_gene_filter(monkeypatch):
    # A column of ones would fail the count filter anyway; assert the stronger
    # property that the filter is not even asked about it.
    seen = {}

    original = _tage._filter_genes

    def spy(df, *args, **kwargs):
        seen["columns"] = list(df.columns)
        return original(df, *args, **kwargs)

    monkeypatch.setattr(_tage, "_filter_genes", spy)
    _prepare(_with_species(_adata(), "mouse", values=np.full(4, 1.0)))
    assert seen["columns"] == ["G1", "G2", "G3", "G4"]


# --- reference group --------------------------------------------------------


def test_reference_group_obs_column_centres_on_that_group_only():
    a = _adata()
    a.obs["tage_reference_group"] = [True, True, False, False]
    out = _prepare(a)
    # The median of the two reference samples is their mean, so the two rows are
    # equal and opposite after centring.
    np.testing.assert_allclose(out.to_numpy()[0], -out.to_numpy()[1], atol=1e-12)
    assert a.uns["tage_preparation"]["n_reference_samples"] == 2
    assert a.uns["tage_preparation"]["reference_group"] == ["s0", "s1"]


def test_numeric_reference_group_column_is_accepted():
    a = _adata()
    a.obs["tage_reference_group"] = [1, 1, 0, 0]
    boolean = _adata()
    boolean.obs["tage_reference_group"] = [True, True, False, False]
    np.testing.assert_allclose(_prepare(a).to_numpy(), _prepare(boolean).to_numpy(), atol=1e-12)


def test_absent_reference_group_column_centres_on_every_sample():
    a = _adata()
    everything = _adata()
    everything.obs["tage_reference_group"] = [True] * 4
    np.testing.assert_allclose(_prepare(a).to_numpy(), _prepare(everything).to_numpy(), atol=1e-12)


def test_empty_reference_group_column_raises():
    a = _adata()
    a.obs["tage_reference_group"] = [False] * 4
    with pytest.raises(ValueError, match="selects no samples"):
        _prepare(a)


def test_non_numeric_reference_group_column_raises():
    a = _adata()
    a.obs["tage_reference_group"] = ["yes", "no", "yes", "no"]
    with pytest.raises(ValueError, match="boolean"):
        _prepare(a)


# --- errors and warnings ----------------------------------------------------


def test_single_sample_raises():
    with pytest.raises(ValueError, match="at least two samples"):
        _prepare(_adata(n_obs=1))


def test_no_mappable_gene_raises():
    a = _adata()
    a.var_names = ["X1", "X2", "X3", "X4"]
    with pytest.raises(ValueError, match="mapped"):
        _prepare(a)


def test_everything_filtered_out_raises():
    a = _adata()
    a.X[:] = 1.0
    with pytest.raises(ValueError, match="raw RNA-seq counts"):
        _prepare(a)


def test_low_overlap_warns(warnings):
    # One of six retained genes maps, well under the 50% threshold.
    rng = np.random.default_rng(1)
    a = anndata.AnnData(X=rng.integers(10, 1000, size=(4, 6)).astype(float))
    a.var_names = ["G1", "U1", "U2", "U3", "U4", "U5"]
    a.obs_names = [f"s{i}" for i in range(4)]
    _prepare(a, messages=warnings)
    assert any("1 of 6 expressed genes mapped" in message for message in warnings)


def test_good_overlap_does_not_warn(warnings):
    _prepare(_adata(), messages=warnings)
    assert not any("mouse Entrez" in message for message in warnings)


def test_non_mouse_species_warns_about_the_mouse_calibration(warnings, monkeypatch):
    monkeypatch.setattr(_tage, "_load_mapping", lambda dir: MAPPING.assign(species="human"))
    _prepare(_with_species(_adata(), "human"), messages=warnings)
    assert any("months of mouse age" in message for message in warnings)


def test_mouse_cohort_does_not_warn_about_calibration(warnings):
    _prepare(_with_species(_adata(), "mouse"), messages=warnings)
    assert not any("months of mouse age" in message for message in warnings)


def test_prepare_tage_is_not_public():
    assert "prepare_tage" not in pya.pp.__all__
    assert not hasattr(pya.pp, "prepare_tage")


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
