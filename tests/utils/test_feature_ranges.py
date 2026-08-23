import math
from pathlib import Path

import numpy as np
import pytest
import torch

from pyaging.utils._feature_ranges import get_feature_ranges, load_feature_range_registry, resolve_feature_ranges

WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "clocks" / "weights"


def test_registry_has_schema_version_and_sections():
    registry = load_feature_range_registry()
    assert registry["schema_version"] == 1
    assert "modality_defaults" in registry
    assert "features" in registry


def test_clinical_features_resolve_from_per_feature_entries():
    records = resolve_feature_ranges(["albumin", "age", "female"], "clinical biomarkers")
    by_feature = {record["feature"]: record for record in records}
    assert by_feature["albumin"]["unit"] == "g/L"
    assert by_feature["age"] == {"feature": "age", "unit": "years", "low": 0.0, "high": 122.5}
    assert by_feature["female"]["low"] == 0.0
    assert by_feature["female"]["high"] == 1.0


def test_methylation_features_use_modality_default():
    records = resolve_feature_ranges(["cg00000029", "cg00000108"], "DNA methylation")
    assert all(record["low"] == 0.0 and record["high"] == 1.0 for record in records)
    assert all(record["unit"] == "beta value" for record in records)


@pytest.mark.parametrize("data_type", ["transcriptomics", "histone modification", "chromatin accessibility"])
def test_count_modalities_are_non_negative_and_unbounded_above(data_type):
    (record,) = resolve_feature_ranges(["some_feature"], data_type)
    assert record["low"] == 0.0
    assert math.isinf(record["high"])


def test_per_feature_entry_wins_over_modality_default():
    (record,) = resolve_feature_ranges(["age"], "DNA methylation")
    assert record["high"] == 122.5


def test_unknown_feature_in_unknown_modality_is_unbounded():
    (record,) = resolve_feature_ranges(["mystery"], None)
    assert math.isinf(record["high"]) and math.isinf(-record["low"])
    assert record["unit"] is None


def test_explicit_feature_units_override_registry_unit():
    (record,) = resolve_feature_ranges(["albumin"], "clinical biomarkers", ["g/dL"])
    assert record["unit"] == "g/dL"


def test_none_feature_unit_falls_back_to_the_registry_unit():
    records = resolve_feature_ranges(["age", "albumin"], "clinical biomarkers", [None, "g/dL"])
    by_feature = {record["feature"]: record for record in records}
    assert by_feature["age"]["unit"] == "years"
    assert by_feature["albumin"]["unit"] == "g/dL"


def test_get_feature_ranges_rejects_a_clock_without_features(monkeypatch):
    class _FeaturelessClock:
        features = None
        metadata = {"data_type": "DNA methylation"}

    monkeypatch.setattr("pyaging.predict.load_clock", lambda clock_name, **kwargs: _FeaturelessClock(), raising=True)

    with pytest.raises(ValueError, match="horvath2013"):
        get_feature_ranges("horvath2013")


def _local_clock(monkeypatch, clock_name):
    """Serve a built weight file to ``get_feature_ranges`` instead of downloading it."""
    path = WEIGHTS_DIR / f"{clock_name}.pt"
    if not path.exists():
        pytest.skip(f"{WEIGHTS_DIR} is build output; generate it by running the clocks/notebooks")
    model = torch.load(path, weights_only=False, map_location="cpu")
    monkeypatch.setattr("pyaging.predict.load_clock", lambda name, **kwargs: model, raising=True)
    return model


def test_get_feature_ranges_returns_one_row_per_feature(monkeypatch):
    model = _local_clock(monkeypatch, "phenoage")

    frame = get_feature_ranges("phenoage")

    assert list(frame.columns) == ["feature", "unit", "low", "high"]
    assert list(frame["feature"]) == list(model.features)
    assert (frame["low"] < frame["high"]).all()


def test_get_feature_ranges_prefers_the_units_stored_on_the_clock(monkeypatch):
    """A clock is authoritative about its own units, so a stored copy must win."""
    model = _local_clock(monkeypatch, "phenoage")
    monkeypatch.setattr(model, "feature_units", ["stored unit"] * len(model.features), raising=False)

    frame = get_feature_ranges("phenoage")

    assert list(frame["unit"]) == ["stored unit"] * len(model.features)


def test_get_feature_ranges_falls_back_to_the_registry_without_stored_units(monkeypatch):
    """Clocks built before ``feature_units`` existed have no attribute to read."""
    model = _local_clock(monkeypatch, "phenoage")
    monkeypatch.delattr(model, "feature_units", raising=False)

    frame = get_feature_ranges("phenoage")

    expected = [record["unit"] for record in resolve_feature_ranges(model.features, model.metadata["data_type"])]
    assert list(frame["unit"]) == expected


def test_get_feature_ranges_uses_a_models_internal_range_profile(monkeypatch):
    class _CenteredTranscriptomicsClock:
        features = ["12575"]
        feature_units = None
        feature_range_data_type = "transcriptomics (relative)"
        metadata = {"data_type": "transcriptomics"}

    monkeypatch.setattr(
        "pyaging.predict.load_clock",
        lambda clock_name, **kwargs: _CenteredTranscriptomicsClock(),
        raising=True,
    )

    record = get_feature_ranges("centered-transcriptomics").iloc[0]

    assert math.isinf(-record["low"]) and math.isinf(record["high"])


@pytest.mark.parametrize(
    ("feature", "value"),
    [
        ("creatinine", 2000.0),  # untreated end-stage renal disease
        ("alkaline_phosphatase", 3000.0),  # Paget's disease of bone
        ("white_blood_cell_count", 300.0),  # leukemic hyperleukocytosis
        ("white_blood_cell_count", 0.1),  # post-chemotherapy agranulocytosis
        ("glucose", 55.0),  # hyperosmolar hyperglycemic state
        ("mean_cell_volume", 145.0),  # severe megaloblastic anemia
    ],
)
def test_extreme_but_real_clinical_values_stay_within_bounds(feature, value):
    (record,) = resolve_feature_ranges([feature], "clinical biomarkers")
    assert record["low"] <= value <= record["high"]


@pytest.mark.parametrize(
    ("feature", "unit"),
    [
        ("grimage", "years"),
        ("grimage2timp1", "pg/mL"),
        ("grimage2packyrs", "pack-years"),
        ("grimage2logcrp", "natural log of mg/L"),
        ("cpgpt_timp1", "z-score of a DNAm-predicted plasma protein level"),
    ],
)
def test_dnam_surrogate_features_are_not_treated_as_beta_values(feature, unit):
    """These live on DNA methylation clocks but are ages, protein levels and z-scores."""
    (record,) = resolve_feature_ranges([feature], "DNA methylation")
    assert record["unit"] == unit
    assert (record["low"], record["high"]) != (0.0, 1.0)


@pytest.mark.parametrize("clock_name", ["cpgptgrimage3", "cpgptpcgrimage3"])
def test_surrogate_bounds_contain_the_clocks_own_training_distribution(monkeypatch, clock_name):
    """Both clocks standardize their input, so the scaler carries the training mean and sd.

    A bound the training mean sits near would fire on ordinary data. The upper bound
    must clear three standard deviations; the lower bound may instead clear an order
    of magnitude, since a plasma concentration is right-skewed and its mean minus
    three standard deviations is often negative.
    """
    model = _local_clock(monkeypatch, clock_name)
    mean, deviation = (np.asarray(dependency) for dependency in model.preprocess_dependencies[:2])

    records = resolve_feature_ranges(model.features, model.metadata["data_type"])

    for record, center, spread in zip(records, mean, deviation, strict=True):
        assert record["low"] < max(center - 3 * spread, center / 10), record["feature"]
        assert center + 3 * spread < record["high"], record["feature"]


def test_registry_feature_names_are_lowercase_and_have_no_sex_aliases():
    features = load_feature_range_registry()["features"]
    assert all(name == name.lower() for name in features)
    assert "female" in features
    assert not {"sex", "gender", "male"} & set(features)


def test_registry_bounds_are_ordered():
    for name, entry in load_feature_range_registry()["features"].items():
        low = -math.inf if entry["low"] is None else entry["low"]
        high = math.inf if entry["high"] is None else entry["high"]
        assert low < high, name


def test_relative_transcriptomics_is_unbounded():
    """A cohort-centered expression value is legitimately negative and has no ceiling.

    The modality has to be declared rather than left to the unknown-modality fallback:
    the registry is what documents that the silence is deliberate, not an oversight.
    """
    assert "transcriptomics (relative)" in load_feature_range_registry()["modality_defaults"]

    (record,) = resolve_feature_ranges(["12575"], "transcriptomics (relative)")
    assert record["unit"] is None
    assert math.isinf(-record["low"]) and math.isinf(record["high"])
