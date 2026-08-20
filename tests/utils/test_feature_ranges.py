import math

import pytest

from pyaging.utils._feature_ranges import get_feature_ranges, load_feature_range_registry, resolve_feature_ranges


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
