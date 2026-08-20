import math

import pytest

from pyaging.utils._feature_ranges import load_feature_range_registry, resolve_feature_ranges


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
