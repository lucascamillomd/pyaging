import json
import math
from pathlib import Path

import numpy as np
import pytest

PARAMS = json.loads((Path(__file__).resolve().parents[1] / "clocks" / "linage2_params.json").read_text())


def test_feature_vector_has_59_entries_in_fixed_order():
    assert len(PARAMS["features"]) == 59
    assert len(set(PARAMS["features"])) == 59


def test_feature_names_are_pyaging_snake_case_not_nhanes_codes():
    for name in PARAMS["features"]:
        assert name == name.lower(), name
        assert not name.startswith(("lb", "bp", "bm", "urx", "ss", "mcq", "huq")), name


def test_overlapping_clinical_features_reuse_package_names():
    from pyaging.utils._feature_ranges import load_feature_range_registry

    registry = load_feature_range_registry()["features"]
    shared = {
        "albumin",
        "creatinine",
        "glucose",
        "alkaline_phosphatase",
        "white_blood_cell_count",
        "lymphocyte_percent",
        "mean_cell_volume",
        "red_cell_distribution_width",
        "total_cholesterol",
        "blood_urea_nitrogen",
        "hemoglobin_a1c",
        "systolic_blood_pressure",
    }
    assert shared <= set(PARAMS["features"]) | set(PARAMS["inputs"]["numeric"])
    assert shared <= set(registry)
    # total_cholesterol is the one overlap LinAge2 consumes without modelling: it feeds the
    # Friedewald LDL and is then dropped, so it is an input but not a feature-vector slot.
    assert "total_cholesterol" in PARAMS["inputs"]["numeric"]
    assert "total_cholesterol" not in PARAMS["features"]
    assert shared - {"total_cholesterol"} <= set(PARAMS["features"])


def test_masks_align_with_the_feature_vector():
    assert len(PARAMS["log_mask"]) == 59
    assert len(PARAMS["skip_mask"]) == 59
    assert sum(PARAMS["skip_mask"]) == 5


@pytest.mark.parametrize("sex", ["male", "female"])
def test_per_sex_constants_have_consistent_shapes(sex):
    fit = PARAMS[sex]
    assert len(fit["median"]) == 59
    assert len(fit["mad"]) == 59
    assert len(fit["loadings"]) == 59
    assert all(len(row) == 59 for row in fit["loadings"])
    assert len(fit["pc_index"]) == 17
    assert len(fit["beta"]) == 18
    assert len(fit["means"]) == 18
    assert fit["beta_null"] > 0
    assert math.isclose(fit["mrdt"], round(math.log(2) / fit["beta_null"], 2))


def test_skipped_columns_are_the_five_documented_ones():
    skipped = [name for name, skip in zip(PARAMS["features"], PARAMS["skip_mask"], strict=True) if skip]
    assert len(skipped) == 5


def test_validation_holds_the_two_published_subjects():
    validation = PARAMS["validation"]
    assert len(validation) == 2
    by_seqn = {case["seqn"]: case for case in validation}
    assert math.isclose(by_seqn[8881]["biological_age"], 88.69, abs_tol=0.02)
    assert math.isclose(by_seqn[9106]["biological_age"], 64.36, abs_tol=0.02)
    for case in validation:
        assert set(PARAMS["inputs"]["numeric"]) <= set(case["inputs"])


def test_input_contract_covers_every_name_the_pipeline_consumes():
    inputs = PARAMS["inputs"]
    assert len(inputs["numeric"]) == 57
    assert len(inputs["questionnaire"]) == 26
    assert inputs["age"] == "age"
    assert inputs["female"] == "female"
    supplied = set(inputs["numeric"]) | set(inputs["questionnaire"])
    derived = set(PARAMS["derived"])
    # every model feature is either supplied directly or produced by a documented derivation
    assert set(PARAMS["features"]) <= supplied | derived
    # every derivation names inputs the user actually supplies
    for spec in PARAMS["derived"].values():
        assert set(spec["inputs"]) <= supplied


def test_every_name_resolves_in_the_package_feature_registry():
    from pyaging.utils._feature_ranges import resolve_feature_ranges

    names = sorted(set(PARAMS["features"]) | set(PARAMS["inputs"]["numeric"]) | set(PARAMS["inputs"]["questionnaire"]))
    for record in resolve_feature_ranges(names, "clinical biomarkers"):
        assert record["unit"] is not None, record["feature"]
        assert math.isfinite(record["low"]) and math.isfinite(record["high"]), record["feature"]


@pytest.mark.parametrize("seqn", [8881, 9106])
def test_constants_alone_reproduce_the_published_biological_age(seqn):
    """The JSON is the ground truth for the clock task, so drive the whole pipeline from it."""
    case = {entry["seqn"]: entry for entry in PARAMS["validation"]}[seqn]
    fit = PARAMS["female" if case["female"] else "male"]
    values = dict(case["inputs"])

    cot = values["cotinine"]
    values["smoking_intensity"] = 0 if cot < 10 else (1 if cot < 100 else (2 if cot < 200 else 3))
    binary = PARAMS["derived"]["comorbidity_index"]["inputs"]
    values["comorbidity_index"] = (
        sum((values[name] in (1, 3)) if name == "told_diabetes" else (values[name] == 1) for name in binary) / 22
    )
    health, versus = values["general_health_condition"], values["health_compared_to_one_year_ago"]
    values["self_reported_health_index"] = ((health == 4) * 2 + (health == 5) * 4) * (
        1 - (versus == 1) * 0.5 + (versus == 2)
    )
    visits = values["healthcare_visits_past_year"]
    values["healthcare_use_index"] = 0 if visits in (77, 99) else visits
    values["ldl_cholesterol"] = values["total_cholesterol"] - values["triglycerides"] / 5 - values["hdl_cholesterol"]
    values["urine_albumin_creatinine_ratio"] = values["urine_albumin"] / (values["urine_creatinine"] * 1.1312e-4)

    vector = np.array([values[name] for name in PARAMS["features"]], dtype=float)
    with np.errstate(divide="ignore"):  # log(0) must give -inf and fold to -6, not raise
        vector = np.where(PARAMS["log_mask"], np.log(vector), vector)
    median, mad = np.array(fit["median"]), np.array(fit["mad"])
    z = np.where(PARAMS["skip_mask"], vector, (vector - median) / np.where(mad == 0, 1, mad))
    pcs = np.clip(z, -6, 6) @ np.array(fit["loadings"])

    age_months = round(case["age_years"] * 12)
    covariates = np.concatenate([[age_months], pcs[np.array(fit["pc_index"]) - 1]])
    lp = float(np.array(fit["beta"]) @ (covariates - np.array(fit["means"])))
    lp_null = fit["beta_null"] * (age_months - fit["mean_null"])
    biological_age = (age_months + (lp - lp_null) / math.log(2) * fit["mrdt"]) / 12

    assert math.isclose(biological_age, case["biological_age"], abs_tol=0.01)
