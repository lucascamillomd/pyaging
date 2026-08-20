"""Parity tests for LinAge2 (Fong et al. 2025).

The expected values are the two biological ages the paper prints for its example
NHANES subjects, so these pin the port to the published output rather than to our
reading of the reference R script.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

PARAMS = json.loads((Path(__file__).resolve().parents[2] / "clocks" / "linage2_params.json").read_text())
WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "clocks" / "weights"


def _weights(clock_name):
    path = WEIGHTS_DIR / f"{clock_name}.pt"
    if not path.exists():
        pytest.skip(f"{path} not built; run clocks/notebooks/{clock_name}.ipynb")
    return path


def _model():
    return torch.load(_weights("linage2"), weights_only=False).eval().to(torch.float64)


def _predict(model, rows):
    matrix = torch.tensor([[float(row[name]) for name in model.features] for row in rows], dtype=torch.float64)
    with torch.inference_mode():
        return model(matrix).squeeze(-1).numpy()


def _validation_rows(model):
    rows = []
    for case in PARAMS["validation"]:
        row = dict(case["inputs"])
        row["age"] = case["age_years"]
        row["female"] = case["female"]
        rows.append({name: row[name] for name in model.features})
    return rows


def test_published_example_subjects_reproduce():
    model = _model()
    predicted = _predict(model, _validation_rows(model))
    expected = [case["biological_age"] for case in PARAMS["validation"]]
    np.testing.assert_allclose(predicted, expected, rtol=0, atol=0.02)


def test_batch_of_one_matches_batch_of_two():
    """LinAge2 has no batch dependence; one subject alone must give the same answer."""
    model = _model()
    rows = _validation_rows(model)
    together = _predict(model, rows)
    alone = [_predict(model, [row])[0] for row in rows]
    np.testing.assert_allclose(together, alone, rtol=0, atol=1e-9)


def test_log_of_zero_folds_to_the_lower_clamp_instead_of_raising():
    """A zero in a log-transformed feature must fold to -6, not produce NaN."""
    model = _model()
    row = dict(_validation_rows(model)[0])
    logged = [name for name, flag in zip(PARAMS["features"], PARAMS["log_mask"], strict=True) if flag and name in row]
    row[logged[0]] = 0.0
    (prediction,) = _predict(model, [row])
    assert math.isfinite(prediction)


def test_extreme_inputs_stay_bounded_by_the_six_sd_clamp():
    model = _model()
    row = dict(_validation_rows(model)[0])
    numeric = PARAMS["inputs"]["numeric"][0]
    row[numeric] = row[numeric] * 1e6
    (prediction,) = _predict(model, [row])
    assert 0 < prediction < 150


def test_sex_changes_the_prediction():
    model = _model()
    row = dict(_validation_rows(model)[0])
    as_male = _predict(model, [dict(row, female=0.0)])
    as_female = _predict(model, [dict(row, female=1.0)])
    assert not np.isclose(as_male, as_female)


def test_feature_names_are_harmonized():
    model = _model()
    assert model.features[-2:] == ["age", "female"]
    assert not {"sex", "gender", "Female", "Age", "RIAGENDR", "RIDAGEEX"} & set(model.features)


def test_age_is_in_years_not_months():
    """A 72-year-old must be passed as 72, not 864."""
    model = _model()
    row = dict(_validation_rows(model)[0])
    assert 20 < row["age"] < 100


def test_age_is_a_cox_covariate_not_a_bare_offset():
    """The full and null models weight chronological age differently, so the delta shrinks with age.

    Treating age as an offset that cancels between the two models would make the
    delta identical at every age.
    """
    model = _model()
    row = dict(_validation_rows(model)[0])
    deltas = [_predict(model, [dict(row, age=age)])[0] - age for age in (50.0, 80.0)]
    assert deltas[0] - deltas[1] > 1


def test_mortality_rate_doubling_time_is_the_pre_rounded_constant():
    """``calcBioAge`` rounds ln(2)/beta_null to two decimals before scaling the delta."""
    model = _model()
    for sex in ("male", "female"):
        stored = getattr(model, f"mrdt_{sex}").item()
        unrounded = math.log(2) / getattr(model, f"beta_null_{sex}").item()
        assert stored == round(unrounded, 2)
        assert stored != unrounded


def test_c_reactive_protein_is_not_floored_the_way_the_bioage_clocks_floor_it():
    """LinAge2 takes a plain log with no floor, so 0 and the shared 0.01 floor differ."""
    from pyaging.models._models import CRP_FLOOR_MG_DL

    model = _model()
    row = dict(_validation_rows(model)[0])
    at_zero = _predict(model, [dict(row, c_reactive_protein=0.0)])[0]
    at_floor = _predict(model, [dict(row, c_reactive_protein=CRP_FLOOR_MG_DL)])[0]
    assert math.isfinite(at_zero)
    assert not math.isclose(at_zero, at_floor)


def test_the_fold_covers_the_features_that_skip_z_scoring():
    """``healthcare_use_index`` is never z-scored but is still capped at 6, so 7 and 8 tie."""
    model = _model()
    row = dict(_validation_rows(model)[0])
    at = {
        visits: _predict(model, [dict(row, healthcare_visits_past_year=visits)])[0]
        for visits in (0.0, 5.0, 6.0, 7.0, 8.0)
    }
    assert not math.isclose(at[5.0], at[6.0])
    assert at[6.0] == at[7.0] == at[8.0]

    # 77 (refused), 99 (don't know) and a missing answer all mean "no visit count", not a
    # count of 77 or 99. Reading them literally would fold the index to the 6 cap and move
    # this subject by 0.70 years.
    for sentinel in (77.0, 99.0, float("nan")):
        assert _predict(model, [dict(row, healthcare_visits_past_year=sentinel)])[0] == at[0.0]
    assert abs(at[0.0] - at[6.0]) > 0.5


def test_told_congestive_heart_failure_is_declared_but_never_consumed():
    """``popPCFIfs1`` reads MCQ160B and leaves it out of the 22-item sum. Do not "fix" this."""
    model = _model()
    row = dict(_validation_rows(model)[0])
    assert "told_congestive_heart_failure" in model.features
    answers = [_predict(model, [dict(row, told_congestive_heart_failure=code)])[0] for code in (1.0, 2.0)]
    assert answers[0] == answers[1]


def test_reference_values_put_each_measured_feature_on_the_young_reference_median():
    """Zero-filling an absent lab would pin it at the fold; the reference median is neutral."""
    model = _model()
    logged = dict(zip(PARAMS["features"], PARAMS["log_mask"], strict=True))
    medians = {
        name: (male + female) / 2
        for name, male, female in zip(
            PARAMS["features"], PARAMS["male"]["median"], PARAMS["female"]["median"], strict=True
        )
    }
    for name, value in zip(model.features, model.reference_values, strict=True):
        if name not in medians:
            continue
        expected = math.exp(medians[name]) if logged[name] else medians[name]
        assert math.isclose(value, expected), name


def test_an_absent_lipid_panel_lands_on_the_reference_median_ldl():
    """The reference substitutes a hard 0 mmol/L LDL; reference_values keep that off the table."""
    model = _model()
    lipids = ["total_cholesterol", "hdl_cholesterol", "triglycerides"]
    substitute = {name: model.reference_values[model.features.index(name)] for name in lipids}
    derived = substitute["total_cholesterol"] - substitute["triglycerides"] / 5 - substitute["hdl_cholesterol"]
    index = PARAMS["features"].index("ldl_cholesterol")
    median = (PARAMS["male"]["median"][index] + PARAMS["female"]["median"][index]) / 2
    assert math.isclose(derived, median)

    row = dict(_validation_rows(model)[0])
    reported = _predict(model, [row])[0]
    with_zero_ldl = _predict(model, [dict(row, **dict.fromkeys(lipids, float("nan")))])[0]
    with_reference = _predict(model, [dict(row, **substitute)])[0]
    assert with_zero_ldl < reported < with_reference


def test_an_absent_questionnaire_block_biases_the_estimate_downward():
    """Substituting the healthy reference profile hides a subject's reported ill health.

    The direction and rough size are what the registry notes promise, so they are
    pinned here rather than left to the prose.
    """
    model = _model()
    substitute = dict(zip(model.features, model.reference_values, strict=True))
    unwell = {name: 1.0 for name in PARAMS["derived"]["comorbidity_index"]["inputs"] if name != "told_diabetes"}
    unwell.update(
        told_diabetes=1.0,
        general_health_condition=5.0,  # poor
        health_compared_to_one_year_ago=2.0,  # worse than a year ago
        healthcare_visits_past_year=8.0,  # 16 or more visits
    )
    questionnaire = PARAMS["inputs"]["questionnaire"]

    # Every questionnaire column is absent, so the substitute overwrites all of `unwell`.
    blanked = {name: substitute[name] for name in questionnaire}
    for base in _validation_rows(model):
        reported = _predict(model, [dict(base, **unwell)])[0]
        absent = _predict(model, [dict(base, **blanked)])[0]
        assert absent < reported
        assert 4.5 < reported - absent < 6.0
