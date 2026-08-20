import math

import torch

import pyaging as pya


def _phenoage():
    return torch.load("clocks/weights/phenoage.pt", weights_only=False).eval().to(torch.float64)


def test_phenoage_takes_raw_crp_not_a_logged_value():
    model = _phenoage()
    assert "c_reactive_protein" in model.features
    assert "log_crp" not in model.features


def test_phenoage_applies_natural_log_to_crp():
    """Levine 2018 uses ln(CRP in mg/dL); the clock must apply it, not the user."""
    model = _phenoage()
    index = model.features.index("c_reactive_protein")

    row = torch.zeros(1, len(model.features), dtype=torch.float64)
    row[0, index] = math.e  # ln(e) == 1
    transformed = model.preprocess(row)
    assert math.isclose(transformed[0, index].item(), 1.0, abs_tol=1e-12)


def test_crp_is_registered_in_mg_dl():
    (record,) = pya.utils.resolve_feature_ranges(["c_reactive_protein"], "clinical biomarkers")
    assert record["unit"] == "mg/dL"
    assert record["low"] >= 0.0


def test_a_logged_crp_value_now_falls_outside_the_registered_range():
    """The safeguard should catch a user who passes the old pre-logged value."""
    (record,) = pya.utils.resolve_feature_ranges(["c_reactive_protein"], "clinical biomarkers")
    assert math.log(0.21) < record["low"]
