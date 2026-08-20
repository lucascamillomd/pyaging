"""C-reactive protein is supplied raw, in mg/dL, and logged inside the clock.

Levine 2018 fits phenoage's CRP coefficient against ln(CRP in mg/dL), while the
BioAge-derived clocks use log1p of the same measurement. Users therefore supply
the raw value and each clock applies its own transform, so that one feature name
means one unit package-wide.
"""

import math
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest
import torch

import pyaging as pya
from pyaging.predict._pred_utils import check_feature_ranges, check_features_in_adata

WEIGHTS = Path(__file__).resolve().parents[2] / "clocks" / "weights" / "phenoage.pt"

# clocks/weights/ is gitignored build output and is empty on a clean checkout.
requires_weights = pytest.mark.skipif(
    not WEIGHTS.exists(),
    reason=f"{WEIGHTS} is build output; generate it by running clocks/notebooks/phenoage.ipynb",
)


class _RecordingLogger:
    """Captures pipeline warnings. Mirrors the stub in test_feature_range_check.py."""

    def __init__(self):
        self.warnings = []

    def warning(self, message, indent_level=2):
        self.warnings.append(message)

    def info(self, message, indent_level=2):
        pass

    def error(self, message, indent_level=2):
        pass

    # The @progress decorator calls these on the logger it finds as the last
    # positional argument.
    def start_progress(self, message, indent_level=1):
        pass

    def finish_progress(self, message, indent_level=1):
        pass


class _FakeModel:
    def __init__(self, features, data_type):
        self.features = features
        self.metadata = {"clock_name": "fakeclock", "data_type": data_type}
        self.feature_units = None


def _phenoage():
    return torch.load(WEIGHTS, weights_only=False).eval().to(torch.float64)


def _warnings_for_crp(value):
    """Run the range safeguard over a single CRP column holding ``value``."""
    model = _FakeModel(["c_reactive_protein"], "clinical biomarkers")
    adata = anndata.AnnData(np.zeros((1, 1), dtype=float))
    adata.obsm["X_fakeclock"] = np.array([[value]], dtype=float)
    logger = _RecordingLogger()
    check_feature_ranges(adata, model, logger)
    return " ".join(logger.warnings)


def _predict(frame, imputer_strategy="constant"):
    """Mirror predict_age's ordering: impute, fill missing features, then run the model."""
    model = _phenoage()
    adata = pya.pp.df_to_adata(frame, imputer_strategy=imputer_strategy, verbose=False)
    check_features_in_adata(adata, model, _RecordingLogger())
    row = torch.tensor(np.asarray(adata.obsm["X_phenoage"], dtype=float), dtype=torch.float64)
    with torch.inference_mode():
        return model(row)


# A physiological row in the shipped feature order, CRP excluded.
_ROW = {
    "albumin": 46.0,
    "creatinine": 70.0,
    "glucose": 5.0,
    "lymphocyte_percent": 30.0,
    "mean_cell_volume": 90.0,
    "red_cell_distribution_width": 13.0,
    "alkaline_phosphatase": 70.0,
    "white_blood_cell_count": 7.0,
    "age": 50.0,
}


@requires_weights
def test_phenoage_takes_raw_crp_not_a_logged_value():
    model = _phenoage()
    assert "c_reactive_protein" in model.features
    assert "log_crp" not in model.features


@requires_weights
def test_phenoage_applies_natural_log_to_crp():
    """Levine 2018 uses ln(CRP in mg/dL); the clock must apply it, not the user."""
    model = _phenoage()
    index = model.features.index("c_reactive_protein")

    # Distinct positive values, so a transform applied to the whole tensor rather
    # than to the CRP column alone cannot pass.
    row = torch.arange(1, len(model.features) + 1, dtype=torch.float64).unsqueeze(0)
    row[0, index] = math.e  # ln(e) == 1

    expected = row.clone()
    expected[0, index] = 1.0
    assert torch.equal(model.preprocess(row), expected)


@requires_weights
def test_the_transform_is_natural_log_not_log1p():
    """BioAge's lncrp is log1p of the same measurement; phenoage must not use it."""
    model = _phenoage()
    index = model.features.index("c_reactive_protein")
    row = torch.ones(1, len(model.features), dtype=torch.float64)
    row[0, index] = math.e
    transformed = model.preprocess(row)[0, index].item()
    assert math.isclose(transformed, 1.0, abs_tol=1e-12)
    assert not math.isclose(transformed, math.log1p(math.e), abs_tol=1e-6)


@requires_weights
def test_moving_the_log_into_the_clock_did_not_change_any_prediction():
    """The acceptance gate: raw CRP now must equal a pre-logged CRP before.

    The old contract was preprocess-as-identity over a user-supplied ln(CRP), so
    running the base model directly on a logged row reproduces it exactly.
    """
    model = _phenoage()
    index = model.features.index("c_reactive_protein")

    raw = torch.tensor([[46.0, 70.0, 5.0, 0.0, 30.0, 90.0, 13.0, 70.0, 7.0, 50.0]], dtype=torch.float64)
    raw[0, index] = 0.5
    logged = raw.clone()
    logged[0, index] = math.log(0.5)

    with torch.inference_mode():
        assert model(raw).item() == model.postprocess(model.base_model(logged)).item()


def test_crp_is_registered_in_mg_dl():
    (record,) = pya.utils.resolve_feature_ranges(["c_reactive_protein"], "clinical biomarkers")
    assert record["unit"] == "mg/dL"
    assert record["low"] > 0.0


def test_a_logged_crp_value_now_falls_outside_the_registered_range():
    """The safeguard should catch a user who passes the old pre-logged value."""
    assert "c_reactive_protein" in _warnings_for_crp(math.log(0.21))


def test_a_zero_crp_falls_outside_the_registered_range():
    """The registry floor sits above the log's singularity, so an explicit 0 warns."""
    assert "c_reactive_protein" in _warnings_for_crp(0.0)


def test_a_plausible_crp_does_not_warn():
    assert _warnings_for_crp(0.21) == ""


@requires_weights
def test_an_explicitly_zero_crp_still_predicts_a_finite_age():
    """Datasets encode a below-detection CRP as 0; ln(0) must not reach the output."""
    frame = pd.DataFrame([{**_ROW, "c_reactive_protein": 0.0}])
    assert torch.isfinite(_predict(frame)).all()


@requires_weights
def test_a_constant_imputed_crp_still_predicts_a_finite_age():
    """imputer_strategy='constant' fills a missing reading with 0."""
    frame = pd.DataFrame([{**_ROW, "c_reactive_protein": np.nan}])
    assert torch.isfinite(_predict(frame)).all()


@requires_weights
def test_a_missing_crp_column_still_predicts_a_finite_age():
    """A cohort with no CRP cycle at all: the column is filled with 0 downstream."""
    frame = pd.DataFrame([_ROW])
    assert "c_reactive_protein" not in frame.columns
    assert torch.isfinite(_predict(frame)).all()
