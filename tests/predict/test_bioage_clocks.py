"""Parity tests for the clocks ported from the BioAge R package.

The expected values in ``reference_predictions.json`` were produced by BioAge
itself, not by a re-implementation, so they pin these ports to the published
behaviour rather than to our reading of it.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

PARAMS_DIR = Path(__file__).resolve().parents[2] / "clocks" / "bioage_params"
WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "clocks" / "weights"


@pytest.fixture(scope="module")
def reference():
    return json.loads((PARAMS_DIR / "reference_predictions.json").read_text())


def _weights(name):
    """Load a clock's weights, skipping when the gitignored build output is absent."""
    path = WEIGHTS_DIR / f"{name}.pt"
    if not path.exists():
        pytest.skip(f"{path} is build output; generate it by running clocks/notebooks/{name}.ipynb")
    return torch.load(path, weights_only=False)


def _predict(model, rows, features):
    matrix = torch.tensor([[row[name] for name in features] for row in rows], dtype=torch.float64)
    model.eval().to(torch.float64)
    with torch.inference_mode():
        return model(matrix).squeeze(-1).numpy()


def test_kdmage_matches_bioage_reference(reference):
    model = _weights("kdmage")
    predicted = _predict(model, reference["rows"], model.features)
    np.testing.assert_allclose(predicted, reference["expected"]["kdmage"], rtol=0, atol=1e-6)


def test_kdmage_uses_sex_specific_parameters(reference):
    model = _weights("kdmage")
    row = dict(reference["rows"][0])
    as_female = dict(row, female=1.0)
    as_male = dict(row, female=0.0)
    female_prediction = _predict(model, [as_female], model.features)
    male_prediction = _predict(model, [as_male], model.features)
    assert not np.isclose(female_prediction, male_prediction)


def test_kdmage_feature_names_are_harmonized():
    model = _weights("kdmage")
    assert model.features[-2:] == ["age", "female"]
    assert not {"sex", "gender", "Age", "Female"} & set(model.features)


def test_kdmage_takes_raw_crp_not_a_logged_value():
    """CRP is supplied raw in mg/dL package-wide; the clock applies BioAge's log1p."""
    model = _weights("kdmage")
    assert "c_reactive_protein" in model.features
    assert "log_crp" not in model.features


def test_kdmage_applies_log1p_to_crp_alone():
    """BioAge's lncrp is log1p(CRP in mg/dL), not ln, and only that column moves."""
    model = _weights("kdmage")
    index = model.features.index("c_reactive_protein")

    # Distinct positive values, so a transform applied to the whole tensor rather
    # than to the CRP column alone cannot pass.
    row = torch.arange(1, len(model.features) + 1, dtype=torch.float64).unsqueeze(0)
    expected = row.clone()
    expected[0, index] = np.log1p(expected[0, index].item())
    assert torch.equal(model.preprocess(row), expected)


def test_kdmage_survives_a_zero_crp(reference):
    """A below-detection or constant-imputed 0 must not reach log1p unclamped."""
    model = _weights("kdmage")
    row = dict(reference["rows"][0], c_reactive_protein=0.0)
    assert np.isfinite(_predict(model, [row], model.features)).all()
