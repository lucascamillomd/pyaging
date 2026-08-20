"""Parity tests for the clocks ported from the BioAge R package.

The expected values in ``reference_predictions.json`` were produced by BioAge
itself, not by a re-implementation, so they pin these ports to the published
behaviour rather than to our reading of it.
"""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import pyaging as pya
from pyaging.predict._pred_utils import check_features_in_adata

PARAMS_DIR = Path(__file__).resolve().parents[2] / "clocks" / "bioage_params"
WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "clocks" / "weights"


class _SilentLogger:
    """The pipeline's logger interface, quieted. Mirrors the stub in test_crp_transform.py."""

    def warning(self, message, indent_level=2):
        pass

    def info(self, message, indent_level=2):
        pass

    def error(self, message, indent_level=2):
        pass

    def start_progress(self, message, indent_level=1):
        pass

    def finish_progress(self, message, indent_level=1):
        pass


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


# --- The biomarker reindex -------------------------------------------------


def _reindexed(params, sex, features):
    """Mirror the notebook's reindex of one sex's parameter vectors."""
    fit = params[sex]
    order = [fit["biomarkers"].index(name) for name in features[:-2]]
    return {key: [fit[key][index] for index in order] for key in ("q", "k", "s")}


def test_kdmage_buffers_are_keyed_by_biomarker_name_not_position():
    """kdmage.json happens to list its biomarkers in feature order, so the reindex is a
    no-op and a positional build would be indistinguishable. Permute the JSON and the
    shipped buffers must still be reproduced, which is only true of a name-keyed build.
    """
    model = _weights("kdmage")
    params = json.loads((PARAMS_DIR / "kdmage.json").read_text())

    permuted = copy.deepcopy(params)
    for sex in ("male", "female"):
        fit = permuted[sex]
        order = list(reversed(range(len(fit["biomarkers"]))))
        fit["biomarkers"] = [fit["biomarkers"][index] for index in order]
        for key in ("q", "k", "s"):
            fit[key] = [fit[key][index] for index in order]

    # Guard against a permutation that did not actually move anything.
    assert permuted["male"]["biomarkers"] != params["male"]["biomarkers"]

    for sex in ("male", "female"):
        rebuilt = _reindexed(permuted, sex, model.features)
        for key, values in rebuilt.items():
            shipped = getattr(model, f"{key}_{sex}").tolist()
            assert values == shipped, f"{key}_{sex} does not follow the biomarker names"


# --- Missing biomarkers ----------------------------------------------------


def _drop_one_estimate(params, row, features, dropped):
    """KDM with one biomarker's numerator term removed: R's ``na.rm = TRUE``.

    The denominator still runs over all nine markers, because ``kdm_calc`` does not
    rescale the estimate for missingness.
    """
    biomarkers = features[:-2]
    fit = params["female" if row["female"] == 1 else "male"]
    order = [fit["biomarkers"].index(name) for name in biomarkers]
    q, k, s = ([fit[key][index] for index in order] for key in ("q", "k", "s"))
    numerator = 0.0
    for position, name in enumerate(biomarkers):
        if name == dropped:
            continue
        value = math.log1p(row[name]) if name == "c_reactive_protein" else row[name]
        numerator += (value - q[position]) * k[position] / s[position] ** 2
    denominator = sum((k[i] / s[i]) ** 2 for i in range(len(biomarkers)))
    s_ba2 = fit["s_ba2"]
    return (numerator + row["age"] / s_ba2) / (denominator + 1 / s_ba2)


def test_kdmage_reference_values_are_the_mean_of_the_sex_specific_intercepts():
    """An absent biomarker should contribute ~zero to the numerator, so the reference
    is q. q is sex-specific and reference_values is one vector, hence the mean.
    """
    model = _weights("kdmage")
    assert model.reference_values is not None
    crp = model.features.index("c_reactive_protein")

    for position in range(len(model.features) - 2):
        expected = (model.q_male[position].item() + model.q_female[position].item()) / 2
        stored = model.reference_values[position]
        # CRP is stored raw because preprocess will log1p whatever sits in that slot.
        assert math.isclose(math.log1p(stored) if position == crp else stored, expected)

    assert model.reference_values[-2] == 52.5  # midpoint of the 30-75 training window
    assert model.reference_values[-1] == 0.0  # no sex column means the male parameters


@pytest.mark.parametrize("dropped", ["albumin", "forced_expiratory_volume", "c_reactive_protein"])
def test_kdmage_reference_fill_beats_zero_fill_for_a_missing_biomarker(reference, dropped):
    """Filling an absent biomarker with its reference must land far closer to the
    exact na.rm estimate than the 0 the pipeline would otherwise substitute.
    """
    model = _weights("kdmage")
    params = json.loads((PARAMS_DIR / "kdmage.json").read_text())
    position = model.features.index(dropped)

    for row in reference["rows"]:
        target = _drop_one_estimate(params, row, model.features, dropped)
        with_reference = _predict(model, [dict(row, **{dropped: model.reference_values[position]})], model.features)
        with_zero = _predict(model, [dict(row, **{dropped: 0.0})], model.features)
        reference_error = abs(with_reference[0] - target)
        zero_error = abs(with_zero[0] - target)
        assert reference_error < zero_error


def test_kdmage_worst_case_missing_biomarker_error_is_bounded(reference):
    """Pin both measured bounds. The reference is the mean of two sex-specific
    intercepts, so it cancels the numerator term only approximately; the residual is
    the half-gap between the sexes, largest for forced expiratory volume.
    """
    model = _weights("kdmage")
    params = json.loads((PARAMS_DIR / "kdmage.json").read_text())

    worst_reference = worst_zero = 0.0
    for dropped in model.features[:-2]:
        position = model.features.index(dropped)
        for row in reference["rows"]:
            target = _drop_one_estimate(params, row, model.features, dropped)
            filled = _predict(model, [dict(row, **{dropped: model.reference_values[position]})], model.features)
            zeroed = _predict(model, [dict(row, **{dropped: 0.0})], model.features)
            worst_reference = max(worst_reference, abs(filled[0] - target))
            worst_zero = max(worst_zero, abs(zeroed[0] - target))

    assert worst_reference < 8.0
    assert worst_zero > 50.0


def test_kdmage_a_missing_column_uses_the_reference_through_the_pipeline(reference):
    """The vector is only worth setting if predict_age's imputation actually reads it."""
    model = _weights("kdmage")
    params = json.loads((PARAMS_DIR / "kdmage.json").read_text())
    row = reference["rows"][0]

    frame = pd.DataFrame([{name: row[name] for name in model.features if name != "albumin"}])
    adata = pya.pp.df_to_adata(frame, imputer_strategy="constant", verbose=False)
    check_features_in_adata(adata, model, _SilentLogger())
    matrix = torch.tensor(np.asarray(adata.obsm["X_kdmage"], dtype=float), dtype=torch.float64)
    model.eval().to(torch.float64)
    with torch.inference_mode():
        predicted = model(matrix).squeeze(-1).numpy()[0]

    assert math.isclose(predicted, _drop_one_estimate(params, row, model.features, "albumin"), abs_tol=2.0)
