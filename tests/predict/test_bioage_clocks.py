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


# --- The shared log1p_crp transform ----------------------------------------

CLOCKS = ["kdmage", "homeostaticdysregulation", "phenoagesaopaulo"]


@pytest.mark.parametrize("clock", CLOCKS)
def test_bioage_clocks_apply_log1p_to_crp_alone(clock):
    """BioAge's lncrp is log1p(CRP in mg/dL), not ln, and only that column moves."""
    model = _weights(clock)
    index = model.features.index("c_reactive_protein")

    # Distinct positive values, so a transform applied to the whole tensor rather
    # than to the CRP column alone cannot pass.
    row = torch.arange(1, len(model.features) + 1, dtype=torch.float64).unsqueeze(0)
    expected = row.clone()
    expected[0, index] = np.log1p(expected[0, index].item())
    assert torch.equal(model.preprocess(row), expected)


@pytest.mark.parametrize("clock", CLOCKS)
def test_bioage_clocks_survive_a_zero_crp(reference, clock):
    """A below-detection or constant-imputed 0 must not reach log1p unclamped."""
    model = _weights(clock)
    row = dict(reference["rows"][0], c_reactive_protein=0.0)
    assert np.isfinite(_predict(model, [row], model.features)).all()


# --- kdmage ----------------------------------------------------------------


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


# --- homeostaticdysregulation ----------------------------------------------


def _standardized(model, row, sex):
    """The row's standardized, centered deviation vector for one sex's reference."""
    biomarkers = model.features[:-1]
    values = np.array([math.log1p(row[name]) if name == "c_reactive_protein" else row[name] for name in biomarkers])
    mean = getattr(model, f"reference_mean_{sex}").numpy()
    sd = getattr(model, f"reference_sd_{sex}").numpy()
    return (values - mean) / sd - getattr(model, f"center_{sex}").numpy()


def test_homeostaticdysregulation_matches_bioage_reference(reference):
    model = _weights("homeostaticdysregulation")
    predicted = _predict(model, reference["rows"], model.features)
    np.testing.assert_allclose(predicted, reference["expected"]["homeostaticdysregulation"], rtol=0, atol=1e-6)


def test_homeostaticdysregulation_is_lowest_at_the_reference_center(reference):
    model = _weights("homeostaticdysregulation")
    biomarkers = model.features[:-1]
    center_row = {name: value.item() for name, value in zip(biomarkers, model.reference_mean_female, strict=True)}
    center_row["c_reactive_protein"] = math.expm1(center_row["c_reactive_protein"])
    center_row["female"] = 1.0
    off_center_row = dict(center_row)
    off_center_row[biomarkers[0]] = center_row[biomarkers[0]] * 1.5

    at_center = _predict(model, [center_row], model.features)
    off_center = _predict(model, [off_center_row], model.features)
    assert at_center < off_center


def test_homeostaticdysregulation_output_is_not_in_years():
    model = _weights("homeostaticdysregulation")
    assert "not" in model.metadata["notes"].lower()
    assert "age" not in model.features


def test_homeostaticdysregulation_centers_on_a_nonzero_standardized_center():
    """``hd_calc`` takes column means with ``na.rm = TRUE`` over the full column and
    only then drops incomplete rows, so the surviving rows' mean is not the centering
    constant. The residual offset is real and reaches 0.19 standard deviations; a port
    that treats it as zero is silently wrong by ~1-3% on every subject.
    """
    model = _weights("homeostaticdysregulation")
    assert abs(model.center_male).max().item() > 0.1
    assert abs(model.center_female).max().item() > 0.05


def test_homeostaticdysregulation_center_is_load_bearing_for_the_prediction(reference):
    """Pin the size of the error a dropped ``center`` would introduce, so the term
    cannot be removed as a no-op refinement.
    """
    model = _weights("homeostaticdysregulation")
    predicted = _predict(model, reference["rows"], model.features)

    uncentered = []
    for row in reference["rows"]:
        sex = "female" if row["female"] == 1 else "male"
        deviation = _standardized(model, row, sex) + getattr(model, f"center_{sex}").numpy()
        precision = getattr(model, f"precision_{sex}").numpy()
        squared = deviation @ precision @ deviation
        uncentered.append(math.log(math.sqrt(squared)) / getattr(model, f"log_hd_sd_{sex}").item())

    assert np.abs(np.array(uncentered) - predicted).max() > 0.05


def test_homeostaticdysregulation_buffers_are_keyed_by_biomarker_name_not_position():
    """The JSON happens to list its biomarkers in feature order, so the reindex is a
    no-op and a positional build would be indistinguishable. Permute the JSON and the
    shipped buffers must still be reproduced — including the covariance, which has to
    be permuted on both axes.
    """
    model = _weights("homeostaticdysregulation")
    params = json.loads((PARAMS_DIR / "homeostaticdysregulation.json").read_text())
    biomarkers = model.features[:-1]

    permuted = copy.deepcopy(params)
    for sex in ("male", "female"):
        fit = permuted[sex]
        order = list(reversed(range(len(fit["biomarkers"]))))
        fit["biomarkers"] = [fit["biomarkers"][index] for index in order]
        for key in ("reference_mean", "reference_sd", "standardized_center"):
            fit[key] = [fit[key][index] for index in order]
        fit["standardized_covariance"] = [
            [fit["standardized_covariance"][row][column] for column in order] for row in order
        ]

    assert permuted["male"]["biomarkers"] != params["male"]["biomarkers"]

    for sex in ("male", "female"):
        fit = permuted[sex]
        order = [fit["biomarkers"].index(name) for name in biomarkers]
        for key, buffer in (
            ("reference_mean", f"reference_mean_{sex}"),
            ("reference_sd", f"reference_sd_{sex}"),
            ("standardized_center", f"center_{sex}"),
        ):
            assert [fit[key][index] for index in order] == getattr(model, buffer).tolist(), buffer
        covariance = np.array(fit["standardized_covariance"])[np.ix_(order, order)]
        np.testing.assert_allclose(np.linalg.inv(covariance), getattr(model, f"precision_{sex}").numpy(), rtol=1e-9)


def test_homeostaticdysregulation_reference_values_sit_at_the_reference_centre():
    """An absent biomarker should sit where its standardized, centred deviation is zero,
    so it adds nothing of its own to the distance. The centre is sex-specific and
    ``reference_values`` is one vector, so the compromise minimises the worst-case
    residual: the value whose standardized residual is equal and opposite for the two
    sexes. Creatinine is irreducibly the worst, its two centres being 2.4 standard
    deviations apart.
    """
    model = _weights("homeostaticdysregulation")
    assert model.reference_values is not None
    assert len(model.reference_values) == len(model.features)

    row = dict(zip(model.features, model.reference_values, strict=True))
    male, female = (_standardized(model, row, sex) for sex in ("male", "female"))
    np.testing.assert_allclose(male, -female, atol=1e-9)
    assert np.abs(male).max() < 1.3
    assert model.features[np.abs(male).argmax()] == "creatinine"
    assert np.abs(np.delete(male, np.abs(male).argmax())).max() < 0.55

    assert model.reference_values[-1] == 0.0  # no sex column means the male parameters


def test_homeostaticdysregulation_reference_fill_beats_zero_fill(reference):
    """Pin both measured bounds over every biomarker and reference subject. A zero assay
    reads as a measurement 5 to 25 standard deviations from the reference centre, and
    because the score is a distance, that one column dominates it.
    """
    model = _weights("homeostaticdysregulation")

    worst_reference = worst_zero = 0.0
    for dropped in model.features[:-1]:
        position = model.features.index(dropped)
        for row in reference["rows"]:
            target = _predict(model, [row], model.features)[0]
            filled = _predict(model, [dict(row, **{dropped: model.reference_values[position]})], model.features)
            zeroed = _predict(model, [dict(row, **{dropped: 0.0})], model.features)
            worst_reference = max(worst_reference, abs(filled[0] - target))
            worst_zero = max(worst_zero, abs(zeroed[0] - target))

    assert worst_reference < 3.0
    assert worst_zero > 4.5


def test_homeostaticdysregulation_a_missing_biomarker_biases_the_score_downward(reference):
    """The registry notes warn users that an incomplete panel reads as healthier than it
    is. Pin the direction that claim rests on: the score is a distance, so removing a
    marker's own contribution shrinks it for every marker on average, and by enough to
    matter against the 1.98-6.76 spread the reference subjects occupy.
    """
    model = _weights("homeostaticdysregulation")
    baseline = _predict(model, reference["rows"], model.features)

    worst_drop = 0.0
    for dropped in model.features[:-1]:
        position = model.features.index(dropped)
        filled = _predict(
            model,
            [dict(row, **{dropped: model.reference_values[position]}) for row in reference["rows"]],
            model.features,
        )
        shifts = filled - baseline
        assert shifts.mean() < 0, f"omitting {dropped} does not lower the score on average"
        worst_drop = max(worst_drop, -shifts.min())

    assert worst_drop > 2.5


def test_homeostaticdysregulation_zero_filled_crp_is_the_one_benign_omission(reference):
    """CRP is the exception to the test above, and it is worth pinning so the general
    claim is not overstated. ``log1p`` compresses a clamped zero to 1.8 standard
    deviations below a reference centre that already sits near the floor, because the
    reference cohort was screened to ``crp < 2``.
    """
    model = _weights("homeostaticdysregulation")
    position = model.features.index("c_reactive_protein")
    centre = (model.reference_mean_male + model.center_male * model.reference_sd_male)[position]
    floor = (math.log1p(0.01) - centre) / model.reference_sd_male[position]

    assert abs(floor) < 2.0
    assert 0.15 < model.reference_values[position] < 0.35  # mg/dL, the young-healthy centre


# --- phenoagesaopaulo ------------------------------------------------------


def test_phenoagesaopaulo_matches_bioage_reference(reference):
    model = _weights("phenoagesaopaulo")
    predicted = _predict(model, reference["rows"], model.features)
    np.testing.assert_allclose(predicted, reference["expected"]["phenoagesaopaulo"], rtol=0, atol=1e-6)


def test_phenoagesaopaulo_excludes_the_three_dropped_biomarkers():
    model = _weights("phenoagesaopaulo")
    assert not {"creatinine", "albumin", "alkaline_phosphatase"} & set(model.features)
    assert "age" in model.features


def test_phenoagesaopaulo_increases_with_age(reference):
    model = _weights("phenoagesaopaulo")
    row = dict(reference["rows"][0])
    younger = _predict(model, [dict(row, age=40.0)], model.features)
    older = _predict(model, [dict(row, age=70.0)], model.features)
    assert older > younger


def test_phenoagesaopaulo_has_no_sex_term():
    """The refit is pooled across sexes, unlike kdmage and homeostaticdysregulation,
    so there is no ``female`` column to code and no sex-specific parameter set.
    """
    model = _weights("phenoagesaopaulo")
    assert "female" not in model.features


def test_phenoagesaopaulo_uses_its_own_refit_gompertz_constants():
    """The refit's mortality-to-age constants are fit alongside the coefficients and
    are not Levine's published ones, so reusing ``mortality_to_phenoage`` would be
    wrong even though the algebraic shape is identical.
    """
    model = _weights("phenoagesaopaulo")
    assert model.postprocess_name == "mortality_to_phenoage_saopaulo"
    assert not math.isclose(model.ba_i.item(), 141.50225, abs_tol=0.1)
    assert not math.isclose(model.ba_d.item(), 0.090165, abs_tol=1e-3)
    assert not math.isclose(model.ba_n.item(), -0.00553, abs_tol=1e-4)

    from pyaging.predict._inverse_transforms import mortality_to_phenoage

    linear_predictor = torch.tensor([-6.0], dtype=torch.float64)
    assert not math.isclose(
        model.postprocess(linear_predictor).item(), mortality_to_phenoage(linear_predictor.item()), abs_tol=1.0
    )


def test_phenoagesaopaulo_reference_values_are_the_training_means():
    """An absent predictor should contribute the population-average amount to the
    linear predictor, which for a linear model is exactly its training mean.
    """
    model = _weights("phenoagesaopaulo")
    params = json.loads((PARAMS_DIR / "phenoagesaopaulo.json").read_text())
    crp = model.features.index("c_reactive_protein")

    assert model.reference_values is not None
    assert len(model.reference_values) == len(model.features)
    for position, expected in enumerate(params["training_mean"]):
        stored = model.reference_values[position]
        # CRP is stored raw because preprocess will log1p whatever sits in that slot.
        assert math.isclose(math.log1p(stored) if position == crp else stored, expected)


def test_phenoagesaopaulo_reference_fill_beats_zero_fill(reference):
    """Pin both measured bounds. Substituting a training mean leaves an error bounded
    by that subject's own deviation, whereas the 0 the pipeline would otherwise
    substitute is not a physiological value for any of these assays.
    """
    model = _weights("phenoagesaopaulo")
    worst_reference = worst_zero = 0.0
    for dropped in model.features[:-1]:
        position = model.features.index(dropped)
        for row in reference["rows"]:
            target = _predict(model, [row], model.features)[0]
            filled = _predict(model, [dict(row, **{dropped: model.reference_values[position]})], model.features)
            zeroed = _predict(model, [dict(row, **{dropped: 0.0})], model.features)
            worst_reference = max(worst_reference, abs(filled[0] - target))
            worst_zero = max(worst_zero, abs(zeroed[0] - target))

    assert worst_reference < worst_zero
