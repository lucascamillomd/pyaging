"""epiTOC2 averages over the CpGs a sample actually has, as Teschendorff's R code does."""

import math

import pytest
import torch

from pyaging.models import EpiTOC3, epiTOC2

DELTA = torch.tensor([0.5, 0.25, 2.0, 1.0], dtype=torch.float64)
BETA0 = torch.tensor([0.1, 0.2, 0.0, 0.5], dtype=torch.float64)


def _model(model_class):
    model = model_class()
    model.delta = DELTA
    model.beta0 = BETA0
    model.reference_values = [-1.0] * len(DELTA)
    return model


def _r_reference(row):
    """``2 * mean((beta - beta0) / (delta * (1 - beta0)))`` over the CpGs present."""
    terms = [
        (beta - b0) / (d * (1 - b0))
        for beta, b0, d in zip(row, BETA0.tolist(), DELTA.tolist(), strict=True)
        if beta != -1 and not math.isnan(beta)
    ]
    return 2 * sum(terms) / len(terms) if terms else math.nan


@pytest.mark.parametrize("model_class", [epiTOC2, EpiTOC3])
def test_complete_samples_keep_the_published_score(model_class):
    x = torch.tensor([[0.3, 0.4, 0.9, 0.7], [0.1, 0.2, 0.0, 0.5]], dtype=torch.float64)
    contrib = (x - BETA0) / (DELTA * (1 - BETA0))
    expected = (2.0 * torch.sum(contrib, dim=1) / x.shape[1]).unsqueeze(1)
    assert torch.equal(_model(model_class)(x), expected)


@pytest.mark.parametrize("model_class", [epiTOC2, EpiTOC3])
def test_sentinel_and_nan_cpgs_are_left_out_of_the_average(model_class):
    rows = [
        [0.3, -1.0, 0.9, 0.7],
        [0.3, math.nan, 0.9, -1.0],
        [-1.0, -1.0, -1.0, 0.7],
    ]
    result = _model(model_class)(torch.tensor(rows, dtype=torch.float64))
    expected = torch.tensor([[_r_reference(row)] for row in rows], dtype=torch.float64)
    torch.testing.assert_close(result, expected, rtol=1e-15, atol=0)


@pytest.mark.parametrize("model_class", [epiTOC2, EpiTOC3])
def test_sample_without_any_cpg_scores_nan(model_class):
    x = torch.tensor([[-1.0, math.nan, -1.0, -1.0], [0.3, 0.4, 0.9, 0.7]], dtype=torch.float64)
    result = _model(model_class)(x)
    assert math.isnan(result[0, 0])
    assert math.isfinite(result[1, 0])


def test_missing_cpgs_do_not_pull_the_score_towards_beta_minus_one():
    complete = [0.3, 0.4, 0.9, 0.7]
    partial = [0.3, 0.4, 0.9, -1.0]
    model = _model(epiTOC2)
    result = model(torch.tensor([complete, partial], dtype=torch.float64)).flatten().tolist()
    assert result[1] == pytest.approx(_r_reference(partial), rel=1e-15)
    assert result[1] > 0
