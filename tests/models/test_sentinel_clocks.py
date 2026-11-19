"""Clocks that skip the ``-1`` missing sentinel keep their per-sample results when batched."""

import math

import pytest
import torch

from pyaging.models import EpiCMITHyper, EpiCMITHypo, HypoClock, StemTOCvitro, epiTOC1, stemTOC


def _loop_mean(x):
    means = []
    for row in x:
        filtered_row = row[row != -1]
        means.append(torch.mean(filtered_row) if len(filtered_row) > 0 else x.new_tensor(float("nan")))
    return torch.vstack(means)


def _loop_quantile(x):
    quantiles = []
    for row in x:
        filtered_row = row[row != -1]
        quantiles.append(torch.quantile(filtered_row, 0.95) if len(filtered_row) > 0 else x.new_tensor(float("nan")))
    return torch.vstack(quantiles)


REFERENCE = {
    HypoClock: _loop_mean,
    epiTOC1: _loop_mean,
    EpiCMITHyper: _loop_mean,
    EpiCMITHypo: _loop_mean,
    stemTOC: _loop_quantile,
    StemTOCvitro: _loop_quantile,
}


def _batches(dtype):
    generator = torch.Generator().manual_seed(1)
    batches = []
    for n_features in (1, 2, 7, 371, 385, 678):
        x = torch.rand(12, n_features, generator=generator, dtype=dtype)
        sentinel = torch.rand(12, n_features, generator=generator) < torch.linspace(0, 1, 12)[:, None]
        x[sentinel] = -1
        x[0] = -1  # every CpG missing
        x[1] = torch.rand(n_features, generator=generator, dtype=dtype)  # none missing
        x[2, 0] = math.nan
        batches += [x, x.t().contiguous().t(), x[3:4]]
    return batches


@pytest.mark.parametrize("model_class", list(REFERENCE))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batched_preprocess_matches_the_per_sample_loop(model_class, dtype):
    model = model_class()
    for x in _batches(dtype):
        expected = REFERENCE[model_class](x)
        result = model.preprocess(x)
        assert result.shape == expected.shape
        assert result.dtype == x.dtype
        torch.testing.assert_close(result, expected, rtol=0, atol=0, equal_nan=True)
