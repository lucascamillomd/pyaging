"""Pasta, Reg and PastaMouse rank a whole batch at once with the same average ranks."""

import math

import pytest
import torch

from pyaging.models import Pasta, PastaMouse, Reg
from pyaging.models._models import _average_ranks, _global_median


def _loop_preprocess(x, median=None):
    """The per-sample ranking the rank clocks used before batching."""
    median = _global_median(x) if median is None else median.to(device=x.device, dtype=x.dtype)
    x = torch.where(torch.isnan(x), median, x)
    ranked = torch.empty_like(x)
    for i in range(x.size(0)):
        ranked[i] = _average_ranks(x[i])
    return ranked


def _batches(dtype):
    generator = torch.Generator().manual_seed(2)
    heavy_ties = torch.randint(0, 4, (9, 50), generator=generator).to(dtype)
    distinct = torch.rand(9, 50, generator=generator, dtype=dtype)
    with_nan = torch.randint(0, 10, (9, 50), generator=generator).to(dtype)
    with_nan[with_nan == 3] = math.nan
    constant = torch.full((2, 50), 0.25, dtype=dtype)
    signed_zero = torch.tensor([[0.0, -0.0, 1.0, -0.0], [-1.0, 0.0, -0.0, -1.0]], dtype=dtype)
    batches = []
    for x in (heavy_ties, distinct, with_nan, constant, signed_zero):
        batches += [x, x.t().contiguous().t(), x[:1]]
    return batches


@pytest.mark.parametrize("model_class", [Pasta, Reg, PastaMouse])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batched_ranks_match_the_per_sample_loop(model_class, dtype):
    model = model_class()
    for x in _batches(dtype):
        torch.testing.assert_close(model.preprocess(x), _loop_preprocess(x), rtol=0, atol=0)
        context = torch.tensor(2.5, dtype=torch.float64)
        torch.testing.assert_close(model.preprocess(x, context), _loop_preprocess(x, context), rtol=0, atol=0)


def test_batched_ranks_are_one_based_averages():
    x = torch.tensor([[3.0, 1.0, 1.0, 2.0], [5.0, 5.0, 5.0, 5.0]], dtype=torch.float64)
    assert Pasta().preprocess(x).tolist() == [[4.0, 1.5, 1.5, 3.0], [2.5, 2.5, 2.5, 2.5]]
