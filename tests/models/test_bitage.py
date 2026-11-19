"""BiTAge binarises each sample at its non-zero median without writing into the input."""

import math

import pytest
import torch

from pyaging.models import BiTAge


def _loop_binarize(x):
    """The per-sample loop BiTAge used before it was vectorised, applied to a copy."""
    x = x.clone()
    non_zero_mask = x != 0
    for i in range(x.size(0)):
        non_zero_elements = x[i][non_zero_mask[i]]
        if non_zero_elements.nelement() > 0:
            x[i] = (x[i] > non_zero_elements.median()).float()
        else:
            x[i] = torch.zeros_like(x[i])
    return x


def _edge_cases(dtype):
    generator = torch.Generator().manual_seed(0)
    random = torch.rand(6, 9, generator=generator, dtype=dtype)
    random[random < 0.3] = 0
    rows = [
        [0.0, 0.0, 0.0, 0.0],  # all zero
        [0.0, 2.0, 0.0, 1.0],  # even non-zero count: torch's lower median
        [3.0, 1.0, 2.0, 0.0],  # odd non-zero count
        [1.0, 1.0, 1.0, 1.0],  # ties at the median
        [0.0, math.nan, 2.0, 1.0],  # NaN reaches the median
        [-1.0, -3.0, 0.0, 2.0],
    ]
    small = torch.tensor(rows, dtype=dtype)
    return [random, random.t().contiguous().t(), random[:1], small]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matches_the_per_sample_loop(dtype):
    model = BiTAge()
    for x in _edge_cases(dtype):
        expected = _loop_binarize(x)
        result = model.preprocess(x)
        assert result.dtype == x.dtype
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_preprocess_does_not_modify_its_input():
    x = torch.tensor([[0.0, 2.0, 0.0, 1.0], [3.0, 1.0, 2.0, 0.0]], dtype=torch.float64)
    original = x.clone()
    BiTAge().preprocess(x)
    assert torch.equal(x, original)
