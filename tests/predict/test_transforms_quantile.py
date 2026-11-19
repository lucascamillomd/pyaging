"""NumPy quantile normalisation: preprocessCore's tie rule and NaN propagation."""

import numpy as np
import pytest

from pyaging.predict._transforms import quantile_normalize_with_gold_standard

REFERENCE = np.array([50.0, 10.0, 40.0, 20.0, 30.0])
TIES = [
    ([0.3, 0.1, 0.3, 0.5, 0.3], [30.0, 10.0, 30.0, 50.0, 30.0]),
    ([0.2, 0.2, 0.9, 0.1, 0.5], [25.0, 25.0, 50.0, 10.0, 40.0]),
    ([0.1, 0.1, 0.1, 0.1, 0.9], [25.0, 25.0, 25.0, 25.0, 50.0]),
    ([0.4] * 5, [30.0] * 5),
    ([0.7, 0.2, 0.7, 0.2, 0.7], [40.0, 15.0, 40.0, 15.0, 40.0]),
]


def _loop_quantile_normalize(x, gold_standard_means):
    """The per-sample argsort normalisation used before ties were handled."""
    sorted_gold_standard = np.sort(gold_standard_means)
    x_normalized = np.array(x, dtype=np.result_type(x.dtype, sorted_gold_standard.dtype), copy=True)
    quantile_indices = np.round(np.linspace(0, len(sorted_gold_standard) - 1, x.shape[1])).astype(int)
    for i in range(x_normalized.shape[0]):
        x_normalized[i, np.argsort(x_normalized[i, :])] = sorted_gold_standard[quantile_indices]
    return x_normalized


@pytest.mark.parametrize(("row", "expected"), TIES)
def test_ties_follow_preprocesscore(row, expected):
    result = quantile_normalize_with_gold_standard(np.array([row]), REFERENCE)
    assert result.tolist() == [expected]


def test_ties_are_resolved_per_sample_in_one_batch():
    x = np.asfortranarray([row for row, _ in TIES])
    result = quantile_normalize_with_gold_standard(x, REFERENCE)
    assert result.tolist() == [expected for _, expected in TIES]


@pytest.mark.parametrize("n_samples", [1, 2, 9])
@pytest.mark.parametrize("n_reference", [5, 23, 120])
def test_untied_inputs_are_bit_identical_to_the_argsort_loop(n_samples, n_reference):
    rng = np.random.default_rng(n_samples * 1000 + n_reference)
    reference = rng.random(n_reference)
    x = np.asfortranarray(rng.random((n_samples, 23)))
    np.testing.assert_array_equal(
        quantile_normalize_with_gold_standard(x, reference), _loop_quantile_normalize(x, reference)
    )


def test_nan_inputs_stay_nan_and_the_rest_rank_against_the_full_target():
    x = np.array([[0.3, np.nan, 0.1], [0.2, 0.9, 0.5]])
    result = quantile_normalize_with_gold_standard(x, np.array([30.0, 10.0, 20.0]))
    np.testing.assert_array_equal(result, [[20.0, np.nan, 10.0], [10.0, 30.0, 20.0]])


def test_input_is_not_modified():
    x = np.array([[0.3, np.nan, 0.1], [0.2, 0.2, 0.5]])
    original = x.copy()
    quantile_normalize_with_gold_standard(x, np.array([30.0, 10.0, 20.0]))
    np.testing.assert_array_equal(x, original)
