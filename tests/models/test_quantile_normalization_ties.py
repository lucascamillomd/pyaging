"""DunedinPACE and Stubbs quantile normalisation follows preprocessCore's tie rule.

``normalize.quantiles.use.target`` gives a tie group spanning sorted positions
i..j (0-based) the rank ``(i + j + 2) / 2`` and assigns the mean of the two
neighbouring targets when that rank is a half, else the target at the rank.
"""

import pytest
import torch

from pyaging.models import DunedinPACE, Stubbs

# Five features against a five-value reference: the per-position targets are the
# sorted reference itself, 10, 20, 30, 40, 50.
REFERENCE = [50.0, 10.0, 40.0, 20.0, 30.0]
TIES = [
    # One group at sorted positions 1..3: rank 3, target[2].
    ([0.3, 0.1, 0.3, 0.5, 0.3], [30.0, 10.0, 30.0, 50.0, 30.0]),
    # A pair at positions 1..2: rank 2.5, mean of target[1] and target[2].
    ([0.2, 0.2, 0.9, 0.1, 0.5], [25.0, 25.0, 50.0, 10.0, 40.0]),
    # Four lowest tied at 0..3: rank 2.5.
    ([0.1, 0.1, 0.1, 0.1, 0.9], [25.0, 25.0, 25.0, 25.0, 50.0]),
    # A constant sample: rank 3 everywhere.
    ([0.4] * 5, [30.0] * 5),
    # Two groups: 0..1 (rank 1.5) and 2..4 (rank 4).
    ([0.7, 0.2, 0.7, 0.2, 0.7], [40.0, 15.0, 40.0, 15.0, 40.0]),
]


def _dunedinpace(reference, n_features):
    model = DunedinPACE()
    model.reference_values = reference
    model.preprocess_dependencies = [list(range(n_features))]
    return model


def _stubbs(reference, stds, n_features):
    model = Stubbs()
    model.reference_values = reference
    model.preprocess_dependencies = [stds, list(range(n_features))]
    return model


def _loop_quantile_normalize(x, reference):
    """The per-sample argsort normalisation the clocks used before ties were handled."""
    sorted_gold_standard = torch.sort(torch.tensor(reference, dtype=x.dtype))[0]
    quantile_indices = torch.linspace(0, len(sorted_gold_standard) - 1, steps=x.size(1)).long()
    normalized_data = torch.empty_like(x)
    for i in range(x.size(0)):
        normalized_data[i, torch.argsort(x[i, :])] = sorted_gold_standard[quantile_indices]
    return normalized_data


def _column_major(rows):
    """Batches are cut from a Fortran-ordered matrix, so they arrive column-major."""
    x = torch.tensor(rows, dtype=torch.float64)
    return x.t().contiguous().t()


@pytest.mark.parametrize(("row", "expected"), TIES)
def test_dunedinpace_ties_follow_preprocesscore(row, expected):
    result = _dunedinpace(REFERENCE, 5).preprocess(_column_major([row]))
    assert result.flatten().tolist() == expected


def test_dunedinpace_ties_are_resolved_per_sample_in_one_batch():
    rows = [row for row, _ in TIES]
    result = _dunedinpace(REFERENCE, 5).preprocess(_column_major(rows))
    assert result.tolist() == [expected for _, expected in TIES]


@pytest.mark.parametrize(("row", "expected"), TIES)
def test_stubbs_ties_follow_preprocesscore_before_scaling(row, expected):
    stds = [2.0, 1.0, 0.0, 4.0, 5.0]
    result = _stubbs(REFERENCE, stds, 5).preprocess(_column_major([row]))
    scale = [2.0, 1.0, 1.0, 4.0, 5.0]  # a zero standard deviation is replaced by one
    scaled = [(value - mean) / sd for value, mean, sd in zip(expected, REFERENCE, scale, strict=True)]
    assert result.flatten().tolist() == scaled


@pytest.mark.parametrize("n_samples", [1, 2, 9])
@pytest.mark.parametrize("n_reference", [37, 120])
def test_untied_inputs_are_bit_identical_to_the_argsort_loop(n_samples, n_reference):
    generator = torch.Generator().manual_seed(n_samples * 1000 + n_reference)
    reference = torch.rand(n_reference, generator=generator, dtype=torch.float64).tolist()
    x = torch.rand(23, n_samples, generator=generator, dtype=torch.float64).t()  # column-major
    expected = _loop_quantile_normalize(x, reference)

    assert torch.equal(_dunedinpace(reference, 23).preprocess(x), expected)

    stds = torch.rand(23, generator=generator, dtype=torch.float64).tolist()
    # Stubbs' reference doubles as its per-feature means, so it has one entry per feature.
    stubbs = _stubbs(reference[:23], stds, 23)
    means = torch.tensor(stubbs.reference_values, dtype=x.dtype)
    expected_stubbs = (_loop_quantile_normalize(x, stubbs.reference_values) - means) / torch.tensor(stds, dtype=x.dtype)
    assert torch.equal(stubbs.preprocess(x), expected_stubbs)


def test_float32_batches_keep_their_dtype():
    x = _column_major([TIES[1][0]]).float()
    result = _dunedinpace(REFERENCE, 5).preprocess(x)
    assert result.dtype == torch.float32
    assert result.flatten().tolist() == TIES[1][1]


def test_preprocess_selects_the_model_subset_after_normalising():
    model = _dunedinpace(REFERENCE, 5)
    model.preprocess_dependencies = [[3, 0]]
    result = model.preprocess(_column_major([TIES[1][0]]))
    assert result.flatten().tolist() == [10.0, 25.0]
