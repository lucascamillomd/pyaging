"""Parity tests for the tAge cohort transforms against the authors' R output.

Every expectation here is a fixture dumped from the reference R pipeline (see
``tests/data/tage/README.md``), so a passing test means agreement with the
authors' implementation rather than with another pyaging code path. The stage
chain is mapping -> RLE -> log -> scale -> align -> center; ``align`` is the
predict pipeline's feature matching in pyaging, not a transform, so the
centring test chains from the ``after_align`` fixture.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyaging.preprocess._tage import (
    _center_against_reference,
    _log_transform,
    _rle_normalize,
    _scale_samples,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests/data/tage"
# Paired with rtol=0 at every use: numpy's default rtol of 1e-7 would dominate on
# the RLE stage, whose values reach ~1e6, and let deviations of ~0.1 pass.
TOL = 1e-6

STAGES = (
    "after_mapping",
    "after_rle",
    "after_log",
    "after_scale",
    "after_align",
    "after_center_all",
    "after_center_refgroup",
)


def _load(name: str) -> pd.DataFrame:
    # Fixtures are written in R orientation (genes x samples); transpose to the
    # samples x genes layout the transforms take.
    return pd.read_csv(FIXTURES / f"{name}.csv.gz", index_col=0).T


@pytest.fixture(scope="module")
def stages():
    return {name: _load(name) for name in STAGES}


@pytest.fixture(scope="module")
def reference_ids():
    return FIXTURES.joinpath("reference_group_sample_ids.txt").read_text().split()


def test_rle_matches_reference(stages):
    ours = _rle_normalize(stages["after_mapping"])
    expected = stages["after_rle"]
    assert list(ours.index) == list(expected.index)
    assert list(ours.columns) == list(expected.columns)
    np.testing.assert_allclose(ours.values, expected.values, rtol=0, atol=TOL)


def test_log_matches_reference(stages):
    ours = _log_transform(stages["after_rle"])
    np.testing.assert_allclose(ours.values, stages["after_log"].values, rtol=0, atol=TOL)


def test_scale_matches_reference(stages):
    ours = _scale_samples(stages["after_log"])
    np.testing.assert_allclose(ours.values, stages["after_scale"].values, rtol=0, atol=TOL)


def test_center_all_matches_reference(stages):
    ours = _center_against_reference(stages["after_align"])
    np.testing.assert_allclose(ours.values, stages["after_center_all"].values, rtol=0, atol=TOL)


def test_center_refgroup_matches_reference(stages, reference_ids):
    ours = _center_against_reference(stages["after_align"], reference_index=reference_ids)
    np.testing.assert_allclose(ours.values, stages["after_center_refgroup"].values, rtol=0, atol=TOL)


def test_center_keeps_padded_genes_missing(stages):
    # The align stage pads absent clock genes with NaN so the model's imputer can
    # fill them; centring must not turn them into zeros.
    ours = _center_against_reference(stages["after_align"])
    all_na = stages["after_align"].isna().all(axis=0)
    assert all_na.sum() == 3542
    assert ours.loc[:, all_na].isna().all().all()


def test_center_rejects_an_empty_reference(stages):
    with pytest.raises(ValueError, match="reference"):
        _center_against_reference(stages["after_align"].iloc[:4], reference_index=[])


def test_center_rejects_unknown_reference_samples(stages):
    with pytest.raises(KeyError):
        _center_against_reference(stages["after_align"].iloc[:4], reference_index=["not_a_sample"])


def test_rle_rejects_missing_counts():
    frame = pd.DataFrame([[1.0, np.nan], [3.0, 4.0]])
    with pytest.raises(ValueError, match="missing values"):
        _rle_normalize(frame)


def test_rle_rejects_a_cohort_with_no_shared_gene():
    # Every gene is zero somewhere, so no gene has a nonzero geometric mean.
    frame = pd.DataFrame([[0.0, 5.0], [7.0, 0.0]])
    with pytest.raises(ValueError, match="expressed in every sample"):
        _rle_normalize(frame)


def test_log_transform_is_base_ten():
    frame = pd.DataFrame({"a": [0.0, 9.0], "b": [99.0, 999.0]})
    np.testing.assert_allclose(_log_transform(frame).values, [[0.0, 2.0], [1.0, 3.0]])


def test_scale_is_per_sample():
    # R's scale() works down the columns of a genes x samples matrix, i.e. one
    # z-score per sample across its genes.
    frame = pd.DataFrame([[1.0, 2.0, 3.0], [10.0, 20.0, 60.0]], index=["s1", "s2"])
    scaled = _scale_samples(frame)
    np.testing.assert_allclose(scaled.mean(axis=1).values, [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(scaled.std(axis=1, ddof=1).values, [1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(scaled.loc["s1"].values, [-1.0, 0.0, 1.0])
