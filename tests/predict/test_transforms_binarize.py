"""``binarize`` thresholds each row at its non-zero median without touching its input."""

import warnings

import numpy as np

from pyaging.predict._transforms import binarize


def test_rows_are_thresholded_at_their_non_zero_median():
    x = np.array([[0.0, 1.0, 3.0, 2.0], [5.0, 0.0, 0.0, 1.0]])
    np.testing.assert_array_equal(binarize(x), [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]])


def test_input_is_not_modified():
    x = np.array([[0.0, 1.0, 3.0, 2.0], [5.0, 0.0, 0.0, 1.0]])
    original = x.copy()
    binarize(x)
    np.testing.assert_array_equal(x, original)


def test_all_zero_row_becomes_zeros_without_a_warning():
    x = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = binarize(x)
    np.testing.assert_array_equal(result, [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
