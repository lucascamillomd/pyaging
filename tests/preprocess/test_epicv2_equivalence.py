"""epicv2_probe_aggregation must match the straightforward groupby-mean implementation."""

import numpy as np
import pandas as pd
import pytest

from pyaging.preprocess import epicv2_probe_aggregation
from pyaging.preprocess._preprocess import _EPICV2_SUFFIX


def _reference(df):
    """Average every column of a probe CpG through a transposed groupby (the slow path)."""
    cpg_sites = pd.Index(
        [_EPICV2_SUFFIX.sub("", column) if isinstance(column, str) else column for column in df.columns],
        name=df.columns.name,
    )
    if not cpg_sites.duplicated().any():
        return df if df.columns.equals(cpg_sites) else df.set_axis(cpg_sites, axis=1)
    is_probe = pd.Index(df.columns).map(lambda c: isinstance(c, str) and bool(_EPICV2_SUFFIX.search(c)))
    is_probe = np.asarray(cpg_sites.isin(cpg_sites[np.asarray(is_probe, dtype=bool)]), dtype=bool)
    aggregated_probes = df.loc[:, is_probe].T.groupby(cpg_sites[is_probe], sort=False).mean().T
    others = df.loc[:, ~is_probe].set_axis(cpg_sites[~is_probe], axis=1)
    return pd.concat([others, aggregated_probes], axis=1)[cpg_sites.unique()]


def _values(columns, n=4, seed=0, dtype=np.float64):
    rng = np.random.default_rng(seed)
    values = rng.random((n, len(columns))).astype(dtype)
    values[rng.random(values.shape) < 0.2] = np.nan
    return pd.DataFrame(values, index=[f"s{i}" for i in range(n)], columns=columns)


def _mixed_duplicates():
    columns = ["cg3_TC11", "cg1", "cg1_BC11", "cg2_TC21", "cg1_TC12", "cg3_BO11", "cg4", "cg2_BC22"]
    return _values(columns)


def _metadata_with_underscores():
    frame = _values(["cg1_TC11", "cg1_BC12", "cg2", "tissue_type_x", "cg5_TC11"])
    frame.insert(1, "tissue_type", ["blood", "saliva", "blood", "buccal"])
    frame.insert(0, "age_years", [30, 40, 50, 60])
    return frame


def _other_probe_types():
    columns = ["ch.1.123R_TC11", "ch.1.123R_BC11", "rs123_TC11", "rs123_BO11", "nv-GRCh38-chr1-1_TC11", "cg9_TC11"]
    return _values(columns, seed=1)


def _all_missing_group():
    frame = _values(["cg1_TC11", "cg1_BC11", "cg2_TC11", "cg2_BC11"], seed=2)
    frame[["cg1_TC11", "cg1_BC11"]] = np.nan
    frame.loc["s0", ["cg2_TC11", "cg2_BC11"]] = np.nan
    return frame


def _duplicated_sample_index():
    frame = _values(["cg1_TC11", "cg1_BC11", "cg2_TC11"], seed=3)
    frame.index = ["s0", "s0", "s1", "s1"]
    return frame


def _float32_probes():
    return _values(["cg1_TC11", "cg1_BC11", "cg2_TC11", "cg3"], seed=4, dtype=np.float32)


def _nothing_to_aggregate():
    return _values(["cg1", "cg2", "tissue_type"], seed=5)


def _duplicated_labels():
    frame = _values(["cg2_TC11", "cg1_BC11", "cg1_BC11", "cg1_BC12"], seed=6)
    frame.index.name = "sample"
    frame.columns.name = "probe"
    return frame


def _nonstring_columns():
    frame = _values(["cg1_TC11", "cg1_BC11", "cg2"], seed=7)
    frame[0] = 1.0
    return frame


def _integer_probes():
    return pd.DataFrame({"cg1_TC11": [1, 2], "cg1_BC11": [4, 7], "cg2_TC11": [5, 6]}, index=["a", "b"])


def _float32_without_missing():
    return _float32_probes().fillna(0.5)


def _mixed_float_widths():
    frame = _values(["cg1_TC11", "cg1_BC11", "cg2_TC11", "cg2_BC11"], seed=8)
    return frame.astype({"cg1_BC11": np.float32, "cg2_TC11": np.float32})


def _nullable_probes():
    return _values(["cg1_TC11", "cg1_BC11", "cg2_TC11"], seed=9).astype("Float64")


CASES = {
    "float32_without_missing": _float32_without_missing,
    "mixed_float_widths": _mixed_float_widths,
    "nullable_probes": _nullable_probes,
    "mixed_duplicates": _mixed_duplicates,
    "metadata_with_underscores": _metadata_with_underscores,
    "other_probe_types": _other_probe_types,
    "all_missing_group": _all_missing_group,
    "duplicated_sample_index": _duplicated_sample_index,
    "float32_probes": _float32_probes,
    "nothing_to_aggregate": _nothing_to_aggregate,
    "duplicated_labels": _duplicated_labels,
    "nonstring_columns": _nonstring_columns,
    "integer_probes": _integer_probes,
}


@pytest.mark.parametrize("case", CASES)
def test_aggregation_matches_groupby_reference(case):
    frame = CASES[case]()
    original = frame.copy()
    result = epicv2_probe_aggregation(frame, verbose=False)
    pd.testing.assert_frame_equal(result, _reference(frame), check_exact=True)
    pd.testing.assert_frame_equal(frame, original)
