"""Regression tests for cohort beta references."""

import gzip
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_path = Path(__file__).resolve().parents[1] / "clocks" / "build_dnam_feature_medians.py"
_spec = importlib.util.spec_from_file_location("build_dnam_feature_medians", _path)
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)


def test_epicv2_collapses_replicates_before_taking_the_sample_median():
    beta = pd.DataFrame(
        {
            "cg00000001_TC21": [0.2, 0.3, 0.7],
            "cg00000001_BC21": [0.8, None, 0.9],
            "cg00000002_TC21": [-1.0, 2.0, float("inf")],
            "cg00000003_TC21": [0.0, 0.0, None],
        }
    )

    assert builder.compute_cohort_medians(beta, epicv2=True).to_dict() == pytest.approx(
        {"cg00000001": 0.5, "cg00000003": 0.0}
    )


def test_combining_medians_weights_cohorts_equally_and_keeps_unique_probes():
    hannum = pd.Series({"cg00000001": 0.2, "cg00000002": 0.0})
    boa = pd.Series({"cg00000001": 0.8, "cg00000003": 0.4})

    assert builder.combine_cohort_medians(hannum, boa).to_dict() == pytest.approx(
        {"cg00000001": 0.5, "cg00000002": 0.0, "cg00000003": 0.4}
    )


def test_epicv2_replicate_group_crossing_a_read_block(tmp_path):
    path = tmp_path / "boa.csv.gz"
    with gzip.open(path, "wt") as stream:
        stream.write("probe,s1,s2,s3\n")
        for index in range(4095):
            stream.write(f"cg{index:08d}_TC21,0.1,0.2,0.3\n")
        stream.write("cg00004095_TC21,0.2,0.3,0.7\n")
        stream.write("cg00004095_BC21,0.8,,0.9\n")

    medians, provenance = builder.read_cohort(path, epicv2=True)

    assert medians["cg00004095"] == pytest.approx(0.5)
    assert provenance["source_rows"] == 4097
    assert provenance["cpgs_with_medians"] == 4096
