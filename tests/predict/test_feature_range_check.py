import anndata
import numpy as np
import pandas as pd
import pytest

from pyaging.predict._pred_utils import _SCAN_BLOCK_COLUMNS, check_feature_ranges, check_features_in_adata


class _RecordingLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, message, indent_level=2):
        self.warnings.append(message)

    def info(self, message, indent_level=2):
        pass

    # The @progress decorator on check_feature_ranges calls these on the logger
    # it finds as the last positional argument.
    def start_progress(self, message, indent_level=1):
        pass

    def finish_progress(self, message, indent_level=1):
        pass


def _adata_for(clock_name, values):
    matrix = np.asarray(values, dtype=float)
    adata = anndata.AnnData(np.zeros((matrix.shape[0], 1), dtype=float))
    adata.obsm[f"X_{clock_name}"] = matrix
    return adata


class _FakeModel:
    def __init__(self, features, data_type, feature_units=None, reference_values=None):
        self.features = features
        self.metadata = {"clock_name": "fakeclock", "data_type": data_type}
        self.feature_units = feature_units
        self.reference_values = reference_values


def _run_pipeline(model, supplied):
    """Assemble the clock matrix the way predict_age does, then check its ranges.

    ``supplied`` maps feature name to a single sample's value; every other model
    feature is left out, so ``check_features_in_adata`` substitutes for it.
    """
    adata = anndata.AnnData(pd.DataFrame([supplied], index=["sample"], dtype=float))
    check_features_in_adata(adata, model, _RecordingLogger())
    logger = _RecordingLogger()
    check_feature_ranges(adata, model, logger)
    return logger.warnings


def test_in_range_methylation_produces_no_warning():
    model = _FakeModel(["cg1", "cg2"], "DNA methylation")
    adata = _adata_for("fakeclock", [[0.1, 0.9], [0.5, 0.5]])
    logger = _RecordingLogger()
    check_feature_ranges(adata, model, logger)
    assert logger.warnings == []


def test_out_of_range_methylation_warns_with_feature_range_and_percent():
    model = _FakeModel(["cg1", "cg2"], "DNA methylation")
    adata = _adata_for("fakeclock", [[1.5, 0.2], [2.0, 0.3]])
    logger = _RecordingLogger()
    check_feature_ranges(adata, model, logger)
    joined = " ".join(logger.warnings)
    assert "cg1" in joined
    assert "cg2" not in joined
    assert "100.00%" in joined
    assert "[0, 1]" in joined


def test_warning_reports_unit_for_clinical_features():
    model = _FakeModel(["albumin", "age"], "clinical biomarkers")
    adata = _adata_for("fakeclock", [[4.5, 50.0]])
    logger = _RecordingLogger()
    check_feature_ranges(adata, model, logger)
    joined = " ".join(logger.warnings)
    assert "albumin" in joined
    assert "g/L" in joined


def test_nan_values_are_ignored():
    model = _FakeModel(["cg1"], "DNA methylation")
    adata = _adata_for("fakeclock", [[np.nan], [0.5]])
    logger = _RecordingLogger()
    check_feature_ranges(adata, model, logger)
    assert logger.warnings == []


def test_nan_values_are_excluded_from_the_reported_percentage():
    # One NaN and one offender in the same column: the NaN must leave the
    # denominator, so this is 100% of the values that were actually there,
    # not 50% of the rows.
    model = _FakeModel(["cg1"], "DNA methylation")
    adata = _adata_for("fakeclock", [[np.nan], [1.5]])
    logger = _RecordingLogger()
    check_feature_ranges(adata, model, logger)
    joined = " ".join(logger.warnings)
    assert "100.00%" in joined
    assert "50.00%" not in joined


def test_unknown_data_type_produces_no_warning():
    model = _FakeModel(["prot1"], "proteomics")
    adata = _adata_for("fakeclock", [[-1e6], [1e6]])
    logger = _RecordingLogger()
    check_feature_ranges(adata, model, logger)
    assert logger.warnings == []


def test_mismatched_feature_units_warns_instead_of_raising():
    model = _FakeModel(["cg1"], "DNA methylation", feature_units=["beta value", "beta value"])
    adata = _adata_for("fakeclock", [[0.4]])
    logger = _RecordingLogger()
    check_feature_ranges(adata, model, logger)
    assert len(logger.warnings) == 1
    assert "Could not resolve feature ranges" in logger.warnings[0]


def test_warning_reports_the_observed_minimum_and_maximum():
    model = _FakeModel(["cg1"], "DNA methylation")
    adata = _adata_for("fakeclock", [[12.4], [97.3]])
    logger = _RecordingLogger()
    check_feature_ranges(adata, model, logger)
    assert "observed 12.4 to 97.3" in " ".join(logger.warnings)


def test_half_bounded_range_is_phrased_as_below_the_bound():
    model = _FakeModel(["gene1"], "transcriptomics")
    adata = _adata_for("fakeclock", [[-3.0], [5.0]])
    logger = _RecordingLogger()
    check_feature_ranges(adata, model, logger)
    joined = " ".join(logger.warnings)
    assert "below 0" in joined
    assert "inf" not in joined


def test_check_does_not_mutate_the_matrix():
    model = _FakeModel(["cg1"], "DNA methylation")
    adata = _adata_for("fakeclock", [[5.0]])
    before = adata.obsm["X_fakeclock"].copy()
    check_feature_ranges(adata, model, _RecordingLogger())
    np.testing.assert_array_equal(adata.obsm["X_fakeclock"], before)


def test_model_without_feature_units_attribute_is_supported():
    model = _FakeModel(["cg1"], "DNA methylation")
    del model.feature_units
    adata = _adata_for("fakeclock", [[0.4]])
    check_feature_ranges(adata, model, _RecordingLogger())


def test_many_offending_features_are_summarized_not_listed_in_full():
    features = [f"cg{index}" for index in range(50)]
    model = _FakeModel(features, "DNA methylation")
    adata = _adata_for("fakeclock", [[5.0] * 50])
    logger = _RecordingLogger()
    check_feature_ranges(adata, model, logger)
    assert len(logger.warnings) <= 6
    assert "50" in " ".join(logger.warnings)


def test_sentinel_reference_values_for_missing_features_are_not_reported():
    """Eight shipped clocks fill a missing CpG with -1, which is not a beta value.

    Those columns are the pipeline's own substitution, so reporting them blames the
    user for values they never supplied and crowds a real offender out of the report.
    """
    features = [f"cg{index}" for index in range(10)]
    model = _FakeModel(features, "DNA methylation", reference_values=[-1.0] * 10)

    warnings = _run_pipeline(model, {"cg0": 0.4, "cg1": 0.6})

    assert warnings == []


def test_an_out_of_range_value_in_a_supplied_column_is_still_reported():
    """The counterpart: scoping to supplied columns must not silence a real unit error."""
    features = [f"cg{index}" for index in range(10)]
    model = _FakeModel(features, "DNA methylation", reference_values=[-1.0] * 10)

    warnings = _run_pipeline(model, {"cg0": 0.4, "cg1": 62.5})

    joined = " ".join(warnings)
    assert "cg1" in joined
    assert "observed 62.5 to 62.5" in joined
    assert "cg2" not in joined  # a substituted -1 sentinel


def test_the_summary_counts_supplied_features_not_every_clock_feature():
    features = [f"cg{index}" for index in range(10)]
    model = _FakeModel(features, "DNA methylation", reference_values=[-1.0] * 10)

    warnings = _run_pipeline(model, {"cg0": 5.0, "cg1": 0.5})

    assert "1 of 2 supplied features" in warnings[0]


def test_a_zero_filled_missing_feature_is_not_reported_against_a_positive_lower_bound():
    """With no reference values the fill is 0, which is below most clinical bounds."""
    model = _FakeModel(["albumin", "glucose"], "clinical biomarkers")

    warnings = _run_pipeline(model, {"albumin": 46.0})

    assert warnings == []


def test_offenders_are_found_on_both_sides_of_a_scan_block_boundary():
    """The scan walks the columns in blocks, so an offender in each block must survive."""
    width = 2 * _SCAN_BLOCK_COLUMNS + 7
    model = _FakeModel([f"cg{index}" for index in range(width)], "DNA methylation")
    values = np.full((1, width), 0.5)
    values[0, 1] = 5.0
    values[0, _SCAN_BLOCK_COLUMNS + 1] = 5.0
    values[0, width - 1] = 5.0
    logger = _RecordingLogger()

    check_feature_ranges(_adata_for("fakeclock", values), model, logger)

    assert f"3 of {width} supplied features" in logger.warnings[0]
    assert [warning.split(":")[0] for warning in logger.warnings[1:]] == [
        "cg1",
        f"cg{_SCAN_BLOCK_COLUMNS + 1}",
        f"cg{width - 1}",
    ]


def test_the_vectorised_scan_agrees_with_a_column_by_column_reference():
    """Property check over random matrices with NaNs, since the scan is now blocked."""
    generator = np.random.default_rng(7)
    for _ in range(50):
        width = int(generator.integers(1, 30))
        rows = int(generator.integers(1, 6))
        values = generator.normal(0.5, 0.8, size=(rows, width))
        values[generator.random(values.shape) < 0.2] = np.nan
        model = _FakeModel([f"cg{index}" for index in range(width)], "DNA methylation")
        logger = _RecordingLogger()

        check_feature_ranges(_adata_for("fakeclock", values), model, logger)

        expected = []
        for index in range(width):
            column = values[:, index]
            column = column[~np.isnan(column)]
            outside = int(((column < 0.0) | (column > 1.0)).sum())
            if outside:
                expected.append((f"cg{index}", 100 * outside / column.size, column.min(), column.max()))

        if not expected:
            assert logger.warnings == []
            continue
        assert f"{len(expected)} of {width} supplied features" in logger.warnings[0]
        for (feature, percent, low, high), warning in zip(expected[:5], logger.warnings[1:], strict=True):
            assert warning.startswith(f"{feature}: ")
            assert f"{percent:.2f}%" in warning
            assert f"observed {low:g} to {high:g}" in warning


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_boundary_violations_are_detected(value):
    model = _FakeModel(["cg1"], "DNA methylation")
    adata = _adata_for("fakeclock", [[value]])
    logger = _RecordingLogger()
    check_feature_ranges(adata, model, logger)
    assert logger.warnings
