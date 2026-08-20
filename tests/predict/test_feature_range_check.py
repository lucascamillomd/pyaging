import anndata
import numpy as np
import pytest

from pyaging.predict._pred_utils import check_feature_ranges


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


def _adata_for(clock_name, features, values):
    matrix = np.asarray(values, dtype=float)
    adata = anndata.AnnData(np.zeros((matrix.shape[0], 1), dtype=float))
    adata.obsm[f"X_{clock_name}"] = matrix
    return adata


class _FakeModel:
    def __init__(self, features, data_type, feature_units=None):
        self.features = features
        self.metadata = {"clock_name": "fakeclock", "data_type": data_type}
        self.feature_units = feature_units


def test_in_range_methylation_produces_no_warning():
    model = _FakeModel(["cg1", "cg2"], "DNA methylation")
    adata = _adata_for("fakeclock", model.features, [[0.1, 0.9], [0.5, 0.5]])
    logger = _RecordingLogger()
    check_feature_ranges(adata, model, logger)
    assert logger.warnings == []


def test_out_of_range_methylation_warns_with_feature_range_and_percent():
    model = _FakeModel(["cg1", "cg2"], "DNA methylation")
    adata = _adata_for("fakeclock", model.features, [[1.5, 0.2], [2.0, 0.3]])
    logger = _RecordingLogger()
    check_feature_ranges(adata, model, logger)
    joined = " ".join(logger.warnings)
    assert "cg1" in joined
    assert "cg2" not in joined
    assert "100.00%" in joined
    assert "[0, 1]" in joined


def test_warning_reports_unit_for_clinical_features():
    model = _FakeModel(["albumin", "age"], "clinical biomarkers")
    adata = _adata_for("fakeclock", model.features, [[4.5, 50.0]])
    logger = _RecordingLogger()
    check_feature_ranges(adata, model, logger)
    joined = " ".join(logger.warnings)
    assert "albumin" in joined
    assert "g/L" in joined


def test_nan_values_are_ignored():
    model = _FakeModel(["cg1"], "DNA methylation")
    adata = _adata_for("fakeclock", model.features, [[np.nan], [0.5]])
    logger = _RecordingLogger()
    check_feature_ranges(adata, model, logger)
    assert logger.warnings == []


def test_nan_values_are_excluded_from_the_reported_percentage():
    # One NaN and one offender in the same column: the NaN must leave the
    # denominator, so this is 100% of the values that were actually there,
    # not 50% of the rows.
    model = _FakeModel(["cg1"], "DNA methylation")
    adata = _adata_for("fakeclock", model.features, [[np.nan], [1.5]])
    logger = _RecordingLogger()
    check_feature_ranges(adata, model, logger)
    joined = " ".join(logger.warnings)
    assert "100.00%" in joined
    assert "50.00%" not in joined


def test_unknown_data_type_produces_no_warning():
    model = _FakeModel(["prot1"], "proteomics")
    adata = _adata_for("fakeclock", model.features, [[-1e6], [1e6]])
    logger = _RecordingLogger()
    check_feature_ranges(adata, model, logger)
    assert logger.warnings == []


def test_mismatched_feature_units_warns_instead_of_raising():
    model = _FakeModel(["cg1"], "DNA methylation", feature_units=["beta value", "beta value"])
    adata = _adata_for("fakeclock", model.features, [[0.4]])
    logger = _RecordingLogger()
    check_feature_ranges(adata, model, logger)
    assert len(logger.warnings) == 1
    assert "Could not resolve feature ranges" in logger.warnings[0]


def test_warning_reports_the_observed_minimum_and_maximum():
    model = _FakeModel(["cg1"], "DNA methylation")
    adata = _adata_for("fakeclock", model.features, [[12.4], [97.3]])
    logger = _RecordingLogger()
    check_feature_ranges(adata, model, logger)
    assert "observed 12.4 to 97.3" in " ".join(logger.warnings)


def test_half_bounded_range_is_phrased_as_below_the_bound():
    model = _FakeModel(["gene1"], "transcriptomics")
    adata = _adata_for("fakeclock", model.features, [[-3.0], [5.0]])
    logger = _RecordingLogger()
    check_feature_ranges(adata, model, logger)
    joined = " ".join(logger.warnings)
    assert "below 0" in joined
    assert "inf" not in joined


def test_check_does_not_mutate_the_matrix():
    model = _FakeModel(["cg1"], "DNA methylation")
    adata = _adata_for("fakeclock", model.features, [[5.0]])
    before = adata.obsm["X_fakeclock"].copy()
    check_feature_ranges(adata, model, _RecordingLogger())
    np.testing.assert_array_equal(adata.obsm["X_fakeclock"], before)


def test_model_without_feature_units_attribute_is_supported():
    model = _FakeModel(["cg1"], "DNA methylation")
    del model.feature_units
    adata = _adata_for("fakeclock", model.features, [[0.4]])
    check_feature_ranges(adata, model, _RecordingLogger())


def test_many_offending_features_are_summarized_not_listed_in_full():
    features = [f"cg{index}" for index in range(50)]
    model = _FakeModel(features, "DNA methylation")
    adata = _adata_for("fakeclock", features, [[5.0] * 50])
    logger = _RecordingLogger()
    check_feature_ranges(adata, model, logger)
    assert len(logger.warnings) <= 6
    assert "50" in " ".join(logger.warnings)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_boundary_violations_are_detected(value):
    model = _FakeModel(["cg1"], "DNA methylation")
    adata = _adata_for("fakeclock", ["cg1"], [[value]])
    logger = _RecordingLogger()
    check_feature_ranges(adata, model, logger)
    assert logger.warnings
