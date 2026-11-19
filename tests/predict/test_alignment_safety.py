"""Feature alignment keeps missing values, covariates and failures visible."""

import warnings
import weakref
from unittest.mock import Mock

import anndata
import numpy as np
import pandas as pd
import pytest
import torch

import pyaging as pya
from pyaging.predict import _pred_utils
from pyaging.predict._pred_utils import check_features_in_adata, predict_ages_with_model
from tests.helpers import RecordingLogger


class _WeightedClock(pya.models.pyagingModel):
    """Predicts ``x @ weights`` and records what it was given."""

    def __init__(self, features, *, name="weighted", reference_values=None, weights=None, **metadata):
        super().__init__()
        self.features = list(features)
        self.reference_values = reference_values
        self.metadata.update(clock_name=name, **metadata)
        weights = [10.0**position for position in range(len(features))] if weights is None else weights
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float64))
        self.seen = []

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x

    def forward(self, x):
        self.seen.append(x.clone())
        return (x * self.weights).sum(dim=1, keepdim=True)


def _adata(rows, columns, index=None):
    frame = pd.DataFrame(rows, columns=columns, index=index, dtype=float)
    return anndata.AnnData(frame)


def _our_warnings(records):
    return [str(record.message) for record in records if issubclass(record.category, UserWarning)]


def _use_model(monkeypatch, *models):
    by_name = {model.metadata["clock_name"]: model for model in models}
    load = Mock(side_effect=lambda name, *args, **kwargs: by_name[name])
    monkeypatch.setattr("pyaging.predict._pred.load_clock", load)
    return load


# 1. NaN cells inside supplied columns


def test_nan_cells_in_supplied_columns_use_finite_reference_values():
    model = _WeightedClock(["a", "b", "c"], reference_values=[10.0, 20.0, 30.0])
    values = [[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]]
    adata = _adata(values, ["a", "b", "c"])

    with pytest.warns(UserWarning, match=r"1 of 2 samples.*1 of 3 supplied features"):
        check_features_in_adata(adata, model, RecordingLogger())

    np.testing.assert_array_equal(adata.obsm["X_weighted"], [[1.0, 20.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_array_equal(adata.X, values)
    assert adata.uns["weighted_percent_nan_values"] == pytest.approx(100 / 6)
    assert type(adata.uns["weighted_percent_nan_values"]) is float
    # The column was supplied, so it is not counted as a missing feature.
    assert adata.uns["weighted_percent_na"] == 0
    assert adata.uns["weighted_missing_features"] == []


def test_nan_cells_fall_back_to_packaged_medians_and_otherwise_stay_nan(monkeypatch):
    monkeypatch.setattr(_pred_utils, "_load_dnam_feature_medians", lambda: {"cg00000001": 0.6, "cg00000002": 0.7})
    model = _WeightedClock(
        ["cg00000000", "cg00000001", "cg00000002"], species="Homo sapiens", data_type="DNA methylation"
    )
    adata = _adata([[np.nan, np.nan, 0.5], [0.2, 0.3, 0.4]], ["cg00000000", "cg00000001", "cg00000002"])

    with pytest.warns(UserWarning):
        check_features_in_adata(adata, model, RecordingLogger())

    # cg00000000 has no packaged median: it stays NaN rather than becoming 0.
    np.testing.assert_array_equal(adata.obsm["X_weighted"], [[np.nan, 0.6, 0.5], [0.2, 0.3, 0.4]])


def test_non_finite_reference_entries_fall_back_to_medians_for_eligible_clocks(monkeypatch):
    monkeypatch.setattr(_pred_utils, "_load_dnam_feature_medians", lambda: {"cg00000001": 0.6})
    model = _WeightedClock(
        ["cg00000000", "cg00000001"],
        reference_values=[0.1, np.nan],
        species="Homo sapiens",
        data_type="DNA methylation",
    )
    adata = _adata([[0.9, np.nan], [np.nan, 0.3]], ["cg00000000", "cg00000001"])

    with pytest.warns(UserWarning):
        check_features_in_adata(adata, model, RecordingLogger())

    # A finite reference wins; only the non-finite entry falls through to the median.
    np.testing.assert_array_equal(adata.obsm["X_weighted"], [[0.9, 0.6], [0.1, 0.3]])


@pytest.mark.parametrize(
    "metadata, use_medians",
    [({"species": "Mus musculus", "data_type": "DNA methylation"}, True), ({}, True), (None, False)],
)
def test_ineligible_clocks_without_references_leave_nan_cells_nan(monkeypatch, metadata, use_medians):
    monkeypatch.setattr(_pred_utils, "_load_dnam_feature_medians", lambda: {"cg00000001": 0.6})
    model = _WeightedClock(["cg00000000", "cg00000001"], **(metadata or {}))
    if metadata is None:
        model.metadata.update(species="Homo sapiens", data_type="DNA methylation")
        model.use_dnam_median_fallback = use_medians
    adata = _adata([[0.1, np.nan], [0.2, 0.3]], ["cg00000000", "cg00000001"])

    with pytest.warns(UserWarning):
        check_features_in_adata(adata, model, RecordingLogger())

    np.testing.assert_array_equal(adata.obsm["X_weighted"], [[0.1, np.nan], [0.2, 0.3]])


def test_models_that_handle_nan_receive_nan_cells_untouched():
    model = _WeightedClock(["a", "b"], reference_values=[10.0, 20.0])
    model.handles_nan = True
    adata = _adata([[1.0, np.nan], [3.0, 4.0]], ["a", "b"])

    with pytest.warns(UserWarning, match="left"):
        check_features_in_adata(adata, model, RecordingLogger())

    np.testing.assert_array_equal(adata.obsm["X_weighted"], [[1.0, np.nan], [3.0, 4.0]])
    assert adata.uns["weighted_percent_nan_values"] == pytest.approx(25.0)


def test_complete_input_records_zero_nan_percentage_without_warning():
    model = _WeightedClock(["a", "b", "absent"], reference_values=[0.0, 0.0, 5.0])
    adata = _adata([[1.0, 2.0]], ["a", "b"])

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        check_features_in_adata(adata, model, RecordingLogger())

    assert _our_warnings(records) == []
    assert adata.uns["weighted_percent_nan_values"] == 0.0
    assert type(adata.uns["weighted_percent_nan_values"]) is float


def test_predict_age_warns_about_nan_cells_even_when_quiet(monkeypatch):
    model = _WeightedClock(["a", "b"], reference_values=[10.0, 20.0])
    _use_model(monkeypatch, model)
    adata = _adata([[1.0, np.nan], [3.0, 4.0]], ["a", "b"])

    with pytest.warns(UserWarning, match="NaN"):
        pya.pred.predict_age(adata, "weighted", verbose=False)

    np.testing.assert_array_equal(adata.obs["weighted"], [201.0, 43.0])
    assert adata.uns["weighted_percent_nan_values"] == pytest.approx(25.0)


# 2. Samples without any data


@pytest.mark.parametrize("handles_nan", [False, True])
def test_samples_whose_supplied_features_are_all_nan_get_nan_predictions(handles_nan):
    model = _WeightedClock(["a", "b", "absent"], reference_values=[10.0, 20.0, 30.0])
    model.handles_nan = handles_nan
    adata = _adata([[1.0, 2.0], [np.nan, np.nan], [3.0, np.nan]], ["a", "b"], index=["s0", "empty", "s2"])
    logger = RecordingLogger()

    with pytest.warns(UserWarning):
        check_features_in_adata(adata, model, logger)
    # The empty row is never filled from the reference.
    assert np.isnan(adata.obsm["X_weighted"][1, :2]).all()
    with pytest.warns(UserWarning, match=r"1 sample.*'empty'"):
        predictions = predict_ages_with_model(adata, model, "cpu", 2, logger)

    predictions = predictions.numpy().ravel()
    assert np.isnan(predictions[1])
    assert predictions[0] == pytest.approx(1 + 20 + 3000)
    expected_s2 = 3 + (20.0 if not handles_nan else np.nan) * 10 + 3000
    np.testing.assert_array_equal(predictions[2], expected_s2)
    # A per-sample clock never sees a sample that carries no data.
    assert sum(len(batch) for batch in model.seen) == 2
    assert any("empty" in message for message in logger.warnings)


def test_cohort_steps_still_see_samples_without_data():
    class _CohortClock(_WeightedClock):
        def prepare_cohort_context(self, matrix):
            return torch.tensor([float(matrix.shape[0])], dtype=torch.float64)

        def predict_with_cohort_context(self, batch, context):
            return torch.nan_to_num(batch).sum(dim=1, keepdim=True) + context

    model = _CohortClock(["a", "b"])
    adata = _adata([[1.0, 2.0], [np.nan, np.nan], [3.0, 4.0]], ["a", "b"], index=["s0", "empty", "s2"])
    # Samples without usable data are reported once, by name, when predicted.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_features_in_adata(adata, model, RecordingLogger())

    with pytest.warns(UserWarning, match="empty"):
        predictions = predict_ages_with_model(adata, model, "cpu", 2, RecordingLogger()).numpy().ravel()

    # The cohort context counted all three samples, as the author's code would.
    np.testing.assert_array_equal(predictions, [6.0, np.nan, 10.0])


def test_all_empty_cohort_predicts_nan_without_running_the_model():
    model = _WeightedClock(["a", "b"], reference_values=[10.0, 20.0])
    adata = _adata([[np.nan, np.nan]] * 7, ["a", "b"], index=[f"s{i}" for i in range(7)])

    # Samples without usable data are reported once, by name, when predicted.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_features_in_adata(adata, model, RecordingLogger())
    with pytest.warns(UserWarning, match=r"7 samples.*'s0'.*'s4'"):
        predictions = predict_ages_with_model(adata, model, "cpu", 4, RecordingLogger())

    assert predictions.shape[0] == 7
    assert torch.isnan(predictions).all()
    assert model.seen == []


def test_predict_age_writes_nan_for_an_empty_sample(monkeypatch):
    model = _WeightedClock(["a", "b"], reference_values=[10.0, 20.0])
    _use_model(monkeypatch, model)
    adata = _adata([[1.0, 2.0], [np.nan, np.nan]], ["a", "b"])

    with pytest.warns(UserWarning):
        pya.pred.predict_age(adata, "weighted", verbose=False)

    assert adata.obs["weighted"].iloc[0] == 21.0
    assert np.isnan(adata.obs["weighted"].iloc[1])


# 3. Required per-sample covariates


def test_covariates_absent_from_features_raise_before_writing_anything():
    model = _WeightedClock(["a", "age", "female"], reference_values=[0.0, 65.0, 0.0])
    adata = _adata([[1.0], [2.0]], ["a"])
    adata.obs["age"] = [30.0, 40.0]

    with pytest.raises(ValueError) as caught:
        check_features_in_adata(adata, model, RecordingLogger())

    message = str(caught.value)
    assert "weighted" in message
    assert "'age'" in message and "'female'" in message
    assert "var_names" in message and "metadata_cols" in message
    assert "adata.obs" in message and "'age'" in message.split("adata.obs")[-1]
    assert not adata.obsm
    assert not adata.uns


def test_covariate_error_does_not_mention_obs_when_obs_lacks_the_column():
    model = _WeightedClock(["a", "female"])
    adata = _adata([[1.0]], ["a"])

    with pytest.raises(ValueError, match="female") as caught:
        check_features_in_adata(adata, model, RecordingLogger())

    assert "adata.obs" not in str(caught.value)


def test_predict_age_raises_for_covariates_in_metadata_cols(monkeypatch):
    model = _WeightedClock(["a", "age"], reference_values=[0.0, 65.0])
    _use_model(monkeypatch, model)
    frame = pd.DataFrame({"a": [1.0, 2.0], "age": [30.0, 40.0]}, index=["s0", "s1"])
    adata = pya.pp.df_to_adata(frame, metadata_cols=["age"], imputer_strategy="constant", verbose=False)

    with pytest.raises(ValueError, match="metadata_cols"):
        pya.pred.predict_age(adata, "weighted", verbose=False)

    assert "weighted" not in adata.obs
    assert not adata.obsm


def test_nan_covariate_is_not_filled_and_its_sample_predicts_nan():
    model = _WeightedClock(["a", "age"], reference_values=[0.0, 65.0])
    adata = _adata([[1.0, 30.0], [2.0, np.nan]], ["a", "age"], index=["s0", "no_age"])

    # Samples without usable data are reported once, by name, when predicted.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_features_in_adata(adata, model, RecordingLogger())
    assert np.isnan(adata.obsm["X_weighted"][1, 1])
    with pytest.warns(UserWarning, match="no_age"):
        predictions = predict_ages_with_model(adata, model, "cpu", 8, RecordingLogger()).numpy().ravel()

    assert predictions[0] == 301.0
    assert np.isnan(predictions[1])


def test_clock_validation_runs_before_the_generic_covariate_check(monkeypatch):
    model = _WeightedClock(["a", "age"])
    model.validate_inputs = Mock(side_effect=ValueError("the clock's own message"))
    _use_model(monkeypatch, model)

    with pytest.raises(ValueError, match="the clock's own message"):
        pya.pred.predict_age(_adata([[1.0]], ["a"]), "weighted", verbose=False)


# 4. Failures leave no half-written state


def test_every_feature_missing_raises_before_writing_anything():
    model = _WeightedClock(["a", "b"])
    adata = _adata([[1.0]], ["unrelated"])

    with pytest.raises(NameError, match="Every single feature"):
        check_features_in_adata(adata, model, RecordingLogger())

    assert not adata.obsm
    assert not adata.uns


def test_failed_clock_removes_its_matrix_and_keeps_earlier_results(monkeypatch):
    first = _WeightedClock(["a"], name="first")
    second = _WeightedClock(["a"], name="second")
    second.forward = Mock(side_effect=RuntimeError("model failure"))
    _use_model(monkeypatch, first, second)
    adata = _adata([[1.0], [2.0]], ["a"])

    with pytest.raises(RuntimeError, match="model failure"):
        pya.pred.predict_age(adata, ["first", "second"], clean=False, verbose=False)

    np.testing.assert_array_equal(adata.obs["first"], [1.0, 2.0])
    assert list(adata.obsm) == ["X_first"]


# 5. Backed and empty X


def test_backed_input_with_unsorted_feature_positions_matches_in_memory(monkeypatch, tmp_path):
    model = _WeightedClock(["a", "b", "c"], reference_values=[0.0, 0.0, 0.0])
    _use_model(monkeypatch, model)
    # Columns in the reverse of the clock's order, so the read indices are unsorted.
    frame = pd.DataFrame([[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]], columns=["c", "b", "a"], index=["s0", "s1"])
    path = tmp_path / "cohort.h5ad"
    anndata.AnnData(frame).write_h5ad(path)

    backed = anndata.read_h5ad(path, backed="r")
    try:
        pya.pred.predict_age(backed, "weighted", verbose=False)
        np.testing.assert_array_equal(backed.obs["weighted"], [321.0, 654.0])
    finally:
        backed.file.close()


def test_missing_x_raises_a_clear_error():
    adata = anndata.AnnData(obs=pd.DataFrame(index=["s0"]), var=pd.DataFrame(index=["a"]))

    with pytest.raises(ValueError, match=r"adata\.X"):
        check_features_in_adata(adata, _WeightedClock(["a"]), RecordingLogger())

    assert not adata.obsm


# 6. Clock-name handling


def test_clock_names_are_stripped_lowercased_and_deduplicated(monkeypatch):
    model = _WeightedClock(["a"], name="weighted")
    load = _use_model(monkeypatch, model)
    adata = _adata([[1.0]], ["a"])

    pya.pred.predict_age(adata, [" Weighted ", "WEIGHTED", "weighted"], verbose=False)

    assert [call.args[0] for call in load.call_args_list] == ["weighted"]
    assert list(adata.obs.columns) == ["weighted"]


def test_load_clock_strips_whitespace(monkeypatch):
    impl = Mock(return_value="model")
    monkeypatch.setattr(_pred_utils, "_load_clock_impl", impl)

    assert _pred_utils.load_clock("  HorVath2013\n", "cpu", logger=RecordingLogger()) == "model"

    assert impl.call_args.args[0] == "horvath2013"


def test_clean_removes_the_matrix_under_the_clocks_own_name(monkeypatch):
    model = _WeightedClock(["a"], name="canonical")
    monkeypatch.setattr("pyaging.predict._pred.load_clock", lambda *args, **kwargs: model)
    adata = _adata([[1.0]], ["a"])

    pya.pred.predict_age(adata, "alias", verbose=False)

    assert adata.obs["canonical"].tolist() == [1.0]
    assert not adata.obsm


# 7. Memory release


def test_garbage_collection_is_skipped_on_cpu(monkeypatch):
    collect = Mock()
    monkeypatch.setattr("pyaging.predict._pred.gc.collect", collect)
    _use_model(monkeypatch, _WeightedClock(["a"], name="first"), _WeightedClock(["a"], name="second"))

    pya.pred.predict_age(_adata([[1.0]], ["a"]), ["first", "second"], verbose=False)

    collect.assert_not_called()


def test_garbage_collection_and_cache_release_run_per_clock_on_cuda(monkeypatch):
    from pyaging.predict import _pred

    collect, empty_cache = Mock(), Mock()
    monkeypatch.setattr(_pred.gc, "collect", collect)
    monkeypatch.setattr(_pred.torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(_pred, "set_torch_device", lambda: torch.device("cuda"))
    monkeypatch.setattr(_pred, "predict_ages_with_model", lambda adata, *args, **kwargs: torch.zeros(adata.n_obs, 1))
    _use_model(monkeypatch, _WeightedClock(["a"], name="first"), _WeightedClock(["a"], name="second"))

    pya.pred.predict_age(_adata([[1.0]], ["a"]), ["first", "second"], verbose=False)

    assert collect.call_count == 2
    assert empty_cache.call_count == 2


def test_clock_with_bound_input_hooks_is_released_before_the_next_loads(monkeypatch):
    references = []

    class _ValidatingClock(_WeightedClock):
        def validate_inputs(self, adata):
            pass

    def load(name, *args, **kwargs):
        assert all(reference() is None for reference in references)
        model = _ValidatingClock(["a"], name=name)
        references.append(weakref.ref(model))
        return model

    monkeypatch.setattr("pyaging.predict._pred.load_clock", load)

    pya.pred.predict_age(_adata([[1.0]], ["a"]), ["first", "second"], verbose=False)

    assert len(references) == 2


# 8. Weight prefetching


def test_weights_for_several_uncached_clocks_are_prefetched_once(monkeypatch):
    prefetch = Mock()
    monkeypatch.setattr("pyaging.predict._pred.prefetch_clock_weights", prefetch)
    models = [_WeightedClock(["a"], name=name) for name in ("first", "second", "third")]
    _use_model(monkeypatch, *models)
    cache = pya.pred.ClockCache(maxsize=3)
    cache._get_or_load("second", "cpu", lambda: models[1])

    pya.pred.predict_age(
        _adata([[1.0]], ["a"]), ["First", "second", "third"], dir="somewhere", verbose=False, clock_cache=cache
    )

    prefetch.assert_called_once_with(["first", "third"], "somewhere")


@pytest.mark.parametrize("clock_names", ["first", ["first", "FIRST"]])
def test_a_single_clock_is_not_prefetched(monkeypatch, clock_names):
    prefetch = Mock()
    monkeypatch.setattr("pyaging.predict._pred.prefetch_clock_weights", prefetch)
    _use_model(monkeypatch, _WeightedClock(["a"], name="first"))

    pya.pred.predict_age(_adata([[1.0]], ["a"]), clock_names, verbose=False)

    prefetch.assert_not_called()


def test_cache_membership_does_not_load_or_reorder():
    cache = pya.pred.ClockCache(maxsize=2)
    cache._get_or_load("first", "cpu", lambda: "one")
    cache._get_or_load("second", "cpu", lambda: "two")

    assert cache._contains("FIRST", "cpu")
    assert not cache._contains("third", "cpu")
    cache._get_or_load("third", "cpu", lambda: "three")
    # "first" stayed least recently used, so it was the one evicted.
    assert not cache._contains("first", "cpu")
