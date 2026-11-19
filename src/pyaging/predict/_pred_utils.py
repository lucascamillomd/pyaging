import gzip
import json
import os
import re
import warnings
from copy import deepcopy
from functools import lru_cache
from importlib import resources
from numbers import Integral

import anndata
import numpy as np
import torch
from scipy import sparse

try:
    import cupy as cp

    CUPY_AVAILABLE = cp.cuda.is_available()
except Exception:
    CUPY_AVAILABLE = False

import gc

from ..models._base_models import pyagingModel
from ..preprocess._tage import _prepare_tage
from ..utils._feature_ranges import resolve_feature_bounds
from ..utils._hf import PyAgingResourceNotFoundError, download_clock_weights
from ..utils._utils import progress

# Whole-cohort preprocessing a clock can ask ``predict_age`` to run for it, by
# name. A transform takes the raw AnnData and returns a samples x features
# frame; it must not write to ``adata.X``, since the other clocks in the same
# call still read the original matrix.
COHORT_TRANSFORMS = {"tage": _prepare_tage}


@progress("Run the cohort transform")
def apply_cohort_transform(
    adata: anndata.AnnData,
    transform_name: str,
    dir: str,
    logger,
    indent_level: int = 2,
):
    """Run the named cohort transform over the whole input and return its frame."""
    try:
        transform = COHORT_TRANSFORMS[transform_name]
    except KeyError:
        message = (
            f"This clock asks for a cohort transform named {transform_name!r}, which this version of "
            f"pyaging does not provide. Known transforms: {sorted(COHORT_TRANSFORMS)}."
        )
        logger.error(message, indent_level=indent_level + 1)
        raise ValueError(message) from None
    return transform(adata, dir=dir, logger=logger)


def load_clock(
    clock_name: str,
    device: str = "cpu",
    dir: str = "pyaging_data",
    logger=None,
    indent_level: int = 2,
    verbose: bool = True,
) -> "pyagingModel":
    """
    Loads the specified aging clock from Hugging Face and returns its components.

    This function downloads the weights and configuration of a specified aging clock from
    Hugging Face. This allows users to instantiate and use the clock in their analyses.

    Parameters
    ----------
    clock_name : str
        The name of the aging clock to be loaded. This name identifies the clock's weights
        and configuration on Hugging Face. Case-insensitive.

    device : str, optional
        Device to move the clock to, 'cpu' or 'cuda'. Defaults to 'cpu'.

    dir : str, optional
        Retained for backward compatibility. Hugging Face files use its standard cache.

    logger : optional
        Internal pipeline logger. Leave as None when calling directly; the
        progress display handles output.

    indent_level : int, optional
        Indentation level for internal pipeline logging, by default 2.

    verbose : bool, optional
        Whether to show the progress display for direct calls. Defaults to True.

    Returns
    -------
    pyagingModel
        A clock model

    Notes
    -----
    The clock's weights and configuration are stored in a .pt (PyTorch) file on Hugging Face.
    If the requested clock is unavailable, the function raises a ``NameError``.

    The logger is used extensively for progress tracking and information logging, enhancing
    transparency and user experience.

    Examples
    --------
    >>> clock = load_clock("horvath2013")

    """
    clock_name = clock_name.strip().lower()
    if logger is not None:
        return _load_clock_impl(clock_name, device, dir, logger, indent_level)

    # Direct user call: run under the display instead of a plumbed logger.
    # Imported here to avoid a predict <-> logger import cycle.
    from ..logger._live import live_step, quiet_hf_bars

    with (
        quiet_hf_bars(verbose),
        live_step(f"loading {clock_name}", verbose) as (step, pipeline_logger),
    ):
        model = _load_clock_impl(clock_name, device, dir, pipeline_logger, indent_level)
        step.done(f"{clock_name} ready on {device}")
    return model


def _load_clock_impl(clock_name: str, device: str, dir: str, logger, indent_level: int):
    try:
        weights_path = download_clock_weights(clock_name, dir, logger, indent_level=indent_level)
    except PyAgingResourceNotFoundError as exc:
        from pyaging import __version__

        message = (
            f"Clock {clock_name} is not available on pyaging {__version__}. "
            "This clock may require a newer pyaging release; check PyPI and run "
            "`pip install --upgrade pyaging` if a newer version is available. "
            "Please refer to the clock names in the clock glossary table "
            "in the package documentation page: pyaging.readthedocs.io"
        )
        logger.error(message, indent_level=indent_level + 1)
        raise NameError(message) from exc

    # Restore on CPU even when the weights were saved from a CUDA device. The
    # explicit transfer below then respects the caller's selected device.
    clock = torch.load(weights_path, weights_only=False, map_location="cpu")

    clock.to(torch.float64)
    clock.to(device)
    clock.eval()

    return clock


@progress("Check features in adata")
def check_features_in_adata(
    adata: anndata.AnnData,
    model: pyagingModel,
    logger,
    indent_level: int = 2,
) -> None:
    """
    Verifies if all required features are present in an AnnData object and adds missing features.

    This function checks an AnnData object (commonly used in single-cell analysis) to ensure
    that it contains all the necessary features specified in the 'features' list inside the model.
    If any features are missing, they are added to the AnnData object with a default value of 0 or
    with a reference value if given. This is crucial for downstream analyses where the presence of
    all specified features is assumed.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to be checked. It is a commonly used data structure in single-cell
        genomics containing high-dimensional data.

    model : pyagingModel
        The pyagingModel of the aging clock of interest. Must contain defined features.

    logger : Logger
        A logger object used for logging information about the process, such as the number
        of missing features.

    indent_level : int, optional
        The indentation level for the logger, by default 2. It controls the formatting
        of the log messages.

    Returns
    -------
    None
        The AnnData object is updated in place; missing features are added with a default
        value of 0 (or reference value if provided).

    Notes
    -----
    This function is particularly useful in preprocessing steps where the consistency of
    data structure across different datasets is crucial. The function modifies the AnnData
    object if there are missing features and logs detailed information about these modifications.

    The added features are initialized with zeros. This approach, while providing completeness,
    may introduce biases if not accounted for in downstream analyses. If reference values are
    provided, then they are used instead of zeros.

    Alongside the missing-feature bookkeeping the function records a boolean mask of the columns
    that came from the input, under ``.uns["{clock_name}_supplied_features_mask"]``. Substituted
    values are the pipeline's own, so :func:`check_feature_ranges` uses the mask to judge only
    what the user actually supplied.

    A NaN inside a supplied column gets the same per-feature fallback as an absent column, except
    that there is no zero: without a finite reference or packaged CpG median the cell stays NaN.
    Clocks with ``handles_nan = True`` receive their NaN cells untouched. The share of NaN cells
    among the supplied columns is recorded under ``.uns["{clock_name}_percent_nan_values"]`` and
    any NaN is reported as a ``UserWarning``. Samples whose supplied features are all NaN, or
    whose ``age``/``female`` covariate is NaN, are left unfilled and predicted as NaN by
    :func:`predict_ages_with_model`.

    Raises ``ValueError`` when ``adata.X`` is None or a required ``age``/``female`` covariate is
    not among the features, and ``NameError`` when no feature is present; nothing is written to
    ``adata`` in either case.

    Examples
    --------
    >>> updated_adata = check_features_in_adata(adata, bitage, ["gene1", "gene2"], logger)
    >>> updated_adata.var_names
    Index(['gene1', 'gene2', ...], dtype='object')

    """
    _align_features_into_obsm(adata, model, adata.X, adata.var_names, logger, indent_level)


@progress("Build the cohort feature matrix")
def build_cohort_feature_matrix(
    adata: anndata.AnnData,
    model: pyagingModel,
    frame,
    logger,
    indent_level: int = 2,
) -> None:
    """
    Assemble a cohort-relative clock's feature matrix from its transformed frame.

    A clock declaring ``cohort_transform`` is not scored against ``adata.X``: its
    features live in the space the transform produces (for tAge, cohort-centred
    mouse Entrez columns), so alignment reads that frame instead. Everything else
    -- reference-value substitution for features the transform did not yield, the
    missing-feature bookkeeping, and the supplied-features mask that
    :func:`check_feature_ranges` reads -- is identical to
    :func:`check_features_in_adata`, which is why both share one implementation.

    Parameters
    ----------
    adata : anndata.AnnData
        The object the matrix and the bookkeeping are written to. Its ``.X`` is
        only read by the transform, never here.

    model : pyagingModel
        The clock being aligned.

    frame : pandas.DataFrame
        Samples x transformed-features, in the row order of ``adata``.

    logger : Logger
        A logger object used for logging information about the process.

    indent_level : int, optional
        The indentation level for the logger, by default 2.

    Returns
    -------
    None
        The AnnData object is updated in place.

    Raises
    ------
    ValueError
        If the frame's rows are not the cohort's samples. The matrix is filled
        positionally, so a frame holding different samples -- or the same ones
        under duplicated labels -- would attach one sample's expression to
        another's predicted age, which is worse than not predicting at all. A
        frame that merely orders the same samples differently is reordered to
        match rather than rejected.
    """
    if not frame.index.equals(adata.obs_names):
        if frame.index.has_duplicates or set(frame.index) != set(adata.obs_names):
            message = (
                f"The cohort transform returned {len(frame.index)} rows that are not this cohort's "
                f"{adata.n_obs} samples, so its values cannot be matched to them."
            )
            logger.error(message, indent_level=indent_level + 1)
            raise ValueError(message)
        frame = frame.reindex(adata.obs_names)

    _align_features_into_obsm(adata, model, frame.to_numpy(), frame.columns, logger, indent_level)


@lru_cache(maxsize=1)
def _load_dnam_feature_medians() -> dict[str, float]:
    source = resources.files("pyaging.data").joinpath("dnam_feature_medians.json.gz")
    with source.open("rb") as compressed, gzip.GzipFile(fileobj=compressed) as table:
        return json.load(table)["features"]


# Per-sample covariates some clocks read from their feature matrix. A reference
# or zero in their place would give every sample the same age or sex, so they
# are never substituted: absent ones are an error, NaN ones void the sample.
_REQUIRED_COVARIATES = frozenset({"age", "female"})


def _warn(logger, message: str, indent_level: int) -> None:
    """Report on the display and on Python's warnings channel, which stays
    visible at ``verbose=False`` where the display renders nothing."""
    logger.warning(message, indent_level=indent_level)
    warnings.warn(message, UserWarning, stacklevel=3)


def _uses_dnam_medians(model, features) -> bool:
    """Whether packaged human CpG beta medians may stand in for these features."""
    return (
        model.metadata.get("species") == "Homo sapiens"
        and model.metadata.get("data_type") == "DNA methylation"
        and getattr(model, "use_dnam_median_fallback", True) is not False
        and any(re.fullmatch(r"cg[0-9]{8}", feature) for feature in features)
    )


def _nan_cell_fallbacks(model, features: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Per-feature values for NaN cells: a finite reference, else a packaged
    CpG median for eligible clocks, else NaN. Unlike absent columns there is no
    zero, so a clock that cannot use a NaN still shows it in its prediction."""
    values = np.full(len(features), np.nan)
    if model.reference_values is not None:
        values = np.asarray(model.reference_values, dtype=float)[positions]
        values[~np.isfinite(values)] = np.nan
    unresolved = np.isnan(values)
    if unresolved.any() and _uses_dnam_medians(model, features[unresolved]):
        medians = _load_dnam_feature_medians()
        values[unresolved] = [medians.get(feature, np.nan) for feature in features[unresolved]]
    return values


def _nan_column_positions(values: np.ndarray) -> np.ndarray:
    """Positions of the columns of ``values`` that hold at least one NaN.

    Column sums propagate NaN without a boolean copy of the whole matrix, so a
    complete input costs one pass; only flagged columns are then inspected,
    which also clears columns whose sum is NaN only because of ``inf - inf``.
    """
    if not np.issubdtype(values.dtype, np.floating):
        return np.empty(0, dtype=np.intp)
    with np.errstate(invalid="ignore", over="ignore"):
        candidates = np.flatnonzero(np.isnan(values.sum(axis=0)))
    return candidates[np.isnan(values[:, candidates]).any(axis=0)]


def _missing_covariates_message(adata, clock_name: str, covariates: list) -> str:
    message = (
        f"Clock '{clock_name}' requires the per-sample covariate(s) {covariates}, which are not among the "
        "input features. Supply them as numeric feature columns in adata.var_names (age in years, female as "
        "1 or 0) rather than through metadata_cols; they cannot be replaced by a reference value."
    )
    in_obs = [covariate for covariate in covariates if covariate in adata.obs.columns]
    if len(in_obs) == 1:
        message += f" adata.obs has a column named {in_obs[0]!r}; move it into the features."
    elif in_obs:
        message += (
            f" adata.obs has columns named {', '.join(repr(name) for name in in_obs)}; move them into the features."
        )
    return message


def _align_features_into_obsm(adata, model, source_values, source_features, logger, indent_level: int) -> None:
    """Fill ``obsm["X_{clock}"]`` from ``source_values``, one column per model feature.

    Absent columns use model references, then packaged human CpG beta medians
    for eligible clocks, then zero. NaN cells in supplied columns use the same
    fallbacks except zero, unless the clock handles NaN itself; samples with no
    usable data are left unfilled for :func:`predict_ages_with_model` to void.
    Everything is validated before ``adata`` is written to.
    """
    clock_name = model.metadata["clock_name"]

    if source_values is None:
        raise ValueError("adata.X is empty (None); store the feature matrix in adata.X before prediction.")
    if source_features.has_duplicates:
        raise ValueError("Feature names must be unique; aggregate duplicate features before prediction.")
    if not len(model.features):
        raise ValueError("The clock must define at least one feature.")

    # Find indices of matching features among the source's own feature names
    # pandas retains the index's lookup engine, so another clock can reuse it
    # instead of rebuilding a Python dictionary over every input feature.
    model_feature_indices = source_features.get_indexer(model.features)

    missing_features_mask = model_feature_indices == -1
    feature_names = np.array(model.features)
    missing_features = feature_names[missing_features_mask].tolist()
    num_missing_features = len(missing_features)
    percent_missing = 100 * num_missing_features / len(model.features)

    if percent_missing == 100:
        message = (
            f"Every single feature out of {len(model.features)} features "
            f"is missing. Please double check the features in the adata object"
            f" actually contain the clock features such as "
            f"{missing_features[: np.min([3, num_missing_features])]}, etc."
        )
        logger.error(message, indent_level=3)
        raise NameError(message)

    missing_covariates = [feature for feature in missing_features if feature in _REQUIRED_COVARIATES]
    if missing_covariates:
        message = _missing_covariates_message(adata, clock_name, missing_covariates)
        logger.error(message, indent_level=indent_level + 1)
        raise ValueError(message)

    # Keep the whole cohort on the host: only inference batches belong on the
    # model's device, otherwise batch_size cannot limit the GPU memory needed.
    matrix = np.empty((adata.n_obs, len(model.features)), order="F")

    existing_features_mask = ~missing_features_mask
    # Read the source in increasing column order, which backed (on-disk)
    # matrices require, and write each column straight to its model position.
    supplied_positions = np.flatnonzero(existing_features_mask)
    read_order = np.argsort(model_feature_indices[supplied_positions], kind="stable")
    supplied_positions = supplied_positions[read_order]
    supplied_values = source_values[:, model_feature_indices[supplied_positions]]
    if sparse.issparse(supplied_values):
        supplied_values = supplied_values.toarray()
    elif CUPY_AVAILABLE and isinstance(supplied_values, cp.ndarray):
        supplied_values = cp.asnumpy(supplied_values)
    supplied_values = np.asarray(supplied_values)
    matrix[:, supplied_positions] = supplied_values

    nan_columns = _nan_column_positions(supplied_values)
    nan_cells = np.isnan(supplied_values[:, nan_columns])
    del supplied_values
    nan_positions = supplied_positions[nan_columns]
    nan_features = feature_names[nan_positions]
    no_data_rows = np.zeros(adata.n_obs, dtype=bool)
    if nan_columns.size == supplied_positions.size:
        no_data_rows |= nan_cells.all(axis=1)
    no_data_rows |= nan_cells[:, np.isin(nan_features, list(_REQUIRED_COVARIATES))].any(axis=1)
    filled_count = 0
    if nan_columns.size and getattr(model, "handles_nan", False) is not True:
        fallbacks = _nan_cell_fallbacks(model, nan_features, nan_positions)
        fill = nan_cells & ~no_data_rows[:, None] & ~np.isnan(fallbacks)
        filled_count = int(fill.sum())
        if filled_count:
            matrix[:, nan_positions] = np.where(fill, fallbacks, matrix[:, nan_positions])

    median_count = 0
    if missing_features:
        fill_values = 0
        if model.reference_values is not None:
            fill_values = np.asarray(model.reference_values)[missing_features_mask]
        elif _uses_dnam_medians(model, missing_features):
            medians = _load_dnam_feature_medians()
            fill_values = [medians.get(feature, 0.0) for feature in missing_features]
            median_count = sum(feature in medians for feature in missing_features)
        matrix[:, missing_features_mask] = fill_values

    nan_count = int(nan_cells.sum())
    adata.obsm[f"X_{clock_name}"] = matrix
    adata.uns[f"{clock_name}_percent_na"] = percent_missing
    adata.uns[f"{clock_name}_missing_features"] = missing_features
    # Columns the input carried; check_feature_ranges judges only these.
    adata.uns[f"{clock_name}_supplied_features_mask"] = existing_features_mask
    adata.uns[f"{clock_name}_percent_nan_values"] = (
        100 * nan_count / (adata.n_obs * supplied_positions.size) if nan_count else 0.0
    )

    # Samples with no usable data are reported, by name, when they are predicted.
    reported = nan_cells[~no_data_rows]
    reported_count = int(reported.sum())
    if reported_count:
        message = (
            f"{clock_name}: {reported_count} NaN value(s) in {int(reported.any(axis=1).sum())} of {adata.n_obs} "
            f"samples and {int(reported.any(axis=0).sum())} of {supplied_positions.size} supplied features."
        )
        if getattr(model, "handles_nan", False) is True:
            message += " They were left as NaN for the clock, which handles missing values itself."
        else:
            outcomes = []
            if filled_count:
                outcomes.append(
                    f"{filled_count} were filled with the fallback used for absent features (a finite reference "
                    "value, else a packaged CpG median)"
                )
            if reported_count - filled_count:
                outcomes.append(f"{reported_count - filled_count} had no such fallback and were left as NaN")
            message += f" {'; '.join(outcomes)}."
        _warn(logger, message, indent_level + 1)

    if len(missing_features) > 0:
        logger.warning(
            f"{num_missing_features} out of {len(model.features)} features "
            f"({percent_missing:.2f}%) are missing: {missing_features[: np.min([3, num_missing_features])]}, etc.",
            indent_level=indent_level + 1,
        )
        if model.reference_values is not None:
            logger.info(
                f"Using reference feature values for {clock_name}",
                indent_level=indent_level + 1,
            )
        elif median_count:
            logger.info(
                f"Using packaged CpG medians for {median_count} missing features "
                f"and zero for {num_missing_features - median_count}",
                indent_level=indent_level + 1,
            )
        else:
            logger.info(
                "Filling missing features entirely with 0",
                indent_level=indent_level + 1,
            )
    else:
        logger.info(
            "All features are present in adata.var_names.",
            indent_level=indent_level + 1,
        )


_MAX_REPORTED_FEATURES = 5


def _describe_range(low: float, high: float) -> str:
    """Phrase a violated range, keeping half-bounded ranges readable."""
    if np.isinf(low):
        return f"above {high:g}"
    if np.isinf(high):
        return f"below {low:g}"
    return f"outside [{low:g}, {high:g}]"


def _supplied_columns(adata: anndata.AnnData, model: pyagingModel, n_features: int) -> np.ndarray:
    """Return the indices of the feature columns that came from the input data.

    :func:`check_features_in_adata` substitutes a reference value, or 0, for every
    feature the input did not carry, so those columns say nothing about the user's
    units. It leaves behind a mask of the columns it did not substitute, which this
    reads. A matrix assembled some other way has no mask, and then every column
    counts as supplied.
    """
    mask = adata.uns.get(f"{model.metadata['clock_name']}_supplied_features_mask")
    if mask is None:
        return np.arange(n_features)
    return np.flatnonzero(np.asarray(mask, dtype=bool))


# Columns compared per pass. The comparison is vectorised, but a clock with 453,152
# features against a large cohort would otherwise build a boolean array the size of the
# whole matrix, so the scan walks the columns in blocks of bounded width.
_SCAN_BLOCK_COLUMNS = 4096


def _find_offenders(matrix: np.ndarray, columns: np.ndarray, low: np.ndarray, high: np.ndarray) -> list:
    """Return ``(column index, out-of-range count)`` for each offending column.

    NaN compares false against both bounds, so missing values drop out of the
    count without being tested for separately.
    """
    offenders = []
    for start in range(0, columns.size, _SCAN_BLOCK_COLUMNS):
        block = columns[start : start + _SCAN_BLOCK_COLUMNS]
        values = matrix[:, block]
        counts = ((values < low[block]) | (values > high[block])).sum(axis=0)
        offenders.extend((int(block[position]), int(counts[position])) for position in np.flatnonzero(counts))
    return offenders


def _samples_without_data(adata: anndata.AnnData, model: pyagingModel, matrix) -> np.ndarray:
    """Mask of the samples whose supplied features are all NaN, or whose
    supplied age/female covariate is NaN.

    Alignment leaves exactly these cells unfilled, so they can be recovered
    from the matrix on every path that calls :func:`predict_ages_with_model`.
    Candidate rows shrink block by block, so a complete cohort costs one
    column read.
    """
    matrix = np.asarray(matrix)
    columns = _supplied_columns(adata, model, matrix.shape[1])
    no_data = np.zeros(matrix.shape[0], dtype=bool)
    if not columns.size:
        return no_data
    rows = np.flatnonzero(np.isnan(matrix[:, columns[0]]))
    for start in range(1, columns.size, _SCAN_BLOCK_COLUMNS):
        if not rows.size:
            break
        block = columns[start : start + _SCAN_BLOCK_COLUMNS]
        rows = rows[np.isnan(matrix[np.ix_(rows, block)]).all(axis=1)]
    no_data[rows] = True
    # A matrix assembled outside alignment may come from a model without features.
    features = getattr(model, "features", None)
    features = [] if features is None else list(features)
    covariates = [features.index(name) for name in sorted(_REQUIRED_COVARIATES) if name in features]
    covariates = [column for column in covariates if column in columns]
    if covariates:
        no_data |= np.isnan(matrix[:, covariates]).any(axis=1)
    return no_data


@progress("Check feature ranges")
def check_feature_ranges(
    adata: anndata.AnnData,
    model: pyagingModel,
    logger,
    indent_level: int = 2,
) -> None:
    """
    Warn when input values fall outside a feature's plausible range.

    Ranges come from the package-wide registry (per-feature entry, else the
    modality default for the clock's ``data_type``). Out-of-range values usually
    mean wrong units or swapped columns, so this warns and never blocks: the
    data is not inspected for clinical abnormality and is never modified.

    Only the features the input actually carried are judged. Everything else in
    the matrix was put there by :func:`check_features_in_adata`, and several
    clocks use an out-of-range sentinel such as ``-1`` as their reference value,
    so including those columns would report the pipeline's own substitutions as
    if they were the user's data.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object whose ``.obsm["X_{clock_name}"]`` matrix holds the
        clock's feature values, one column per model feature.

    model : pyagingModel
        The pyagingModel of the aging clock of interest. Clocks saved before the
        ``feature_units`` attribute existed are supported.

    logger : Logger
        A logger object used to report the out-of-range features.

    indent_level : int, optional
        The indentation level for the logger, by default 2.

    Returns
    -------
    None
        Nothing is returned and nothing is modified; the findings are logged.

    Notes
    -----
    NaN values are ignored: missing features are handled by
    :func:`check_features_in_adata`, and should not also be reported here. The
    reported percentage is therefore the share of non-NaN values that fall
    outside the range, and the feature counts are out of the supplied features
    rather than out of all of the clock's features.
    """
    try:
        units, low, high = resolve_feature_bounds(
            model.features,
            getattr(model, "feature_range_data_type", None) or model.metadata.get("data_type"),
            getattr(model, "feature_units", None),
        )
    except Exception as exc:
        # Deliberately broad: one clock whose stored feature_units disagree with its
        # features must not abort predict_age for every user.
        logger.warning(f"Could not resolve feature ranges: {exc}", indent_level=indent_level + 1)
        return

    matrix = adata.obsm[f"X_{model.metadata['clock_name']}"]
    if CUPY_AVAILABLE and isinstance(matrix, cp.ndarray):
        matrix = cp.asnumpy(matrix)
    matrix = np.asarray(matrix, dtype=float)

    columns = _supplied_columns(adata, model, len(units))
    offenders = _find_offenders(matrix, columns, low, high)

    if not offenders:
        logger.info("All feature values are within their expected ranges.", indent_level=indent_level + 1)
        return

    truncated = f" Showing the first {_MAX_REPORTED_FEATURES}." if len(offenders) > _MAX_REPORTED_FEATURES else ""
    logger.warning(
        f"{len(offenders)} of {len(columns)} supplied features have values outside their expected range. "
        f"This usually means the data is in different units than the clock expects.{truncated}",
        indent_level=indent_level + 1,
    )
    for index, count in offenders[:_MAX_REPORTED_FEATURES]:
        observed = matrix[:, index]
        observed = observed[~np.isnan(observed)]
        unit = f" {units[index]}" if units[index] else ""
        logger.warning(
            f"{model.features[index]}: {100 * count / observed.size:.2f}% of values "
            f"{_describe_range(low[index], high[index])}{unit} "
            f"(observed {observed.min():g} to {observed.max():g})",
            indent_level=indent_level + 2,
        )


def _validate_batch_size(batch_size: int) -> None:
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")


@progress("Predict ages with model")
def predict_ages_with_model(
    adata: anndata.AnnData,
    model: pyagingModel,
    device: str,
    batch_size: int,
    logger,
    indent_level: int = 2,
    progress_callback=None,
) -> torch.Tensor:
    """
    Predict biological ages using a trained model and input data.

    This function takes a machine learning model and input data, and returns predictions made by the model.
    It's primarily used for estimating biological ages based on various biological markers. The function
    assumes that the model is already trained. A dataloader is used because of possible memory constraints
    for large datasets.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing the dataset. Its `.X` attribute is expected to be a matrix where rows
        correspond to samples and columns correspond to features.

    model : pyagingModel
        The pyagingModel of the aging clock of interest.

    device : str
        Device to move AnnData to during inference. Eithe 'cpu' or 'cuda'.

    batch_size : int
        Number of samples per prediction batch.

    logger : Logger
        A logger object for logging the progress or any relevant information during the prediction process.

    indent_level : int, optional
        The indentation level for logging messages, by default 2.

    Returns
    -------
    predictions : torch.Tensor
        An array of predicted ages or biological markers, as returned by the model.

    Notes
    -----
    Ensure that the data is preprocessed (e.g., scaled, normalized) as required by the model before
    passing it to this function. The model should be in evaluation mode if it's a type that has different
    behavior during training and inference (e.g., PyTorch models).

    The exact nature of the predictions (e.g., age, biological markers) depends on the model being used.

    Examples
    --------
    >>> model = load_pretrained_model()
    >>> predictions = predict_ages_with_model(model, "cpu", logger)
    >>> print(predictions[:5])
    [34.5, 29.3, 47.8, 50.1, 42.6]

    """

    _validate_batch_size(batch_size)
    if adata.n_obs == 0:
        raise ValueError("Prediction requires at least one sample.")

    if model.preprocess_name is not None:
        logger.info(
            f"The preprocessing method is {model.preprocess_name}",
            indent_level=indent_level + 1,
        )
    else:
        logger.info("There is no preprocessing necessary", indent_level=indent_level + 1)

    if model.postprocess_name is not None:
        logger.info(
            f"The postprocessing method is {model.postprocess_name}",
            indent_level=indent_level + 1,
        )
    else:
        logger.info("There is no postprocessing necessary", indent_level=indent_level + 1)

    matrix = adata.obsm[f"X_{model.metadata['clock_name']}"]
    # A sample with no usable data would otherwise be predicted from whatever
    # stands in for it, so its prediction is NaN. A per-sample clock never sees
    # it, so it cannot reject a sample it has nothing to say about. Clocks with
    # cohort steps (PAOPAC's scaling) still receive it, as the author's
    # implementation does: dropping it would shift every other prediction.
    no_data = _samples_without_data(adata, model, matrix)
    keeps_cohort = any(
        getattr(model, name, None) is not None for name in ("prepare_cohort_context", "postprocess_cohort")
    )
    if no_data.any():
        names = adata.obs_names[no_data].tolist()
        shown = ", ".join(repr(name) for name in names[:_MAX_REPORTED_FEATURES])
        more = ", ..." if len(names) > _MAX_REPORTED_FEATURES else ""
        _warn(
            logger,
            f"{model.metadata['clock_name']}: {len(names)} {'sample has' if len(names) == 1 else 'samples have'} "
            f"no usable data (every supplied feature NaN, or a NaN age/female covariate) and "
            f"{'is' if len(names) == 1 else 'are'} predicted as NaN: [{shown}{more}].",
            indent_level + 1,
        )
        if no_data.all():
            return torch.full((adata.n_obs, 1), float("nan"), dtype=torch.float64, device=device)
        if not keeps_cohort:
            matrix = matrix[~no_data]
    starts = range(0, matrix.shape[0], batch_size)
    predictions = []
    with torch.inference_mode():
        # Rank clocks fill NaNs with a cohort-wide median. Compute that scalar
        # once on the host, rather than deriving a different value per batch.
        # Keep it local to this call so cached models never retain dataset state.
        prepare_context = getattr(model, "prepare_cohort_context", None)
        if prepare_context is not None:
            cohort_context = prepare_context(torch.as_tensor(matrix, dtype=torch.float64, device="cpu")).to(device)
        for index, start in enumerate(starts):
            # Some clocks preprocess in place. Own each batch so those writes
            # cannot change a retained obsm matrix or a later prediction.
            batch = torch.as_tensor(matrix[start : start + batch_size], dtype=torch.float64).to(device, copy=True)
            prediction = (
                model(batch) if prepare_context is None else model.predict_with_cohort_context(batch, cohort_context)
            )
            predictions.append(prediction)
            if progress_callback is not None:
                progress_callback(index + 1, len(starts))
    predictions = torch.cat(predictions)
    postprocess_cohort = getattr(model, "postprocess_cohort", None)
    if postprocess_cohort is not None:
        # Cohort corrections (e.g. PAOPAC's LOWESS age-bias correction) must
        # see every sample together, independently of the inference batch size.
        with torch.inference_mode():
            predictions = postprocess_cohort(predictions, torch.as_tensor(matrix, dtype=torch.float64, device="cpu"))

    if no_data.any():
        mask = torch.as_tensor(no_data, device=predictions.device)
        # Predictions are inference tensors, which only accept in-place writes in this mode.
        with torch.inference_mode():
            if not keeps_cohort:
                complete = predictions.new_full((adata.n_obs, *predictions.shape[1:]), float("nan"))
                complete[~mask] = predictions
                predictions = complete
            else:
                predictions[mask] = float("nan")
    return predictions


@progress("Add predicted ages and clock metadata to adata")
def add_pred_ages_and_clock_metadata_adata(
    adata: anndata.AnnData,
    model: pyagingModel,
    predicted_ages: torch.tensor,
    dir: str,
    logger,
    indent_level: int = 2,
) -> None:
    """
    Add predicted ages to an AnnData object as a new column in the observation (obs) attribute. Also adds
    the specific clock metadata to the `uns` attribute of an AnnData object.

    This function appends the predicted ages, obtained from a biological aging clock or similar model, to
    the AnnData object's `obs` attribute. The predicted ages are added as a new column, named after the
    clock used to generate these predictions.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to which the predicted ages will be added. It's a data structure for handling
        large-scale biological data, like gene expression matrices, commonly used in bioinformatics.

    model : pyagingModel
        The aging clock from which to get the metadata.

    predicted_ages : torch.tensor
        A torch tensor of predicted ages corresponding to the samples in the AnnData object. The length
        of this array should match the number of samples in `adata`.

    dir: str
        Retained for backward compatibility. Hugging Face files use its standard cache.

    logger : Logger
        A logger object for logging the progress or relevant information during the operation.

    indent_level : int, optional
        The indentation level for logging messages, by default 2.

    Returns
    -------
    None
        This function modifies the AnnData object in-place and does not return any value.

    Notes
    -----
    It is essential to ensure that the length of `predicted_ages` matches the number of samples in the
    `adata` object. Mismatch in lengths will lead to errors or misaligned data.

    This function is part of a pipeline that integrates aging clock predictions with the
    standard data structures used in bioinformatics, facilitating downstream analyses like visualization
    or statistical testing.

    Examples
    --------
    >>> adata = anndata.AnnData(np.random.rand(5, 10))
    >>> predicted_ages = [25, 30, 35, 40, 45]
    >>> add_pred_ages_adata(adata, predicted_ages_tensor, clock, "pyaging_data", logger)
    >>> adata.obs["horvath2013"]
    0    25
    1    30
    2    35
    3    40
    4    45
    Name: horvath2013, dtype: int64
    >>> adata.uns["horvath2013_metadata"]
    {'species': 'Homo sapiens', 'data_type': 'methylation', 'citation': 'Horvath, S. (2013)'}

    """
    predicted_ages = predicted_ages.cpu().detach().numpy().flatten()

    adata.obs[model.metadata["clock_name"]] = predicted_ages

    adata.uns[f"{model.metadata['clock_name']}_metadata"] = deepcopy(model.metadata)


def set_torch_device(logger=None, indent_level: int = 1) -> torch.device:
    """
    Set and return the PyTorch device based on the availability of CUDA.

    This function checks if CUDA is available in the system and accordingly sets the PyTorch device to
    either 'cuda' or 'cpu'. If CUDA is available, it utilizes GPU acceleration for PyTorch operations,
    significantly enhancing computation speed for large datasets. The chosen device is logged for
    user reference.

    Parameters
    ----------
    logger : Logger
        A logger object for logging the selected device.

    indent_level : int, optional
        The indentation level for logging messages, by default 1.

    Returns
    -------
    torch.device
        The PyTorch device object set to 'cuda' if CUDA is available, or 'cpu' otherwise.

    Notes
    -----
    The function automatically detects the availability of CUDA and makes a decision without user input.
    This makes it convenient for deploying code on different machines without the need for manual
    configuration.

    It is important to use the returned device for all PyTorch operations to ensure that they are
    executed on the correct hardware (CPU or GPU).

    Examples
    --------
    >>> logger = pyaging.logger.LoggerManager.gen_logger("example")
    >>> device = set_torch_device(logger)
    >>> print(device)
    device(type='cuda')  # or device(type='cpu') if CUDA is not available

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if logger is not None:
        logger.info(f"Using device: {device}", indent_level=2)
    return device


def cleanup_clock_memory(model=None, clock_name=None, dir=None, **kwargs) -> None:
    """
    Explicitly clean up memory and disk space from loaded clock models.

    This function performs aggressive memory and disk cleanup to prevent
    out-of-memory and out-of-disk-space issues during testing or when processing
    multiple clocks sequentially. It deletes specified objects, removes downloaded
    .pt files, and forces garbage collection.

    Parameters
    ----------
    model : pyagingModel, optional
        The loaded clock model to delete from memory.
    clock_name : str, optional
        The name of the clock whose .pt file should be deleted from disk.
    dir : str, optional
        The directory containing the .pt file to delete. Required if clock_name is provided.
    **kwargs : dict
        Additional objects to delete from memory. Each key-value pair
        represents an object name and the object itself to be deleted.

    Notes
    -----
    This function is particularly useful during testing when multiple clocks
    are loaded sequentially, as it prevents memory accumulation and disk space
    consumption that can lead to "No space left on device" errors in CI environments.

    The function performs the following cleanup steps:
    1. Deletes the provided model object if given
    2. Deletes any additional objects passed via kwargs
    3. Removes the downloaded .pt file from disk if clock_name and dir are provided
    4. Forces Python garbage collection
    5. Clears PyTorch CUDA cache if available

    Examples
    --------
    >>> model = load_clock("horvath2013", "cpu", "pyaging_data", logger)
    >>> # ... use model ...
    >>> cleanup_clock_memory(model=model, clock_name="horvath2013", dir="pyaging_data")
    """
    if model is not None:
        del model

    for obj in kwargs.values():
        if obj is not None:
            del obj

    if clock_name is not None and dir is not None:
        weights_path = os.path.join(dir, f"{clock_name}.pt")
        try:
            if os.path.exists(weights_path):
                os.remove(weights_path)
        except OSError:
            # Silently ignore file deletion errors to avoid disrupting tests
            pass

    gc.collect()

    torch.cuda.empty_cache()
