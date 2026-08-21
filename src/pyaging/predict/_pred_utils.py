import os

import anndata
import numpy as np
import torch

try:
    import cupy as cp

    CUPY_AVAILABLE = cp.cuda.is_available()
except Exception:
    CUPY_AVAILABLE = False

import gc

from ..models import pyagingModel
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
    clock_name = clock_name.lower()
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
        message = (
            f"Clock {clock_name} is not available on pyaging. "
            "Please refer to the clock names in the clock glossary table "
            "in the package documentation page: pyaging.readthedocs.io"
        )
        logger.error(message, indent_level=indent_level + 1)
        raise NameError(message) from exc

    # Load the clock from the file
    clock = torch.load(weights_path, weights_only=False)

    # Prepare clock for inference
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


def _align_features_into_obsm(adata, model, source_values, source_features, logger, indent_level: int) -> None:
    """Fill ``obsm["X_{clock}"]`` from ``source_values``, one column per model feature.

    Features the source does not carry take the model's reference value (or 0
    when it has none), and the substitutions are recorded so downstream checks
    can tell the input's own values apart from the pipeline's.
    """

    # Preallocate the data matrix
    adata.obsm[f"X_{model.metadata['clock_name']}"] = (
        cp.empty((adata.n_obs, len(model.features)))
        if CUPY_AVAILABLE
        else np.empty((adata.n_obs, len(model.features)), order="F")
    )

    # Find indices of matching features among the source's own feature names
    feature_indices = {feature: i for i, feature in enumerate(source_features)}
    model_feature_indices = np.array([feature_indices.get(feature, -1) for feature in model.features])

    # Identify missing features
    missing_features_mask = model_feature_indices == -1
    missing_features = np.array(model.features)[missing_features_mask].tolist()

    # Assign values for existing features
    existing_features_mask = ~missing_features_mask
    existing_features_indices = model_feature_indices[existing_features_mask]
    adata.obsm[f"X_{model.metadata['clock_name']}"][:, existing_features_mask] = source_values[
        :, existing_features_indices
    ]

    # Handle missing features
    adata.obsm[f"X_{model.metadata['clock_name']}"][:, missing_features_mask] = (
        np.array(model.reference_values)[missing_features_mask] if model.reference_values is not None else 0
    )

    # Calculate missing features statistics
    num_missing_features = len(missing_features)
    percent_missing = 100 * num_missing_features / len(model.features)

    # Add missing features and percent missing values to the clock
    adata.uns[f"{model.metadata['clock_name']}_percent_na"] = percent_missing
    adata.uns[f"{model.metadata['clock_name']}_missing_features"] = missing_features
    # Columns the input carried; check_feature_ranges judges only these.
    adata.uns[f"{model.metadata['clock_name']}_supplied_features_mask"] = existing_features_mask

    # Raises error if there are no features in the data
    if percent_missing == 100:
        message = (
            f"Every single feature out of {len(model.features)} features "
            f"is missing. Please double check the features in the adata object"
            f" actually contain the clock features such as "
            f"{missing_features[: np.min([3, num_missing_features])]}, etc."
        )
        logger.error(message, indent_level=3)
        raise NameError(message)

    # Log and add missing features if any
    if len(missing_features) > 0:
        logger.warning(
            f"{num_missing_features} out of {len(model.features)} features "
            f"({percent_missing:.2f}%) are missing: {missing_features[: np.min([3, num_missing_features])]}, etc.",
            indent_level=indent_level + 1,
        )
        # If there are reference values provided
        if model.reference_values is not None:
            logger.info(
                f"Using reference feature values for {model.metadata['clock_name']}",
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
            model.metadata.get("data_type"),
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

    # If there is a preprocessing step
    if model.preprocess_name is not None:
        logger.info(
            f"The preprocessing method is {model.preprocess_name}",
            indent_level=indent_level + 1,
        )
    else:
        logger.info("There is no preprocessing necessary", indent_level=indent_level + 1)

    # If there is a postprocessing step
    if model.postprocess_name is not None:
        logger.info(
            f"The postprocessing method is {model.postprocess_name}",
            indent_level=indent_level + 1,
        )
    else:
        logger.info("There is no postprocessing necessary", indent_level=indent_level + 1)

    # Batched prediction over the clock's feature matrix on the model's device
    matrix = adata.obsm[f"X_{model.metadata['clock_name']}"]
    starts = list(range(0, matrix.shape[0], batch_size))
    predictions = []
    with torch.inference_mode():
        for index, start in enumerate(starts):
            batch = torch.as_tensor(matrix[start : start + batch_size], dtype=torch.float64, device=device)
            predictions.append(model(batch))
            if progress_callback is not None:
                progress_callback(index + 1, len(starts))
    # Concatenate all batch predictions
    predictions = torch.cat(predictions)

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
    # Convert from a torch tensor to a flat numpy array
    predicted_ages = predicted_ages.cpu().detach().numpy().flatten()

    # Add predicted ages to adata.obs
    adata.obs[model.metadata["clock_name"]] = predicted_ages

    # Add clock metadata to adata.uns
    adata.uns[f"{model.metadata['clock_name']}_metadata"] = model.metadata


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
    # Delete the model if provided
    if model is not None:
        del model

    # Delete any additional objects passed via kwargs
    for obj in kwargs.values():
        if obj is not None:
            del obj

    # Delete the .pt file from disk if specified
    if clock_name is not None and dir is not None:
        weights_path = os.path.join(dir, f"{clock_name}.pt")
        try:
            if os.path.exists(weights_path):
                os.remove(weights_path)
        except OSError:
            # Silently ignore file deletion errors to avoid disrupting tests
            pass

    # Force garbage collection
    gc.collect()

    # Clear PyTorch CUDA cache
    torch.cuda.empty_cache()
