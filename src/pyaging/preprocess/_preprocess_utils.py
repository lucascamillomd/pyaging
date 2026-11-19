import warnings

import anndata
import numpy as np
import pandas as pd

from ..utils._hf import download_hf_file
from ..utils._utils import progress

# Upper bound on the elements of each float64 temporary in the KNN distance computation.
_KNN_BLOCK_ELEMENTS = 1 << 21


def _nan_euclidean_ranking(X: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """
    Squared nan-euclidean distances from ``X[rows]`` to every row of ``X``, up to a constant.

    Each entry is the sum of squared differences over the features both samples observe,
    divided by the number of such features (NaN when there are none). sklearn's
    ``nan_euclidean_distances`` additionally multiplies by the feature count and takes the
    square root; neither changes the neighbour ranking, which is all that uniform KNN needs.
    Products run in float64 through BLAS and are accumulated over feature blocks so that the
    temporaries stay small whatever the number of features.
    """
    n, p = X.shape
    total = np.zeros((rows.size, n))
    shared = np.zeros((rows.size, n))
    step = max(1, _KNN_BLOCK_ELEMENTS // max(n, 1))
    for start in range(0, p, step):
        block = X[:, start : start + step]
        observed = ~np.isnan(block)
        values = np.where(observed, block, 0.0).astype(np.float64, copy=False)
        squares = values * values
        present = observed.astype(np.float64)
        # Zero-filled values make each product count only features observed by both samples.
        total += squares[rows] @ present.T
        total += present[rows] @ squares.T
        total -= 2.0 * (values[rows] @ values.T)
        shared += present[rows] @ present.T
    np.maximum(total, 0.0, out=total)
    with np.errstate(divide="ignore", invalid="ignore"):
        total /= shared
    total[shared == 0] = np.nan
    return total


def _knn_donor_sums(X, missing, donors, columns, n_neighbors):
    """Sum and count, per column, of the first ``n_neighbors`` donors observing it."""
    observed = ~missing[np.ix_(donors, columns)]
    take = observed & (np.cumsum(observed, axis=0) <= n_neighbors)
    sums = np.where(take, X[np.ix_(donors, columns)], 0.0).sum(axis=0, dtype=np.float64)
    return sums, take.sum(axis=0)


def _knn_nearest_donor_sums(X, missing, donors, columns, n_neighbors):
    """
    Like ``_knn_donor_sums``, visiting donors nearest first and stopping once every column
    has ``n_neighbors`` of them, since most columns need only the first few donors. Returns
    the positions of the columns still short of donors as well.
    """
    sums = np.zeros(columns.size)
    counts = np.zeros(columns.size, dtype=np.intp)
    short = np.arange(columns.size)
    for donor in donors:
        observed = short[~missing[donor, columns[short]]]
        sums[observed] += X[donor, columns[observed]]
        counts[observed] += 1
        short = short[counts[short] < n_neighbors]
        if not short.size:
            break
    return sums, counts, short


def _knn_impute(X: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
    """
    Return a copy of ``X`` imputed like ``KNNImputer(keep_empty_features=True).fit_transform``.

    Uniform weights and the nan-euclidean metric, as in sklearn's defaults. For each missing
    entry, the value is the mean over the ``n_neighbors`` nearest samples (at a defined
    distance) that observe the column, or the column mean when no such sample exists; columns
    with no observed value become 0. Donor values always come from the input, never from
    earlier imputations. Results match sklearn to floating-point rounding, except that exactly
    tied distances are broken by the lower sample index, whereas sklearn's ``argpartition``
    breaks them arbitrarily: the two can differ only when tied donors straddle the
    ``n_neighbors``-th place and disagree on the value. float32 input is averaged in float64
    and returned as float32.
    """
    n = X.shape[0]
    missing = np.isnan(X)
    valid = ~missing.all(axis=0)
    receivers = np.flatnonzero((missing & valid).any(axis=1))

    # The nearest few donors settle almost every column; the rest search all donors at once.
    window = 8 * n_neighbors
    # Distance rows per receiver block are capped like the feature blocks above.
    block = max(1, (4 * _KNN_BLOCK_ELEMENTS) // max(n, 1))
    column_means = {}
    result = X.copy(order="K")
    for start in range(0, receivers.size, block):
        rows = receivers[start : start + block]
        distances = _nan_euclidean_ranking(X, rows)
        for row, distance in zip(rows, distances, strict=True):
            columns = np.flatnonzero(missing[row] & valid)
            # Stable sort puts NaN distances last; donors at NaN distance carry no weight.
            donors = np.argsort(distance, kind="stable")[: np.count_nonzero(~np.isnan(distance))]
            sums, counts, short = _knn_nearest_donor_sums(X, missing, donors[:window], columns, n_neighbors)
            if short.size and donors.size > window:
                sums[short], counts[short] = _knn_donor_sums(X, missing, donors, columns[short], n_neighbors)
            with np.errstate(divide="ignore", invalid="ignore"):
                values = sums / counts
            for index in np.flatnonzero(counts == 0):
                column = columns[index]
                if column not in column_means:
                    column_means[column] = X[~missing[:, column], column].mean(dtype=np.float64)
                values[index] = column_means[column]
            result[row, columns] = values

    result[:, ~valid] = 0
    return result


def _fit_imputer(X: np.ndarray, strategy: str) -> np.ndarray:
    if strategy == "knn":
        return _knn_impute(X)
    # Imported lazily: sklearn adds about half a second to `import pyaging`.
    from sklearn.impute import SimpleImputer

    fill_value = {"fill_value": 0} if strategy == "constant" else {}
    return SimpleImputer(strategy=strategy, keep_empty_features=True, **fill_value).fit_transform(X)


@progress("Impute missing values")
def impute_missing_values(adata: anndata.AnnData, strategy: str, logger, indent_level: int = 1) -> None:
    """
    Imputes missing values in a given adata object using a specified strategy.

    This function handles missing data in by applying various imputation strategies.
    It checks the .X in the adata object for missing values and applies the chosen imputation
    method, which can be mean, median, constant, or K-nearest neighbors (KNN). The function is
    useful in preprocessing steps for datasets where missing data could affect subsequent analyses.
    It also records the fraction of missing values for each feature in ``adata.var["percent_na"]``
    and, when values were imputed, stores a copy of the result in ``adata.layers["X_imputed"]``.

    Parameters
    ----------
    adata : anndata.AnnData
        An adata object containing .X with potential missing values.

    strategy : str
        The imputation strategy to apply. Valid options are 'mean', 'median', 'constant', and 'knn'.

    logger : Logger
        A logging object for tracking the progress and outcomes of the function.

    indent_level : int, optional
        The level of indentation for the logger, with 1 being the default.

    Raises
    ------
    ValueError
        If an invalid imputation strategy is specified.

    Notes
    -----
    The 'constant' strategy fills missing values with 0 by default. The 'knn' strategy uses
    the K-nearest neighbors algorithm to estimate missing values based on similar samples, with
    the defaults of scikit-learn's ``KNNImputer`` (5 neighbors, uniform weights).
    This function is particularly useful in datasets where missing values are common, such as
    in biological or medical data.

    Samples with no observed feature are not imputed: they stay NaN in ``adata.X`` and
    ``adata.layers["X_imputed"]``, do not inform the imputation of other samples, and are
    reported in a warning because they will receive NaN predictions.

    The function ensures that no imputation is performed if there are no missing values in the
    dataset, thus preserving the original data integrity.

    Examples
    --------
    >>> imputed_adata = impute_missing_values(adata, "mean")
    # Imputes missing values using the mean of each column.

    """

    if strategy not in ("mean", "median", "constant", "knn"):
        raise ValueError(f"Invalid imputer strategy: {strategy}")

    # Empty cohorts have no observed missing values.
    missing_counts = np.isnan(adata.X).sum(axis=0)
    adata.var["percent_na"] = missing_counts / adata.n_obs if adata.n_obs else np.zeros(adata.n_vars)

    # Check for missing values
    if adata.var["percent_na"].sum() == 0:
        logger.info("No missing values found. No imputation necessary", indent_level=2)
    else:
        # A sample without a single observed feature has nothing to impute from: filling it
        # would fabricate a typical-looking sample, so it stays NaN and is left out of the fit.
        empty = np.isnan(adata.X).all(axis=1)
        if empty.any():
            names = adata.obs_names[empty]
            shown = ", ".join(repr(name) for name in names[:5]) + (", ..." if len(names) > 5 else "")
            plural = "s have" if len(names) != 1 else " has"
            warnings.warn(
                f"{len(names)} sample{plural} no observed features and will not be imputed: {shown}. "
                "They will receive NaN predictions.",
                UserWarning,
                stacklevel=3,
            )

        logger.info(f"Imputing missing values using {strategy} strategy", indent_level=2)
        if not empty.any():
            adata.X = _fit_imputer(adata.X, strategy)
        else:
            imputed = adata.X.copy(order="K")
            if not empty.all():
                imputed[~empty] = _fit_imputer(adata.X[~empty], strategy)
            adata.X = imputed
        adata.layers["X_imputed"] = adata.X.copy(order="K")


@progress("Log data statistics")
def log_data_statistics(X: np.ndarray, logger, indent_level: int = 1) -> float:
    """
    Logs various statistical properties of a given dataset.

    This function provides a quick summary of key statistics for a numpy array. It calculates
    and logs the number of observations (rows), features (columns), total missing values, and
    the percentage of missing values in the dataset. This function is particularly useful for
    initial data exploration and quality assessment in data analysis workflows.

    Parameters
    ----------
    X : np.ndarray
        A numpy array containing the dataset to be analyzed.

    logger : Logger
        A logging object for documenting the statistics and observations.

    indent_level : int, optional
        The level of indentation for the logger, with 1 being the default.

    Notes
    -----
    Understanding the basic statistics of a dataset is crucial in data preprocessing and
    analysis. This function highlights potential issues with data, like high levels of missing
    values, which could impact subsequent analyses.

    The function is designed to work seamlessly with datasets of varying sizes and complexities.
    The statistical summary provided helps in making informed decisions about further steps in
    data processing, such as imputation or feature selection.

    Example
    -------
    >>> data = np.random.rand(100, 5)
    >>> log_data_statistics(data, logger)
    # Logs number of observations, features, and details about missing values.

    """
    n_obs, n_features = X.shape
    total_nas = np.isnan(X).sum()
    percent_nas = 100 * total_nas / X.size if X.size else 0.0

    # Log various data statistics
    logger.info(f"There are {n_obs} observations", indent_level=2)
    logger.info(f"There are {n_features} features", indent_level=2)
    logger.info(f"Total missing values: {total_nas}", indent_level=2)
    logger.info(f"Percentage of missing values: {percent_nas:.2f}%", indent_level=2)

    return float(percent_nas)


@progress("Create anndata object")
def create_anndata_object(
    df: pd.DataFrame,
    logger,
    indent_level: int = 1,
) -> anndata.AnnData:
    """
    Creates an AnnData object from a pandas DataFrame.

    This function constructs an AnnData object, a central data structure for storing and
    manipulating high-dimensional biological data such as single-cell genomics data. It takes
    a pandas DataFrame and returns an AnnData object suitable for downstream analyses
    in bioinformatics pipelines.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame with sample names as the index and the feature names as columns.

    logger : Logger
        A logging object for documenting the process and any relevant observations.

    indent_level : int, optional
        The level of indentation for the logger, with 1 being the default.

    Returns
    -------
    anndata.AnnData
        An AnnData object populated with the data, observation names, and variable names.

    Notes
    -----
    AnnData objects are widely used in computational biology for storing large, annotated
    datasets. Their structured format ensures easy access and manipulation of data for
    various analytical purposes.

    This function is essential for converting raw or processed data into a format readily
    usable with tools and libraries that support AnnData objects, facilitating a seamless
    integration into existing bioinformatics workflows.

    Example
    -------
    >>> data = pd.DataFrame(np.random.rand(100, 5))
    >>> ann_data = create_anndata_object(data, logger)
    # Creates an AnnData object with 100 observations and 5 variables.

    """

    # Validate before dropping all-NA columns: dropping a duplicate label would
    # otherwise remove its nonmissing counterpart as well.
    if not df.columns.astype(str).is_unique:
        raise ValueError("There are duplicate feature names!")

    # With no samples, columns are unknown rather than known to be all missing.
    na_column_mask = df.isna().all() if len(df) else pd.Series(False, index=df.columns)

    # Calculate the number of columns with only NAs directly
    num_columns_dropped = na_column_mask.sum()

    if num_columns_dropped > 0:
        # Extract column names with only NAs
        columns_with_nas = df.columns[na_column_mask]

        # Prepare a snippet of column names for logging (max 3)
        sample_columns = columns_with_nas[: min(3, len(columns_with_nas))].tolist()

        logger.warning(
            f"Dropping {num_columns_dropped} columns with only NAs: {sample_columns}, etc.",
            indent_level=indent_level + 1,
        )

        # Drop columns with only NAs
        df = df.drop(columns=columns_with_nas)

    # Check each distinct dtype once; wide frames have hundreds of thousands of columns.
    dtypes = df.dtypes
    distinct = dtypes.drop_duplicates()
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in distinct):
        nonnumeric = [name for name, dtype in dtypes.items() if not pd.api.types.is_numeric_dtype(dtype)]
        raise TypeError(f"Feature columns must be numeric; move nonnumeric columns to metadata_cols: {nonnumeric[:3]}")
    if any(pd.api.types.is_complex_dtype(dtype) for dtype in distinct):
        raise ValueError("Feature values must be real numbers, not complex values.")

    # Copy to detach from pandas, whose copy-on-write arrays may be read-only.
    # Nullable numeric and mixed boolean/numeric columns need an explicit dtype.
    X = df.to_numpy(copy=True)
    if X.dtype.kind == "O":
        X = df.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
    if np.isinf(X).any():
        raise ValueError("Feature values must be finite or missing; infinity is not supported.")
    obs_names = df.index.map(str) if isinstance(df.index, pd.MultiIndex) else df.index.astype(str)
    var_names = df.columns.astype(str)

    obs = pd.DataFrame(index=obs_names)
    var = pd.DataFrame(index=var_names)

    # order="K" keeps pandas' column-major layout, which df_to_adata keeps for prediction,
    # and makes the copy a plain memcpy instead of a transpose.
    adata = anndata.AnnData(X=X, obs=obs, var=var, layers={"X_original": X.copy(order="K")})

    return adata


@progress("Add metadata to anndata")
def add_metadata_to_anndata(
    adata: anndata.AnnData,
    metadata: pd.DataFrame | None,
    logger,
    indent_level: int = 1,
) -> None:
    """
    Adds metadata to an AnnData object's observation (obs) attribute.

    This function enriches an AnnData object by integrating metadata. The metadata, provided as
    a pandas DataFrame, is aligned with the observation names in the AnnData object, ensuring
    consistency and completeness of data annotations. This process is crucial for downstream
    analyses where metadata (e.g., sample conditions, phenotypes) is key for interpretation.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to which metadata will be added. The obs attribute of this object
        will be modified.

    metadata : Optional[pd.DataFrame]
        A pandas DataFrame containing the metadata. Each row corresponds to an observation,
        and columns represent different metadata fields.

    logger : Logger
        A logging object for documenting the process and any observations.

    indent_level : int, optional
        The level of indentation for the logger, with 1 being the default.

    Notes
    -----
    The metadata DataFrame's index should match the observation names in the AnnData object for
    proper alignment. This function will reindex the metadata to match the AnnData obs_names,
    ensuring that each sample in the AnnData object is associated with its corresponding metadata.

    Example
    -------
    >>> import pandas as pd
    >>> from anndata import AnnData
    >>> adata = AnnData(np.random.rand(5, 3))
    >>> metadata = pd.DataFrame(
    ...     {"Condition": ["A", "B", "A", "B", "A"]},
    ...     index=[f"Sample_{i}" for i in range(5)],
    ... )
    >>> add_metadata_to_anndata(adata, metadata, logger)
    # Adds the 'Condition' metadata to the AnnData object.

    """
    if metadata is None:
        logger.warning("No metadata provided. Leaving adata.obs empty", indent_level=2)
        return

    # Add metadata to the AnnData object
    logger.info("Adding provided metadata to adata.obs", indent_level=2)
    metadata = metadata.copy()
    metadata.index = (
        metadata.index.map(str) if isinstance(metadata.index, pd.MultiIndex) else metadata.index.astype(str)
    )
    metadata = metadata.reindex(adata.obs_names)
    adata.obs = metadata


@progress("Add imputer strategy to adata.uns")
def add_unstructured_data(adata: anndata.AnnData, imputer_strategy: str, logger, indent_level: int = 1) -> None:
    """
    Adds unstructured data, such as imputer strategy and data type, to an AnnData object.

    This function is designed to annotate an AnnData object with additional unstructured
    information, enhancing data transparency and traceability. Key information, like the
    imputation strategy used and the type of biological data represented, is stored in the
    unstructured (uns) attribute of the AnnData object. This enrichment is vital for ensuring
    clarity and reproducibility in bioinformatics analyses.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to which the unstructured data will be added.

    imputer_strategy : str
        The strategy used for imputing missing values in the dataset, which will be recorded
        in the AnnData object for reference.

    logger : Logger
        A logging object for documenting the process and any important observations.

    indent_level : int, optional
        The level of indentation for the logger, with 1 being the default.

    Notes
    -----
    This function updates the 'uns' attribute of the AnnData object with the 'imputer_strategy'
    key.

    Example
    -------
    >>> from anndata import AnnData
    >>> adata = AnnData(np.random.rand(5, 3))
    >>> add_unstructured_data(adata, "mean", logger)
    # This will add the imputer strategy 'mean' and the data type 'dna_methylation' to the AnnData object.

    """
    # Add imputer strategy and data type to the AnnData object
    adata.uns["imputer_strategy"] = imputer_strategy


@progress("Load Ensembl genome metadata")
def load_ensembl_metadata(dir: str, logger, indent_level: int = 1) -> pd.DataFrame:
    """
    Load and filter Ensembl genome metadata specific to Homo sapiens.

    This function downloads the Ensembl gene metadata for Homo sapiens from the public pyaging
    Hugging Face data repository and filters it to include only genes on specified chromosomes.

    Parameters
    ----------
    dir : str
        Retained for backward compatibility. Hugging Face files use its standard cache.

    logger : Logger
        A logging object for recording the progress and status of the download and filtering process.

    indent_level : int, optional
        The indentation level for logging messages. It helps to organize the log output when this
        function is part of larger workflows. Defaults to 1.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing filtered gene metadata from Ensembl. Rows correspond to genes, indexed
        by their Ensembl gene IDs, and columns include various gene attributes.

    Notes
    -----
    The function currently filters genes based on a predefined set of chromosomes (1-22, X). If different
    chromosomes or additional filtering criteria are needed, modifications to the function will be required.

    Examples
    --------
    >>> logger = LoggerManager.gen_logger("ensembl_metadata")
    >>> ensembl_genes = load_ensembl_metadata("pyaging_data", logger)
    # This returns a DataFrame with Ensembl gene metadata for Homo sapiens filtered by specified chromosomes.

    """
    genes_path = download_hf_file(
        "Ensembl-105-EnsDb-for-Homo-sapiens-genes.csv",
        dir,
        logger,
        indent_level=1,
    )

    # Define chromosomes of interest
    chromosomes = [
        "1",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "2",
        "20",
        "21",
        "22",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "X",
    ]

    # Read and filter the gene data
    genes = pd.read_csv(genes_path, dtype={"chr": str})
    genes = genes[genes["chr"].isin(chromosomes)]
    genes.index = genes.gene_id
    return genes
