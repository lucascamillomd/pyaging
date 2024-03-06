import os
from typing import Optional

import anndata
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer

from ..utils import download, progress


@progress("Impute missing values")
def impute_missing_values(adata: anndata.AnnData, strategy: str, logger, indent_level: int = 1) -> None:
    """
    Imputes missing values in a given adata object using a specified strategy.

    This function handles missing data in by applying various imputation strategies.
    It checks the .X in the adata object for missing values and applies the chosen imputation
    method, which can be mean, median, constant, or K-nearest neighbors (KNN). The function is
    useful in preprocessing steps for datasets where missing data could affect subsequent analyses.
    It also adds the number of missing values for each sample and each feature.

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
    the K-nearest neighbors algorithm to estimate missing values based on similar samples.
    This function is particularly useful in datasets where missing values are common, such as
    in biological or medical data.

    The function ensures that no imputation is performed if there are no missing values in the
    dataset, thus preserving the original data integrity.

    Examples
    --------
    >>> imputed_adata = impute_missing_values(adata, "mean")
    # Imputes missing values using the mean of each column.

    """

    # Add percent of NAs to adata object
    adata.var["percent_na"] = np.isnan(adata.X).sum(axis=0) / adata.X.shape[0]

    # Check for missing values
    if adata.var["percent_na"].sum() == 0:
        logger.info("No missing values found. No imputation necessary", indent_level=2)
    else:
        # Dictionary mapping strategies to imputer objects
        imputers = {
            "mean": SimpleImputer(strategy="mean", keep_empty_features=True),
            "median": SimpleImputer(strategy="median", keep_empty_features=True),
            "constant": SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True),
            "knn": KNNImputer(),
        }

        # Select the appropriate imputer
        imputer = imputers.get(strategy)
        if not imputer:
            raise ValueError(f"Invalid imputer strategy: {strategy}")
        logger.info(f"Imputing missing values using {strategy} strategy", indent_level=2)
        adata.X = imputer.fit_transform(adata.X)
        adata.layers["X_imputed"] = adata.X


@progress("Log data statistics")
def log_data_statistics(X: np.ndarray, logger, indent_level: int = 1) -> None:
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
    percent_nas = 100 * total_nas / (n_obs * n_features)

    # Log various data statistics
    logger.info(f"There are {n_obs} observations", indent_level=2)
    logger.info(f"There are {n_features} features", indent_level=2)
    logger.info(f"Total missing values: {total_nas}", indent_level=2)
    logger.info(f"Percentage of missing values: {percent_nas:.2f}%", indent_level=2)


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

    # Identify columns with only NAs and store the boolean series
    na_column_mask = df.isna().all()

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

    # Extract information from df
    X = df.values
    obs_names = df.index.astype(str)
    var_names = df.columns.astype(str)

    # Check for duplicate features
    if len(np.unique(var_names)) != len(var_names):
        logger.error("There are duplicate feature names!")
        raise ValueError

    obs = pd.DataFrame(index=obs_names)
    var = pd.DataFrame(index=var_names)

    adata = anndata.AnnData(X=X, obs=obs, var=var, layers={"X_original": X})

    return adata


@progress("Add metadata to anndata")
def add_metadata_to_anndata(
    adata: anndata.AnnData,
    metadata: Optional[pd.DataFrame],
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
        return adata

    # Add metadata to the AnnData object
    logger.info("Adding provided metadata to adata.obs", indent_level=2)
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
    >>> adata = add_unstructured_data(adata, "mean", logger)
    # This will add the imputer strategy 'mean' and the data type 'dna_methylation' to the AnnData object.

    """
    # Add imputer strategy and data type to the AnnData object
    adata.uns["imputer_strategy"] = imputer_strategy


@progress("Load Ensembl genome metadata")
def load_ensembl_metadata(dir: str, logger, indent_level: int = 1) -> pd.DataFrame:
    """
    Load and filter Ensembl genome metadata specific to Homo sapiens.

    This function downloads the Ensembl gene metadata for Homo sapiens from a predefined URL and
    filters it to include only the genes located on specified chromosomes.

    Parameters
    ----------
    dir : str
        The directory to deposit the downloaded file.

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
    url = "https://pyaging.s3.amazonaws.com/supporting_files/Ensembl-105-EnsDb-for-Homo-sapiens-genes.csv"
    download(url, dir, logger, indent_level=1)

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
    genes_path = os.path.join(dir, "Ensembl-105-EnsDb-for-Homo-sapiens-genes.csv")
    genes = pd.read_csv(genes_path)
    genes = genes[genes["chr"].apply(lambda x: x in chromosomes)]
    genes.index = genes.gene_id
    return genes
