from typing import Union, List
import ntpath
import os
from urllib.request import urlretrieve
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from typing import Optional
from pyBigWig import open as open_bw
import anndata
from functools import wraps
from ..logger import LoggerManager, main_tqdm, silence_logger
from ..utils import progress
from ..data import *


@progress("Impute missing values")
def impute_missing_values(
    X: np.ndarray, strategy: str, logger, indent_level: int = 1
) -> np.ndarray:
    """
    Imputes missing values in a given dataset using a specified strategy.

    This function handles missing data in a numpy array by applying various imputation strategies.
    It checks the array for missing values and applies the chosen imputation method, which can
    be mean, median, constant, or K-nearest neighbors (KNN). The function is useful in preprocessing
    steps for datasets where missing data could affect subsequent analyses.

    Parameters
    ----------
    X : np.ndarray
        A numpy array containing the dataset with potential missing values.

    strategy : str
        The imputation strategy to apply. Valid options are 'mean', 'median', 'constant', and 'knn'.

    logger : Logger
        A logging object for tracking the progress and outcomes of the function.

    indent_level : int, optional
        The level of indentation for the logger, with 1 being the default.

    Returns
    -------
    np.ndarray
        The imputed dataset as a numpy array.

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
    >>> data = np.array([[1, np.nan, 3], [4, 5, np.nan], [7, 8, 9]])
    >>> imputed_data = impute_missing_values(data, "mean")
    # Imputes missing values using the mean of each column.

    """
    # Check for missing values
    if not np.isnan(X).any():
        logger.info("No missing values found. No imputation necessary", indent_level=2)
        return X

    # Dictionary mapping strategies to imputer objects
    imputers = {
        "mean": SimpleImputer(strategy="mean", keep_empty_features=True),
        "median": SimpleImputer(strategy="median", keep_empty_features=True),
        "constant": SimpleImputer(
            strategy="constant", fill_value=0, keep_empty_features=True
        ),
        "knn": KNNImputer(),
    }

    # Select the appropriate imputer
    imputer = imputers.get(strategy)
    if not imputer:
        raise ValueError(f"Invalid imputer strategy: {strategy}")

    logger.info(f"Imputing missing values using {strategy} strategy", indent_level=2)
    return imputer.fit_transform(X)


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
    X_imputed: np.ndarray,
    obs_names: list,
    var_names: list,
    logger,
    indent_level: int = 1,
) -> anndata.AnnData:
    """
    Creates an AnnData object from imputed data, observation names, and variable names.

    This function constructs an AnnData object, a central data structure for storing and
    manipulating high-dimensional biological data such as single-cell genomics data. It takes
    an imputed numpy array, lists of observation names, and variable names, and returns an
    AnnData object suitable for downstream analyses in bioinformatics pipelines.

    Parameters
    ----------
    X_imputed : np.ndarray
        A numpy array containing the imputed data where rows are observations and columns
        are variables.

    obs_names : list
        A list of strings representing the names of the observations (e.g., cell names in
        single-cell analysis).

    var_names : list
        A list of strings representing the names of the variables (e.g., gene names).

    logger : Logger
        A logging object for documenting the process and any relevant observations.

    indent_level : int, optional
        The level of indentation for the logger, with 1 being the default.

    Returns
    -------
    anndata.AnnData
        An AnnData object populated with the imputed data, observation names, and variable names.

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
    >>> data = np.random.rand(100, 5)
    >>> obs_names = [f'Cell_{i}' for i in range(100)]
    >>> var_names = [f'Gene_{i}' for i in range(5)]
    >>> ann_data = create_anndata_object(data, obs_names, var_names, logger)
    # Creates an AnnData object with 100 observations and 5 variables.

    """
    return anndata.AnnData(
        X=X_imputed,
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=var_names),
        dtype="float",
    )


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

    Raises
    ------
    TypeError
        If the provided metadata is not a pandas DataFrame.

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
    >>> metadata = pd.DataFrame({'Condition': ['A', 'B', 'A', 'B', 'A']}, index=[f'Sample_{i}' for i in range(5)])
    >>> add_metadata_to_anndata(adata, metadata, logger)
    # Adds the 'Condition' metadata to the AnnData object.

    """
    if metadata is None:
        logger.warning("No metadata provided. Leaving adata.obs empty", indent_level=2)
        return

    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("Metadata must be a pandas DataFrame.")

    # Add metadata to the AnnData object
    logger.info("Adding provided metadata to adata.obs", indent_level=2)
    metadata = metadata.reindex(adata.obs_names)
    adata.obs = metadata


@progress("Add unstructured data to anndata")
def add_unstructured_data(
    adata: anndata.AnnData, imputer_strategy: str, logger, indent_level: int = 1
) -> None:
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
    and 'data_type' keys. The 'data_type' is currently hard-coded as 'dna_methylation', which
    may need modification based on different dataset types in future applications.

    Example
    -------
    >>> from anndata import AnnData
    >>> adata = AnnData(np.random.rand(5, 3))
    >>> add_unstructured_data(adata, 'mean', logger)
    # This will add the imputer strategy 'mean' and the data type 'dna_methylation' to the AnnData object.

    """
    # Add imputer strategy and data type to the AnnData object
    adata.uns["imputer_strategy"] = imputer_strategy
    adata.uns["data_type"] = "dna_methylation"


def df_to_adata(
    df: pd.DataFrame,
    metadata: Optional[pd.DataFrame] = None,
    imputer_strategy: str = "knn",
    verbose: bool = True,
) -> anndata.AnnData:
    """
    Converts a pandas DataFrame to an AnnData object.

    This function transforms a DataFrame containing biological data (such as gene expression
    levels, methylation data, etc.) into an AnnData object. It includes steps for handling
    missing values, logging data statistics, and embedding metadata into the AnnData object.
    The function is particularly useful in preparing datasets for downstream analyses in
    bioinformatics and computational biology.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing biological data. Rows represent samples, and columns represent features.

    metadata : Optional[pd.DataFrame], optional
        A DataFrame containing metadata associated with the samples. Each row should correspond to a
        sample in 'df', and columns should represent various metadata attributes. Defaults to None.

    imputer_strategy : str, optional
        The strategy for imputing missing values in 'df'. Supported strategies include 'mean',
        'median', 'constant', and 'knn'. Defaults to 'knn'.

    verbose: bool
        Whether to log the output to console with the logger. Defaults to True.

    Returns
    -------
    anndata.AnnData
        The AnnData object containing the processed data, metadata, and additional annotations.

    Raises
    ------
    TypeError
        If the input 'df' is not a pandas DataFrame.

    Notes
    -----
    The AnnData object produced by this function is ready for various computational biology analyses,
    such as differential expression analysis, clustering, or trajectory inference. The embedded metadata
    and annotations enhance data understanding and facilitate more robust analyses.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(np.random.rand(5, 3), columns=['gene1', 'gene2', 'gene3'])
    >>> metadata = pd.DataFrame({'condition': ['A', 'A', 'B', 'B', 'C']}, index=df.index)
    >>> adata = df_to_adata(df, metadata)
    # This returns an AnnData object with the data from 'df', imputed missing values,
    # and embedded sample metadata.

    """
    logger = LoggerManager.gen_logger("df_to_adata")
    if not verbose:
        silence_logger("df_to_adata")
    logger.first_info("Starting df_to_adata function")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input df must be a pandas DataFrame.")

    # Convert DataFrame to numpy array
    X = df.values
    obs_names = df.index
    var_names = df.columns

    # Impute missing values and log statistics
    X_imputed = impute_missing_values(X, imputer_strategy, logger)
    log_data_statistics(X_imputed, logger)

    # Create an AnnData object
    adata = create_anndata_object(X_imputed, obs_names, var_names, logger)

    # Add metadata and unstructured data
    add_metadata_to_anndata(adata, metadata, logger)
    add_unstructured_data(adata, imputer_strategy, logger)

    logger.done()
    return adata


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


def bigwig_to_df(
    bw_files: Union[str, List[str]], dir: str = "pyaging_data", verbose: bool = True
) -> pd.DataFrame:
    """
    Convert bigWig files to a DataFrame, extracting signal data for genomic regions.

    This function processes a list of bigWig files, extracting signal data (such as chromatin accessibility
    or histone modification levels) for each gene based on genomic annotations from Ensembl. It computes the
    mean signal over the genomic region of each gene, applies an arcsinh transformation for normalization,
    and organizes the data into a DataFrame format.

    Parameters
    ----------
    bw_files: Union[str, List[str]]
        A list of bigWig file paths. If a single string is provided, it is converted to a list.

    dir : str
        The directory to deposit the downloaded file. Defaults to "pyaging_data".

    verbose: bool
        Whether to log the output to console with the logger. Defaults to True.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row represents a bigWig file and each column corresponds to a gene.
        The values in the DataFrame are the transformed signal data for each gene in each bigWig file.

    Notes
    -----
    The function utilizes Ensembl gene annotations and assumes the presence of genes on standard chromosomes
    (1-22, X). Non-standard chromosomes or regions outside annotated genes are not processed. The signal
    transformation uses the arcsinh function for normalization.

    Examples
    --------
    >>> bigwig_files = ["sample1.bw", "sample2.bw"]
    >>> signals_df = bigwig_to_df(bigwig_files)
    # This returns a DataFrame where rows are bigWig files and columns are genes, with signal values.

    """
    logger = LoggerManager.gen_logger("bigwig_to_df")
    if not verbose:
        silence_logger("bigwig_to_df")
    logger.first_info("Starting bigwig_to_df function")

    # Ensure bws is a list
    if isinstance(bw_files, str):
        bw_files = [bw_files]

    # Get genomic annotation data
    genes = load_ensembl_metadata(dir, logger, indent_level=1)

    all_samples = []  # List to store signal data for each sample

    message = "Processing bigWig files"
    logger.start_progress(f"{message} started")
    for bw_file in bw_files:
        logger.info(f"Processing file: {bw_file}", indent_level=2)

        # Open bigWig file
        with open_bw(bw_file) as bw:
            signal_sample = np.empty(shape=(0, 0), dtype=float)
            for i in main_tqdm(range(genes.shape[0]), indent_level=2, logger=logger):
                try:
                    signal = bw.stats(
                        "chr" + genes["chr"].iloc[i],
                        genes["start"].iloc[i] - 1,
                        genes["end"].iloc[i],
                        type="mean",
                        exact=True,
                    )[0]
                except:
                    signal = None

                if signal is not None:
                    signal_transformed = np.arcsinh(signal)
                else:
                    signal_transformed = 0

                signal_sample = np.append(signal_sample, signal_transformed)

        # Append DataFrame for the current sample
        all_samples.append(
            pd.DataFrame(signal_sample[None, :], columns=genes.gene_id.tolist())
        )
    logger.finish_progress(f"{message} finished")

    # Concatenate all sample dataframes
    df_concat = pd.concat(all_samples, ignore_index=True)

    # Add file name as index
    df_concat.index = bw_files

    logger.done()
    return df_concat
