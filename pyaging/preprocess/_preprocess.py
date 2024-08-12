from typing import List, Union

import anndata
import numpy as np
import pandas as pd

try:
    from pyBigWig import open as open_bw

    PYBIGWIG_AVAILABLE = True
except ImportError:
    PYBIGWIG_AVAILABLE = False

try:
    import cupy as cp

    CUPY_AVAILABLE = cp.cuda.is_available()
except:
    CUPY_AVAILABLE = False

from ..logger import LoggerManager, main_tqdm, silence_logger
from ._preprocess_utils import *


def bigwig_to_df(bw_files: Union[str, List[str]], dir: str = "pyaging_data", verbose: bool = True) -> pd.DataFrame:
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

    Raises
    ------
    ImportError
        If pyBigWig is not installed and the function is called.

    Notes
    -----
    The function utilizes Ensembl gene annotations and assumes the presence of genes on standard chromosomes
    (1-22, X). Non-standard chromosomes or regions outside annotated genes are not processed. The signal
    transformation uses the arcsinh function for normalization. This function requires pyBigWig to be installed.
    If pyBigWig is not available, an ImportError will be raised. To use this function, ensure you have installed
    pyaging with the 'bigwig' extra: pip install pyaging[bigwig]

    Examples
    --------
    >>> bigwig_files = ["sample1.bw", "sample2.bw"]
    >>> signals_df = bigwig_to_df(bigwig_files)
    # This returns a DataFrame where rows are bigWig files and columns are genes, with signal values.

    """
    if not PYBIGWIG_AVAILABLE:
        raise ImportError("pyBigWig is not installed. To use this function, please install it.")

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

                signal_transformed = np.arcsinh(signal) if signal is not None else 0

                signal_sample = np.append(signal_sample, signal_transformed)

        # Append DataFrame for the current sample
        all_samples.append(pd.DataFrame(signal_sample[None, :], columns=genes.gene_id.tolist()))
    logger.finish_progress(f"{message} finished")

    # Concatenate all sample dataframes
    df_concat = pd.concat(all_samples, ignore_index=True)

    # Add file name as index
    df_concat.index = bw_files

    logger.done()
    return df_concat


def df_to_adata(
    df: pd.DataFrame,
    metadata_cols: List[str] = [],
    imputer_strategy: str = "knn",
    verbose: bool = True,
) -> anndata.AnnData:
    """
    Converts a pandas DataFrame to an AnnData object.

    This function transforms a DataFrame containing biological data (such as gene expression
    levels, methylation data, etc.) into an AnnData object. It includes steps for handling
    missing values, and logging data statistics. The function is particularly useful
    in preparing datasets for downstream analyses in bioinformatics and computational biology.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing biological data. Rows represent samples, and columns represent features.

    metadata_cols : List[str]
        A list with the name of the columns in 'df' which are part of the metadata. They will be added
        to adata.obs rather than adata.X.

    imputer_strategy : str, optional
        The strategy for imputing missing values in 'df'. Supported strategies include 'mean',
        'median', 'constant' (0 values), and 'knn'. Defaults to 'knn'.

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
    such as differential expression analysis, clustering, or trajectory inference. The embedded annotations
    enhance data understanding and facilitate more robust analyses.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(np.random.rand(5, 3), columns=["gene1", "gene2", "gene3"])
    >>> adata = df_to_adata(df)
    # This returns an AnnData object with the imputed data from 'df'.

    """
    logger = LoggerManager.gen_logger("df_to_adata")
    if not verbose:
        silence_logger("df_to_adata")
    logger.first_info("Starting df_to_adata function")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input df must be a pandas DataFrame.")

    # Split data and metadata
    if len(metadata_cols) > 0:
        metadata = df.loc[:, metadata_cols]
        df = df.drop(metadata_cols, axis=1)
    else:
        metadata = None

    # Create an AnnData object
    adata = create_anndata_object(df, logger)

    # Add metadata
    add_metadata_to_anndata(adata, metadata, logger)

    # Log statistics
    log_data_statistics(adata.X, logger)

    # Impute missing values
    impute_missing_values(adata, imputer_strategy, logger)

    # Add unstructured data
    if "X_imputed" in adata.layers:
        add_unstructured_data(adata, imputer_strategy, logger)

    # Move adata.X to GPU if possible
    adata.X = cp.array(adata.X) if CUPY_AVAILABLE else np.asfortranarray(adata.X)

    logger.done()

    return adata


def epicv2_probe_aggregation(df: pd.DataFrame, verbose: bool = True):
    """
    Aggregates probes targeting the same CpG site in a DataFrame from the Illumina Methylation EPIC array v2.

    Probes targeting the same CpG site are identified by their shared prefix (e.g., "cgXXXXXXX"), and their
    values are averaged to create a single feature for each unique CpG site. This reduces the dimensionality
    of the data by consolidating multiple probes for the same CpG site into a single value.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing probe data. Each column represents a probe, and the column names are
        expected to follow the format "cgXXXXXXX_YYYY".

    verbose: bool
        Whether to log the output to console with the logger. Defaults to True.

    Returns
    -------
    pandas.DataFrame:
        A new DataFrame with averaged values for each unique CpG site. The columns of this DataFrame correspond
        to unique CpG sites, and the column names are the CpG site identifiers (e.g., "cgXXXXXXX").
    """

    logger = LoggerManager.gen_logger("epicv2_probe_aggregation")
    if not verbose:
        silence_logger("epicv2_probe_aggregation")
    logger.first_info("Starting epicv2_probe_aggregation function")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input df must be a pandas DataFrame.")

    # Create an empty dictionary to store aggregated data
    aggregated_data = {}
    n_duplicated_probes = 0

    # Logging the start of probe dictionary creation
    message = "Looking for duplicated probes"
    logger.start_progress(f"{message} started")
    for column in main_tqdm(df.columns, indent_level=2, logger=logger):
        cpg_site = column.split("_")[0]
        if cpg_site in aggregated_data:
            n_duplicated_probes += 1
            aggregated_data[cpg_site].append(df[column])
        else:
            aggregated_data[cpg_site] = [df[column]]
    # In case there are no duplicated probes, just return current array
    if n_duplicated_probes == 0:
        logger.info("There are no duplicated probes. Returning original data", indent_level=2)
        logger.done()
        return df
    else:
        logger.info(
            f"There are {n_duplicated_probes} duplicated probes in the data",
            indent_level=2,
        )
    logger.finish_progress(f"{message} finished")

    # Logging the start of averaging duplicated probes
    message = "Averaging duplicated probes"
    logger.start_progress(f"{message} started")
    aggregated_columns = []
    for cpg_site, columns in main_tqdm(aggregated_data.items(), indent_level=2, logger=logger):
        if len(columns) > 1:
            mean_series = pd.concat(columns, axis=1).mean(axis=1)
            mean_series.name = cpg_site
            aggregated_columns.append(mean_series)
        else:
            # Directly use the single column DataFrame if there's only one probe for the CpG site
            aggregated_columns.append(columns[0].rename(cpg_site))
    logger.finish_progress(f"{message} finished")

    # Concatenate all aggregated columns to form the final DataFrame
    aggregated_df = pd.concat(aggregated_columns, axis=1)

    logger.done()

    return aggregated_df
