from typing import Union, List
import pandas as pd
import numpy as np
from typing import Optional
from pyBigWig import open as open_bw
import anndata

from ..logger import LoggerManager, main_tqdm, silence_logger
from ._preprocess_utils import *
from ..utils import download


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
    >>> df = pd.DataFrame(np.random.rand(5, 3), columns=['gene1', 'gene2', 'gene3'])
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

    logger.done()

    return adata
