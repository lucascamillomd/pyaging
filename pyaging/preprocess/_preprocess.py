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
from ..logger import LoggerManager, main_tqdm
from ..utils import progress
from ..data import *


@progress("Impute missing values")
def impute_missing_values(X: np.ndarray, strategy: str, logger, indent_level: int = 1) -> np.ndarray:
    """
    Imputes missing values in a numpy array using the specified strategy.

    Args:
    - X: numpy array with potentially missing values.
    - strategy: String specifying the imputation strategy ('mean', 'median', 'constant', 'knn').
    - logger: Logger object for logging messages.

    Returns:
    - numpy array with missing values imputed.
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
    Logs statistics about the data array.

    Args:
    - X: numpy array containing the data.
    - logger: Logger object for logging messages.

    No return value; function only logs information.
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
    X_imputed: np.ndarray, obs_names: list, var_names: list, logger, indent_level: int = 1
) -> anndata.AnnData:
    """
    Creates an AnnData object from imputed data.

    Args:
    - X_imputed: numpy array with imputed data.
    - obs_names: list of obs names.
    - var_names: list of feature names.
    - logger: Logger object for logging messages.

    Returns:
    - AnnData object containing the imputed data.
    """
    return anndata.AnnData(X=X_imputed, obs=pd.DataFrame(index=obs_names), var=pd.DataFrame(index=var_names), dtype="float")


@progress("Add metadata to anndata")
def add_metadata_to_anndata(
    adata: anndata.AnnData, metadata: Optional[pd.DataFrame], logger, indent_level: int = 1
) -> None:
    """
    Adds metadata to an AnnData object.

    Args:
    - adata: AnnData object to which metadata should be added.
    - metadata: pandas DataFrame containing metadata. If None, no action is taken.
    - logger: Logger object for logging messages.

    No return value; function modifies AnnData object in place.
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
    Adds unstructured data to an AnnData object.

    Args:
    - adata: AnnData object to be updated.
    - imputer_strategy: String denoting the imputation strategy used.
    - logger: Logger object for logging messages.

    No return value; function modifies AnnData object in place.
    """
    # Add imputer strategy and data type to the AnnData object
    adata.uns["imputer_strategy"] = imputer_strategy
    adata.uns["data_type"] = "dna_methylation"


def df_to_adata(
    df: pd.DataFrame,
    metadata: Optional[pd.DataFrame] = None,
    imputer_strategy: str = "knn",
) -> anndata.AnnData:
    """
    Converts a DataFrame to an AnnData object.

    Args:
    - df: pandas DataFrame with methylation data.
    - metadata: Optional DataFrame with metadata.
    - imputer_strategy: String specifying the imputation method.

    Returns:
    - AnnData object.

    Raises TypeError if input types are incorrect or ValueError for invalid imputer strategy.
    """
    logger = LoggerManager.gen_logger("df_to_adata")
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
def load_ensembl_metadata(logger, indent_level: int = 1) -> pd.DataFrame:
    """
    Loads Ensembl genome metadata.

    Args:
    - logger: Logger object for logging messages.

    Returns:
    - pandas DataFrame with genome metadata.
    """
    url = "https://pyaging.s3.amazonaws.com/supporting_files/Ensembl-105-EnsDb-for-Homo-sapiens-genes.csv"

    download(url, logger, indent_level=1)
        
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
    genes = pd.read_csv("./pyaging_data/Ensembl-105-EnsDb-for-Homo-sapiens-genes.csv")
    genes = genes[genes["chr"].apply(lambda x: x in chromosomes)]
    genes.index = genes.gene_id
    return genes


def bigwig_to_df(
    bw_files
) -> pd.DataFrame:
    """
    Converts a list of bigWig files to a pandas DataFrame.

    Args:
    - bw_files: List of bigWig file paths.

    Returns:
    - DataFrame with transformed signal values from each file as a separate sample.
    """
    logger = LoggerManager.gen_logger("bigwig_to_df")
    logger.first_info("Starting bigwig_to_df function")

    # Ensure bws is a list
    if isinstance(bw_files, str):
        bw_files = [bw_files]

    # Get genomic annotation data
    genes = load_ensembl_metadata(logger, indent_level=1)

    all_samples = []  # List to store signal data for each sample
    
    message = "Processing bigWig files"
    logger.start_progress(f"{message} started")
    for bw_file in bw_files:
        logger.info(f"Processing file: {bw_file}", indent_level=2)

        # Open bigWig file
        with open_bw(bw_file) as bw:
            signal_sample = np.empty(shape=(0, 0), dtype=float)
            for i in main_tqdm(range(genes.shape[0]), indent_level=2):
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
        all_samples.append(pd.DataFrame(signal_sample[None, :], columns=genes.gene_id.tolist()))
    logger.finish_progress(f"{message} finished")

    # Concatenate all sample dataframes
    df_concat = pd.concat(all_samples, ignore_index=True)

    # Add file name as index
    df_concat.index = bw_files
    
    logger.done()
    return df_concat