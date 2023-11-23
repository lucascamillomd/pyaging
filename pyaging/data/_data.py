import os
import torch
import ntpath
import os
from urllib.request import urlretrieve
from functools import wraps
from ..utils import progress, download
from ..logger import LoggerManager, main_tqdm


def download_example_data(data_type: str) -> None:
    """
    Downloads example datasets for various types of biological data used in aging studies.

    This function facilitates the download of example datasets for different types of biological data,
    including methylation, histone mark, RNA-seq, and ATAC-seq data. It is designed to provide quick
    access to standard datasets for users to test and explore the functionalities of the pyaging package.

    Parameters
    ----------
    data_type : str
        The type of data to download. Valid options are 'methylation', 'histone_mark', 'rnaseq', and 'atac'.

    Raises
    ------
    ValueError
        If the specified data_type is not implemented, a ValueError is raised with a message suggesting
        the user to request its implementation.

    Notes
    -----
    The function maps the specified data_type to its corresponding URL and then calls the `download`
    function to retrieve the dataset. The datasets are sourced from AWS S3
    and are chosen to represent typical data formats and structures used in aging research.

    The downloaded data can be used for various analyses, including testing the pyaging package's
    functionalities, learning data processing techniques, or as a benchmark for custom analyses.

    Examples
    --------
    >>> download_example_data("methylation")
    # This will download the example methylation dataset to the local system.

    """
    logger = LoggerManager.gen_logger("download_example_data")
    logger.first_info("Starting download_example_data function")

    data_type_to_url = {
        "methylation": "https://pyaging.s3.amazonaws.com/example_data/GSE139307.pkl",
        "histone_mark": "https://pyaging.s3.amazonaws.com/example_data/ENCFF386QWG.bigWig",
        "rnaseq": "https://pyaging.s3.amazonaws.com/example_data/GSE65765_CPM.pkl",
        "atac": "https://pyaging.s3.amazonaws.com/example_data/atac_example.pkl",
    }

    if data_type not in list(data_type_to_url.keys()):
        logger.error(
            f"Example data {data_type} has not yet been implemented in pyaging.",
            indent_level=2,
        )

    url = data_type_to_url[data_type]
    download(url, logger, indent_level=1)

    logger.done()
