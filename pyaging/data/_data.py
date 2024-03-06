from ..logger import LoggerManager, silence_logger
from ..utils import download


def download_example_data(data_type: str, dir: str = "pyaging_data", verbose: bool = True) -> None:
    """
    Downloads example datasets for various types of biological data used in aging studies.

    This function facilitates the download of example datasets for different types of biological data,
    including methylation, histone mark, RNA-seq, and ATAC-seq data. It is designed to provide quick
    access to standard datasets for users to test and explore the functionalities of the pyaging package.

    Parameters
    ----------
    data_type : str
        The type of data to download. Valid options are 'GSE139307', 'GSE130735', 'GSE223748',
        'ENCFF386QWG', 'GSE65765', 'GSE193140', and 'blood_chemistry_example'.

    dir : str
        The directory to deposit the downloaded file. Defaults to "pyaging_data".

    verbose : bool
        Whether to log the output to console with the logger. Defaults to True.

    Raises
    ------
    ValueError
        If the specified data_type is not implemented, a ValueError is raised with a message suggesting
        the user to request its implementation.

    Notes
    -----
    The function maps the specified data_type to its corresponding URL and then calls the `download`
    function to retrieve the dataset. The datasets are sourced from AWS S3 and are chosen to represent
    typical data formats and structures used in aging research.


    Examples
    --------
    >>> download_example_data("methylation")
    >>> # This will download the example methylation dataset to the local system.

    """
    logger = LoggerManager.gen_logger("download_example_data")
    if not verbose:
        silence_logger("download_example_data")
    logger.first_info("Starting download_example_data function")

    data_type_to_url = {
        "GSE130735": "https://pyaging.s3.amazonaws.com/example_data/GSE130735_subset.pkl",
        "GSE193140": "https://pyaging.s3.amazonaws.com/example_data/GSE193140.pkl",
        "GSE139307": "https://pyaging.s3.amazonaws.com/example_data/GSE139307.pkl",
        "GSE223748": "https://pyaging.s3.amazonaws.com/example_data/GSE223748_subset.pkl",
        "ENCFF386QWG": "https://pyaging.s3.amazonaws.com/example_data/ENCFF386QWG.bigWig",
        "GSE65765": "https://pyaging.s3.amazonaws.com/example_data/GSE65765_CPM.pkl",
        "blood_chemistry_example": "https://pyaging.s3.amazonaws.com/example_data/blood_chemistry_example.pkl",
    }

    if data_type not in data_type_to_url:
        logger.error(
            f"Example data {data_type} has not yet been implemented in pyaging.",
            indent_level=2,
        )
        raise ValueError

    url = data_type_to_url[data_type]
    download(url, dir, logger, indent_level=1)
    logger.done()
