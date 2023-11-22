import os
import torch
import ntpath
import os
from urllib.request import urlretrieve
from functools import wraps
from ..utils import progress
from ..logger import LoggerManager, main_tqdm


@progress("Download data")
def download(url: str, logger, indent_level: int = 1):
    """
    Downloads a specific file given a file_id.

    Args:
    - url: url of the file to be downloaded.
    - logger: Logger object for logging messages.

    Returns:
    - pandas DataFrame with genome metadata.
    """
    file_path = url.split('/')[-1]
    dir="./pyaging_data"
    file_path = os.path.join(dir, file_path)
    
    if os.path.exists(file_path):
        logger.info(f'Data found in {file_path}', indent_level=indent_level+1)
    else:
        if not os.path.exists(dir):
            os.mkdir("pyaging_data")
        logger.info(f"Downloading data to {file_path}", indent_level=indent_level+1)
        logger.indent_level = indent_level+1
        urlretrieve(url, file_path, reporthook=logger.request_report_hook)


def download_example_data(data_type: str) -> None:
    """
    Download an example data file.

    Parameters:
    - data_type (str): the type of data to be downloaded.
    """
    logger = LoggerManager.gen_logger("download_example_data")
    logger.first_info("Starting download_example_data function")

    data_type_to_url = {
        "methylation": 'https://pyaging.s3.amazonaws.com/example_data/GSE139307.pkl',
        "histone_mark":'https://pyaging.s3.amazonaws.com/example_data/ENCFF386QWG.bigWig',
        "rnaseq": 'https://pyaging.s3.amazonaws.com/example_data/GSE65765_CPM.pkl',
        "atac": 'https://pyaging.s3.amazonaws.com/example_data/atac_example.pkl',
    }

    if data_type not in list(data_type_to_url.keys()):
        logger.error(f"Example data of {data_type} has not yet been implemented in pyaging. If you'd like it implemented, please send us an email. We hope to have a two-week max turnaround time.", indent_level=2)

    url = data_type_to_url[data_type]
    download(url, logger, indent_level=1)

    logger.done()
