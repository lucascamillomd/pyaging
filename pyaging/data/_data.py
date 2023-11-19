import os
import torch
import ntpath
import os
from urllib.request import urlretrieve
from functools import wraps
from ..utils import progress
from ..logger import LoggerManager, main_tqdm


@progress("Download example data", indent_level=2)
def download(file_id: str, file_path: str, logger):
    """
    Downloads a specific file given a file_id.

    Args:
    - file_id: id of the file to be downloaded.
    - file_path: name of the file to be downloaded.
    - logger: Logger object for logging messages.

    Returns:
    - pandas DataFrame with genome metadata.
    """
    url = f"https://drive.google.com/uc?id={file_id}"
    dir="./pyaging_data"
    file_path = os.path.join(dir, file_path)
    
    if os.path.exists(file_path):
        logger.info(f'Data found in {file_path}', indent_level=3)
    else:
        if not os.path.exists(dir):
            os.mkdir("pyaging_data")
        logger.info(f"Downloading data to {file_path}", indent_level=3)
        logger.indent_level = 3
        urlretrieve(url, file_path, reporthook=logger.request_report_hook)


def download_example_data(data_type: str) -> None:
    """
    Download an example data file.

    Parameters:
    - data_type (str): the type of data to be downloaded.
    """
    logger = LoggerManager.gen_logger("download_example_data")
    logger.first_info("Starting download_example_data function")

    data_type_to_file_id = {
        "methylation": ['1y3fgki3NPT1rvuxjww-B4LrQNLPKLyCv', 'GSE139307.pkl'],
        "histone_mark": ['19xejGDPuA0OlK7_bnmRLnHS-t46WpgVN', 'ENCFF386QWG.bigWig'],
        "rnaseq": ['1oDxTtAmCYn7GquRPDhoWQMMKiumikNBe', 'GSE65765_CPM.pkl'],
        "atac": ['1T8oBiqtXyBRxTa16hHrSMkjdSGJVANA9', 'atac_example.pkl'],
    }

    if data_type not in list(data_type_to_file_id.keys()):
        logger.error(f"Example data of {data_type} has not yet been implemented in pyaging. If you'd like it implemented, please send us an email. We hope to have a two-week max turnaround time.", indent_level=2)

    file_id = data_type_to_file_id[data_type][0]
    file_path = data_type_to_file_id[data_type][1]
    download(file_id, file_path, logger)

    logger.done()
