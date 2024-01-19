import pytest
import os
import pyaging as pya
from pyaging.utils import *
from unittest.mock import Mock, patch

import os
import torch
import ntpath
import io
from contextlib import redirect_stdout
from IPython.display import display, HTML
from urllib.request import urlretrieve
from functools import wraps

from pyaging.logger import LoggerManager, main_tqdm

@pytest.fixture
def mock_logger():
    return Mock()

@pytest.fixture
def mock_urlretrieve():
    with patch('urllib.request.urlretrieve') as mock:
        yield mock

def test_file_already_exists(tmp_path, mock_logger):
    # Setup
    url = "https://example.com/datafile.zip"
    dir = str(tmp_path)
    file_name = url.split('/')[-1]
    file_path = os.path.join(dir, file_name)

    # Create a dummy file to simulate existing file
    with open(file_path, 'w') as f:
        f.write("dummy data")

    # Call function
    download(url, dir, mock_logger)

    # Assertions
    mock_logger.info.assert_called_with(f"Data found in {file_path}", indent_level=2)

def test_file_downloads_successfully(tmp_path, mock_logger, mock_urlretrieve):
    # Setup
    url = "https://pyaging.s3.amazonaws.com/example_data/blood_chemistry_example.pkl"
    dir = str(tmp_path)
    file_name = url.split('/')[-1]
    file_path = os.path.join(dir, file_name)

    # Call function
    download(url, dir, mock_logger)

    # Assertions
    mock_urlretrieve.assert_called_with(url, file_path, reporthook=mock_logger.request_report_hook)
    mock_logger.info.assert_called_with(f"Downloading data to {file_path}", indent_level=2)

def test_download_failure_raises_IOError(tmp_path, mock_logger, mock_urlretrieve):
    # Setup
    url = "https://example.com/datafile.zip"
    dir = str(tmp_path)
    mock_urlretrieve.side_effect = Exception("Download failed")

    # Assertions
    with pytest.raises(IOError):
        download(url, dir, mock_logger)
