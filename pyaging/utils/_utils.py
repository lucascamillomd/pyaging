import os
from datetime import datetime
from functools import wraps
from pprint import pformat
from urllib.request import urlretrieve

import pytz
import requests
import torch

from ..logger import LoggerManager, main_tqdm


def progress(message: str) -> None:
    """
    A decorator to add progress logging to a function.

    This decorator wraps a function to add starting and finishing progress messages to the
    logger. It extracts the `indent_level` from keyword arguments, defaults to 1 if not provided,
    and assumes the logger is the last positional argument. It logs the start and end of the
    function execution with the provided message.

    Parameters
    ----------
    message : str
        The message to be logged before and after the function execution. This message is
        formatted as '{message} started' at the beginning and '{message} finished' at the end.

    Returns
    -------
    decorator : function
        A decorator function that wraps the original function with progress logging.

    Raises
    ------
    AttributeError
        If the logger object is not found as the last positional argument, an AttributeError
        might be raised when attempting to call `start_progress` or `finish_progress`.

    Notes
    -----
    The decorator assumes that the logger object is passed as the last positional argument to the
    function being decorated. It manipulates `kwargs` to extract `indent_level` if provided,
    otherwise defaults to 1. The `indent_level` controls the indentation of the log messages.

    This will log 'Processing data started' before the `data_processing` function begins and
    'Processing data finished' after it completes.

    Examples
    --------
    >>> @progress("Processing data")
    ... def data_processing(data, logger):
    ...     # data processing logic
    ...     return processed_data

    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract indent_level from kwargs, default to 1 if not provided
            indent_level = kwargs["indent_level"] if "indent_level" in kwargs else 1

            logger = args[-1]  # Assumes logger is the last positional argument
            logger.start_progress(f"{message} started", indent_level=indent_level)
            result = func(*args, **kwargs)
            logger.finish_progress(f"{message} finished", indent_level=indent_level)
            return result

        return wrapper

    return decorator


@progress("Load all clock metadata")
def load_clock_metadata(dir: str, logger, indent_level: int = 2) -> dict:
    """
    Loads the clock metadata from a specified source.

    This function checks if the metadata file exists locally in the specified directory.
    If it doesn't, the function downloads the file from AWS S3 using the provided file name
    and saves it in the 'pyaging_data' directory. After downloading or confirming the file's
    existence, it reads and returns the metadata.

    Parameters
    ----------
    dir : str
        The directory to deposit the downloaded file.
    logger : object
        Logger object used for logging information, warnings, and errors.
    indent_level : int, optional
        The level of indentation for logging messages, by default 1.

    Returns
    -------
    all_clock_metadata : dict
        A dictionary containing the loaded clock metadata.

    Raises
    ------
    IOError
        If the file download fails or the file cannot be read after downloading.

    Notes
    -----
    The function assumes the presence of a folder named 'pyaging_data' in the current directory
    or creates it if it doesn't exist. It uses a predefined AWS S3 url to download the
    metadata file. The function is decorated with `@progress`, which adds start and end log
    messages for this process.

    Examples
    --------
    >>> logger = pyaging.logger.LoggerManager.gen_logger("example")
    >>> metadata = load_clock_metadata("pyaging_data", logger)
    >>> type(metadata)
    <class 'dict'>

    """
    url = "https://pyaging.s3.amazonaws.com/clocks/metadata0.1.0/all_clock_metadata.pt"
    download(url, dir, logger, indent_level=indent_level)
    all_clock_metadata = torch.load(f"{dir}/all_clock_metadata.pt")
    return all_clock_metadata


def download(url: str, dir: str, logger, indent_level: int = 1):
    """
    Downloads a file from a specified URL to a local directory.

    This function checks if the file specified by the URL already exists in the local
    'pyaging_data' directory. If the file is not present or it is not the latest, it downloads
    the file from the URL and saves it in this directory. The function logs the progress of the
    download, including whether the file is found locally or needs to be downloaded.

    Parameters
    ----------
    url : str
        The URL of the file to be downloaded.
    dir : str
        The directory to deposit the downloaded file.
    logger : object
        Logger object for logging messages at various stages of the download process.
    indent_level : int, optional
        The level of indentation for logging messages, by default 1.

    Raises
    ------
    IOError
        If the download fails or the file cannot be saved to the local directory.

    Notes
    -----
    The function assumes the presence of a folder named 'pyaging_data' in the current directory,
    creating it if it doesn't exist. It uses Python's `urlretrieve` function from the `urllib`
    module for downloading the file. The function is decorated with `@progress`, which adds
    start and end log messages for the download process.

    Examples
    --------
    >>> logger = Logger()
    >>> download("https://example.com/datafile.zip", "pyaging_data", logger)
    Data found in pyaging_data/datafile.zip
    or
    Downloading data to pyaging_data/datafile.zip

    """
    file_path = url.split("/")[-1]
    file_path = os.path.join(dir, file_path)

    # aws_newer = is_newer_than_target(url, '2024-01-22')
    aws_newer = False  # REVISIT THIS

    if os.path.exists(file_path) and not aws_newer:
        logger.info(f"Data found in {file_path}", indent_level=indent_level + 1)
    elif os.path.exists(file_path) and aws_newer:
        logger.info(
            f"Data found in {file_path} is not the latest",
            indent_level=indent_level + 1,
        )
        logger.info(f"Redownloading data to {file_path}", indent_level=indent_level + 1)
        logger.indent_level = indent_level + 1
        urlretrieve(url, file_path, reporthook=logger.request_report_hook)
    else:
        if not os.path.exists(dir):
            os.mkdir(dir)
        logger.info(f"Downloading data to {file_path}", indent_level=indent_level + 1)
        logger.indent_level = indent_level + 1
        urlretrieve(url, file_path, reporthook=logger.request_report_hook)


def find_clock_by_doi(search_doi: str, dir: str = "pyaging_data") -> None:
    """
    Searches for aging clocks in the metadata by a specified DOI (Digital Object Identifier).

    This function retrieves the metadata for all aging clocks and searches for clocks that match
    the given DOI. It uses a Logger object for logging the progress and results of the search.
    The function outputs the names of clocks with the matching DOI, or a warning message if no
    matches are found.

    Parameters
    ----------
    search_doi : str
        The DOI to search for in the aging clocks' metadata.
    dir : str
        The directory to deposit the downloaded file. Defaults to 'pyaging_data'.

    Returns
    -------
    None
        The function does not return a value but logs the search results.

    Notes
    -----
    The function internally calls `load_clock_metadata` to load the metadata of all available
    aging clocks. It then iterates over this metadata to find matches. The logging includes
    starting and ending messages for the search process, and a summary of the findings.

    The function assumes the existence of a LoggerManager for generating loggers and uses
    `main_tqdm` for progress tracking in the loop. It's important to ensure that the metadata
    contains the 'doi' field for each clock for the search to be effective.

    Examples
    --------
    >>> find_clock_by_doi("10.1155/2020/123456")
    Clocks with DOI 10.1155/2020/123456: Clock1, Clock2

    or, if no match is found,

    >>> find_clock_by_doi("10.1000/xyz123")
    No files found with DOI 10.1000/xyz123

    """
    logger = LoggerManager.gen_logger("find_clock_by_doi")
    logger.first_info("Starting find_clock_by_doi function")

    # Load all metadata
    all_clock_metadata = load_clock_metadata(dir, logger, indent_level=1)

    # Message to indicate the start of the search process
    message = "Searching for clock based on DOI"
    logger.start_progress(f"{message} started")
    matching_clocks = []

    # Loop through clocks in the dictionary
    for clock_name in main_tqdm(list(all_clock_metadata.keys()), indent_level=2):
        clock_dict = all_clock_metadata[clock_name]
        if "doi" in clock_dict and clock_dict["doi"] == search_doi:
            matching_clocks.append(clock_name)

    # Logging the results
    if matching_clocks:
        logger.info(
            f"Clocks with DOI {search_doi}: {', '.join(matching_clocks)}",
            indent_level=2,
        )
    else:
        logger.warning(f"No files found with DOI {search_doi}", indent_level=2)
    logger.finish_progress(f"{message} finished")

    logger.done()


def cite_clock(clock_name: str, dir: str = "pyaging_data") -> None:
    """
    Retrieves and logs the citation information for a specified aging clock.

    This function searches the metadata for aging clocks to find and log the citation details
    of a specified clock. If the clock is found but no citation information is available,
    it logs a warning indicating the absence of citation data. If the clock is not found in
    the metadata, it logs a warning that the clock is unavailable.

    Parameters
    ----------
    clock_name : str
        The name of the aging clock for which citation information is to be retrieved.
        The function is case-insensitive to the clock name.
    dir : str
        The directory to deposit the downloaded file. Defaults to 'pyaging_data'.

    Returns
    -------
    None
        The function does not return a value but logs the citation details or warnings.

    Notes
    -----
    The function calls `load_clock_metadata` to load the entire metadata of aging clocks and
    then searches for the specified clock. It logs the progress of the search and the results.
    The `LoggerManager` is used for generating loggers for logging purposes.

    The function assumes that the metadata for each clock may contain a 'citation' field. If
    this field is missing, the function will indicate that no citation information is available.

    Examples
    --------
    >>> cite_clock("ClockX")
    Citation for clockx:
    Smith, A. B., et al. (2020). "A New Aging Clock Model." Aging Research, vol. 30, pp. 100-110.

    or, if citation data is not available,

    >>> cite_clock("ClockY")
    Citation not found in clocky

    or, if the clock is not in the metadata,

    >>> cite_clock("UnknownClock")
    UnknownClock is not currently available in pyaging

    """
    logger = LoggerManager.gen_logger("cite_clock")
    logger.first_info("Starting cite_clock function")

    clock_name = clock_name.lower()

    # Load all metadata
    all_clock_metadata = load_clock_metadata(dir, logger, indent_level=1)

    message = f"Searching for citation of clock {clock_name}"
    logger.start_progress(f"{message} started")
    citation = ""

    if clock_name in list(all_clock_metadata.keys()):
        clock_dict = all_clock_metadata[clock_name]
        if "citation" in clock_dict:
            citation = clock_dict["citation"]
            logger.info(f"Citation for {clock_name}:", indent_level=2)
            logger.info(citation, indent_level=2)
            logger.info("Please also consider citing pyaging :)", indent_level=2)
            logger.info(
                'de Lima Camillo, Lucas Paulo. "pyaging: a Python-based compendium of GPU-optimized aging clocks." bioRxiv (2023): 2023-11.',
                indent_level=2,
            )
        else:
            logger.warning(f"Citation not found in {clock_name}", indent_level=2)
    else:
        logger.warning(f"{clock_name} is not currently available in pyaging", indent_level=2)

    logger.finish_progress(f"{message} finished")
    logger.done()


def show_all_clocks(dir: str = "pyaging_data") -> None:
    """
    Displays the names of all aging clocks available in the metadata.

    This function retrieves the metadata for all aging clocks and logs each clock's name.
    It's useful for users to get a quick overview of all the clocks included in the pyaging
    package. The function utilizes a logger for structured output, providing clarity and
    readability in its logs.

    Parameters
    ----------
    dir : str
        The directory to deposit the downloaded file. Defaults to 'pyaging_data'.

    Returns
    -------
    None
        The function only prints the results.

    Notes
    -----
    The function calls `load_clock_metadata` to load the metadata containing the aging clocks.
    It then iterates over this metadata to log the name of each clock. The function uses the
    `LoggerManager` for logging, ensuring that all log messages are properly formatted and
    indented.

    The logger's progress methods (`start_progress` and `finish_progress`) are used to indicate
    the start and end of the process, providing a clear indication of the function's operation.

    Examples
    --------
    >>> all_clocks = show_all_clocks()
    Clock1
    Clock2
    Clock3
    ...

    """
    logger = LoggerManager.gen_logger("show_all_clocks")
    logger.first_info("Starting show_all_clocks function")

    # Load all metadata
    all_clock_metadata = load_clock_metadata(dir, logger, indent_level=1)

    # Message to indicate the start of the search process
    message = "Showing all available clock names"
    logger.start_progress(f"{message} started")
    all_clocks = sorted(all_clock_metadata.keys())
    for clock_name in all_clocks:
        logger.info(clock_name, indent_level=2)
    logger.finish_progress(f"{message} finished")

    logger.done()


def get_clock_metadata(clock_name: str, dir: str = "pyaging_data") -> None:
    """
    Retrieves and logs the metadata of a specified aging clock.

    This function accesses the metadata for a given aging clock and logs detailed
    information about it, such as the data type, model, and citation. It is designed
    to help users quickly understand the characteristics and details of a specific clock
    in the pyaging package. The function uses a logger to ensure that the output is
    structured and easily readable.

    Parameters
    ----------
    clock_name : str
        The name of the aging clock whose metadata is to be retrieved. The name is case-insensitive.
    dir : str
        The directory to deposit the downloaded file. Defaults to 'pyaging_data'.

    Returns
    -------
    None
        The function does not return a value but logs the metadata of the specified clock.

    Notes
    -----
    The function first calls `load_clock_metadata` to load all clock metadata. It then
    extracts the metadata for the specified clock and logs each piece of information.
    The logger's progress methods (`start_progress` and `finish_progress`) are used to
    indicate the start and end of the retrieval process, enhancing user understanding
    of the operation.

    This function assumes that the specified clock name exists in the metadata. If the
    clock name is not found, an error may occur.

    Examples
    --------
    >>> get_clock_metadata("clock1")
    name: Clock1
    data_type: methylation
    species: Homo sapiens
    ...

    """
    logger = LoggerManager.gen_logger("get_clock_metadata")
    logger.first_info("Starting get_clock_metadata function")

    # Load all metadata
    all_clock_metadata = load_clock_metadata(dir, logger, indent_level=1)

    # Lowercase clock name
    clock_name = clock_name.lower()
    clock_dict = all_clock_metadata[clock_name]

    # Message to indicate the start of the search process
    message = f"Showing {clock_name} metadata"
    logger.start_progress(f"{message} started")
    for key in list(clock_dict.keys()):
        logger.info(f"{key}: {clock_dict[key]}", indent_level=2)
    logger.finish_progress(f"{message} finished")

    logger.done()


def print_model_details(model, max_list_length=30, max_tensor_elements=30):
    """
    Prints detailed information about a PyTorch model, including its attributes, structure, and parameters.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to be inspected.

    max_list_length : int
        The maximum length of lists to print in full. Lists longer than this will be summarized.

    max_tensor_elements : int
        The maximum number of elements in a tensor to print in full. Tensors with more elements will be summarized.

    Notes
    -----
    The function outputs:
    - Model Attributes: Non-module, non-parameter attributes of the model, excluding private attributes (those starting with '_').
    - Model Structure: The structure of the model, showing layers and submodules.
    - Model Parameters and Weights: Parameters of the model, including weights and biases, with size and value information.
    """

    divider = "\n%==================================== Model Details ====================================%\n"

    def formatted_print(name, value):
        """
        Prints the name and value of an attribute or parameter, formatting lists and tensors for readability.

        For lists longer than max_list_length and tensors with more elements than max_tensor_elements, a summary is printed instead of the full value.
        """
        if isinstance(value, list) and len(value) > max_list_length:
            print(f"{name}: {value[:max_list_length]}... [Total elements: {len(value)}]")
        elif isinstance(value, torch.Tensor) and value.nelement() > max_tensor_elements:
            flattened_tensor = value.flatten()
            print(f"{name}: {flattened_tensor[:max_tensor_elements].tolist()}... [Tensor of shape {value.size()}]")
        else:
            print(f"{name}: {pformat(value)}")

    print(divider + "Model Attributes:\n")
    for name, value in model.__dict__.items():
        if (
            not isinstance(value, torch.nn.Module)
            and not isinstance(value, torch.nn.Parameter)
            and not name.startswith("_")
        ):
            formatted_print(name, value)

    print(divider + "Model Structure:\n")
    for name, module in model.named_children():
        print(f"{name}: {module}")

    print(divider + "Model Parameters and Weights:\n")
    for name, param in model.named_parameters():
        formatted_print(name, param.data)

    print(divider)


def is_newer_than_target(url, target_date_str):
    """
    Check if the 'Last-Modified' date of the metadata of a url is newer than a
    specific target date.

    Parameters
    ----------
    url : str
        The url of interest from S3. The header must include the 'Last-Modified'
        key with its value in the format: 'Day, DD Mon YYYY HH:MM:SS GMT'.
    target_date_str : str
        The target date as a string in the format 'YYYY-MM-DD'.

    Returns
    -------
    bool
        True if the 'Last-Modified' date is newer than the target date, False otherwise.

    Notes
    -----
    The function parses the 'Last-Modified' date from the provided metadata and compares
    it against a predefined target date (January 21st, 2024). The comparison accounts for
    the UTC timezone.

    Example
    -------
    metadata = {'Last-Modified': 'Sun, 21 Jan 2024 09:54:49 GMT'}
    result = is_newer_than_target(metadata)
    # result will be True if 'Last-Modified' is after Jan 21st, 2024, False otherwise.
    """

    response = requests.head(url)
    metadata = response.headers

    # Parse the Last-Modified timestamp
    last_modified_str = metadata["Last-Modified"]
    timestamp_format = "%a, %d %b %Y %H:%M:%S GMT"
    last_modified = datetime.strptime(last_modified_str, timestamp_format)
    last_modified = last_modified.replace(tzinfo=pytz.UTC)

    # Parse the target date
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
    target_date = target_date.replace(tzinfo=pytz.UTC)

    return last_modified > target_date
