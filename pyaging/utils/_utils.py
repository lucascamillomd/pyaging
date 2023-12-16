import os
import torch
import ntpath
import os
from urllib.request import urlretrieve
from functools import wraps

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
            indent_level = (
                kwargs["indent_level"] if "indent_level" in kwargs.keys() else 1
            )

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
    url = f"https://pyaging.s3.amazonaws.com/clocks/metadata/all_clock_metadata.pt"
    download(url, dir, logger, indent_level=indent_level)
    all_clock_metadata = torch.load(f"{dir}/all_clock_metadata.pt")
    return all_clock_metadata


def download(url: str, dir: str, logger, indent_level: int = 1):
    """
    Downloads a file from a specified URL to a local directory.

    This function checks if the file specified by the URL already exists in the local
    'pyaging_data' directory. If the file is not present, it downloads the file from the URL
    and saves it in this directory. The function logs the progress of the download, including
    whether the file is found locally or needs to be downloaded.

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

    if os.path.exists(file_path):
        logger.info(f"Data found in {file_path}", indent_level=indent_level + 1)
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
        logger.warning(
            f"{clock_name} is not currently available in pyaging", indent_level=2
        )

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
        The function does not return a value but logs the names of all available clocks.

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
    >>> show_all_clocks()
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
    for clock_name in list(all_clock_metadata.keys()):
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


def get_clock_weights(clock_name: str, dir: str = "pyaging_data") -> dict:
    """
    Loads the specified aging clock from a remote source and returns a dictionary with its
    components.

    This function downloads the weights and configuration of a specified aging clock from a
    remote server. It then loads and returns a dictionary with various components of the clock
    such as its features, preprocessing or postprocessing steps, etc.

    Parameters
    ----------
    clock_name : str
        The name of the aging clock whose metadata is to be retrieved. The name is case-insensitive.
    dir : str
        The directory to deposit the downloaded file. Defaults to 'pyaging_data'.

    Returns
    -------
    dict
        A dictionary containing the following components of the clock:
        - features: The features used by the clock.
        - reference_feature_values: Reference values for features in case features are missing.
        - weight_dict: A dictionary of weights used in the clock's model.
        - preprocessing: Any preprocessing steps required for the clock's input data.
        - postprocessing: Any postprocessing steps applied to the clock's output.
        - preprocessing_helper: Any preprocessing helper file.
        - postprocessing_helper: Any postprocessing helper file.

    Notes
    -----
    The clock's weights and configuration are assumed to be stored in a .pt (PyTorch) file
    on a remote server. The URL for the clock is constructed based on the clock's name.
    If the clock or its components are not found, the function may fail or return incomplete
    information.

    The logger is used extensively for progress tracking and information logging, enhancing
    transparency and user experience.

    Examples
    --------
    >>> clock_dict = get_clock_weights("clock1", "pyaging_data")

    """

    logger = LoggerManager.gen_logger("get_clock_weights")
    logger.first_info("Starting get_clock_weights function")

    # Load all metadata
    all_clock_metadata = load_clock_metadata(dir, logger, indent_level=1)

    # Lowercase clock name
    clock_name = clock_name.lower()
    clock_dict = all_clock_metadata[clock_name]

    # Check if clock name is available
    if clock_name not in all_clock_metadata.keys():
        logger.error(f"Clock {clock_name} is not yet available on pyaging")
        raise ValueError

    # Download weights
    url = f"https://pyaging.s3.amazonaws.com/clocks/weights/{clock_name}.pt"
    download(url, dir, logger, indent_level=1)

    # Define the path to the clock weights file
    weights_path = os.path.join(dir, f"{clock_name}.pt")

    # Load the clock dictionary from the file
    clock_dict = torch.load(weights_path)

    logger.done()

    return clock_dict
