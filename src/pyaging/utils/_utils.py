import os
from functools import wraps
from pprint import pformat
from urllib.request import urlretrieve

import torch

from ..logger._live import MUTED, display_enabled, get_console, live_step
from ._hf import download_hf_file


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
            indent_level = kwargs.get("indent_level", 1)

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
    Loads the clock metadata from Hugging Face.

    This function downloads or resolves the metadata file from the standard Hugging Face
    cache, then reads and returns the metadata.

    Parameters
    ----------
    dir : str
        Retained for backward compatibility. Hugging Face files use its standard cache.
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
    The function retrieves the metadata file from the pyaging Hugging Face data repository.
    It is decorated with `@progress`, which adds start and end log messages for this process.

    Examples
    --------
    >>> logger = pyaging.logger.LoggerManager.gen_logger("example")
    >>> metadata = load_clock_metadata("pyaging_data", logger)
    >>> type(metadata)
    <class 'dict'>

    """
    metadata_path = download_hf_file("all_clock_metadata.pt", dir, logger, indent_level=indent_level)
    all_clock_metadata = torch.load(metadata_path, weights_only=False)
    return all_clock_metadata


def download(url: str, dir: str, logger, indent_level: int = 1) -> None:
    """Download a remote file to a flat local directory unless it is already cached.

    Parameters
    ----------
    url : str
        The URL of the file to download.
    dir : str
        The directory in which to save the downloaded file.
    logger : object
        Logger object used to report cache and download progress.
    indent_level : int, optional
        The indentation level for logging messages, by default 1.

    Notes
    -----
    The local filename is the basename of the URL. Existing local files are reused without
    contacting the remote host.
    """
    file_path = os.path.join(dir, url.split("/")[-1])

    if os.path.exists(file_path):
        logger.info(f"Data found in {file_path}", indent_level=indent_level + 1)
        return

    if not os.path.exists(dir):
        os.mkdir(dir)
    logger.info(f"Downloading data to {file_path}", indent_level=indent_level + 1)
    logger.indent_level = indent_level + 1
    urlretrieve(url, file_path, reporthook=logger.request_report_hook)


def find_clock_by_doi(search_doi: str, dir: str = "pyaging_data", verbose: bool = True) -> None:
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
        Retained for backward compatibility. Hugging Face files use its standard cache.

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
    with live_step("searching clocks by DOI", verbose) as (step, pipeline_logger):
        all_clock_metadata = load_clock_metadata(dir, pipeline_logger, indent_level=1)
        matching_clocks = [name for name, meta in all_clock_metadata.items() if meta.get("doi") == search_doi]
        if matching_clocks:
            step.done(f"clocks with DOI {search_doi}: {', '.join(matching_clocks)}")
        else:
            step.warn(f"no clocks found with DOI {search_doi}")
            step.done(f"searched {len(all_clock_metadata)} clocks")


def cite_clock(clock_name: str, dir: str = "pyaging_data", verbose: bool = True) -> None:
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
        Retained for backward compatibility. Hugging Face files use its standard cache.

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
    clock_name = clock_name.lower()
    pyaging_citation = (
        'de Lima Camillo, Lucas Paulo. "pyaging: a Python-based compendium of '
        'GPU-optimized aging clocks." bioRxiv (2023): 2023-11.'
    )

    with live_step(f"looking up citation for {clock_name}", verbose) as (step, pipeline_logger):
        all_clock_metadata = load_clock_metadata(dir, pipeline_logger, indent_level=1)
        clock_dict = all_clock_metadata.get(clock_name)
        citation = clock_dict.get("citation") if clock_dict else None
        if citation:
            step.done(f"citation for {clock_name}")
        elif clock_dict:
            step.warn(f"citation not found in {clock_name}")
            step.done(clock_name)
        else:
            step.warn(f"{clock_name} is not currently available in pyaging")
            step.done("clock not found")

    if display_enabled(verbose) and citation:
        console = get_console()
        console.print(f"  {citation}")
        console.print(f"  Please also consider citing pyaging: {pyaging_citation}", style=MUTED)


def show_all_clocks(dir: str = "pyaging_data", verbose: bool = True) -> None:
    """
    Displays the names of all aging clocks available in the metadata.

    This function retrieves the metadata for all aging clocks and logs each clock's name.
    It's useful for users to get a quick overview of all the clocks included in the pyaging
    package. The function utilizes a logger for structured output, providing clarity and
    readability in its logs.

    Parameters
    ----------
    dir : str
        Retained for backward compatibility. Hugging Face files use its standard cache.

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
    with live_step("loading clock metadata", verbose) as (step, pipeline_logger):
        all_clock_metadata = load_clock_metadata(dir, pipeline_logger, indent_level=1)
        all_clocks = sorted(all_clock_metadata.keys())
        step.done(f"{len(all_clocks)} clocks available")

    if display_enabled(verbose):
        from rich.columns import Columns

        get_console().print(Columns(all_clocks, padding=(0, 2)))


def get_clock_metadata(clock_name: str, dir: str = "pyaging_data", verbose: bool = True) -> None:
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
        Retained for backward compatibility. Hugging Face files use its standard cache.

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
    clock_name = clock_name.lower()
    with live_step(f"loading {clock_name} metadata", verbose) as (step, pipeline_logger):
        all_clock_metadata = load_clock_metadata(dir, pipeline_logger, indent_level=1)
        clock_dict = all_clock_metadata[clock_name]
        step.done(f"{clock_name} metadata")

    if display_enabled(verbose):
        from rich.table import Table

        grid = Table.grid(padding=(0, 2))
        for key, value in clock_dict.items():
            grid.add_row(f"[{MUTED}]{key}[/]", str(value))
        get_console().print(grid)


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

    - Model Attributes: Non-module, non-parameter attributes of the model, excluding private
      attributes (those starting with '_').
    - Model Structure: The structure of the model, showing layers and submodules.
    - Model Parameters and Weights: Parameters of the model, including weights and biases, with
      size and value information.
    """

    divider = "\n%==================================== Model Details ====================================%\n"

    def formatted_print(name, value):
        """
        Prints the name and value of an attribute or parameter, formatting lists and tensors for readability.

        For lists longer than max_list_length and tensors with more elements than
        max_tensor_elements, a summary is printed instead of the full value.
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
