import shutil
from pathlib import Path

from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars

from ..logger import LoggerManager, silence_logger
from ..logger._live import DisplayLogger, SimpleStep, live_display_enabled
from ..utils._hf import download_hf_file

_EXAMPLE_DATA_FILENAMES = {
    "GSE130735": "GSE130735_subset.pkl",
    "GSE193140": "GSE193140.pkl",
    "GSE139307": "GSE139307.pkl",
    "GSE223748": "GSE223748_subset.pkl",
    "ENCFF386QWG": "ENCFF386QWG.bigWig",
    "GSE65765": "GSE65765_CPM.pkl",
    "blood_chemistry_example": "blood_chemistry_example.pkl",
}


def download_example_data(data_type: str, dir: str = "pyaging_data", verbose: bool = True) -> str:
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
        Directory where the example file is placed (default "pyaging_data"). The download
        itself goes through the standard Hugging Face cache and is then copied here.

    verbose : int or bool
        Whether to show progress and warnings. True shows a live display with
        progress bars in interactive runs (classic text logs otherwise);
        False is silent. Defaults to True.

    Raises
    ------
    ValueError
        If the specified data_type is not implemented, a ValueError is raised with a message suggesting
        the user to request its implementation.

    Notes
    -----
    The function maps the specified data_type to its corresponding filename in the public pyaging
    Hugging Face data repository. The datasets represent typical data formats and structures used in
    aging research.


    Examples
    --------
    >>> download_example_data("methylation")
    >>> # This will download the example methylation dataset to the local system.

    """
    logger = LoggerManager.gen_logger("download_example_data")
    live = live_display_enabled(verbose)
    if not verbose or live:
        silence_logger("download_example_data")
    logger.first_info("Starting download_example_data function")

    if data_type not in _EXAMPLE_DATA_FILENAMES:
        logger.error(
            f"Example data {data_type} has not yet been implemented in pyaging.",
            indent_level=2,
        )
        raise ValueError

    filename = _EXAMPLE_DATA_FILENAMES[data_type]
    destination = Path(dir) / filename
    if destination.exists():
        if live:
            SimpleStep(filename).done(f"example data already at {destination}")
        logger.info(f"Example data already exists at {destination}", indent_level=2)
        logger.done()
        return str(destination)

    hf_bars_were_enabled = live and not are_progress_bars_disabled()
    if hf_bars_were_enabled:
        disable_progress_bars()
    try:
        if live:
            with SimpleStep(f"downloading {filename}") as step:
                pipeline_logger = DisplayLogger(step.warn)
                cache_path = download_hf_file(filename, dir, pipeline_logger, indent_level=1)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(cache_path, destination)
                step.done(f"example data at {destination}")
        else:
            cache_path = download_hf_file(filename, dir, logger, indent_level=1)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(cache_path, destination)
    finally:
        if hf_bars_were_enabled:
            enable_progress_bars()
    logger.info(f"Example data available at {destination}", indent_level=2)
    logger.done()
    return str(destination)
