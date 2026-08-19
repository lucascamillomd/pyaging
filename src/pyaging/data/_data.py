import shutil
from pathlib import Path

from ..logger._live import SimpleStep, display_enabled, live_step, quiet_hf_bars
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

    verbose : bool
        Whether to show the progress display and warnings. Animated in
        notebooks and terminals, a plain summary when output is captured,
        and fully silent when False. Defaults to True.

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
    if data_type not in _EXAMPLE_DATA_FILENAMES:
        raise ValueError(f"Example data {data_type} has not yet been implemented in pyaging.")

    enabled = display_enabled(verbose)
    filename = _EXAMPLE_DATA_FILENAMES[data_type]
    destination = Path(dir) / filename
    if destination.exists():
        SimpleStep(filename, enabled=enabled).done(f"example data already at {destination}")
        return str(destination)

    with quiet_hf_bars(verbose), live_step(f"downloading {filename}", verbose) as (step, pipeline_logger):
        cache_path = download_hf_file(filename, dir, pipeline_logger, indent_level=1)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(cache_path, destination)
        step.done(f"example data at {destination}")
    return str(destination)
