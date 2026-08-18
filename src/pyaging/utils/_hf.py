"""Internal boundary for downloading pyaging data from Hugging Face.

Clock weights live in one repository per clock under the ``pyaging``
organization (``pyaging/<clock_name>``); shared assets (example data, the
aggregate metadata file) live in the legacy data repository, which also
serves as a fallback for clocks that have not been migrated yet.

Set the ``PYAGING_DATA_REVISION`` environment variable to pin downloads to a
specific revision of the data repositories (e.g. a release tag such as
``v0.3.1``) for reproducibility. Defaults to ``main``, the live data release.
"""

import os
from contextlib import suppress

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
)

REPO_ID = "lucascamillomd/pyaging-data"
CLOCKS_OWNER = "pyaging"
DEFAULT_REVISION = "main"


def get_data_revision() -> str:
    """Resolve the data-repo revision, honoring PYAGING_DATA_REVISION at call time."""
    return os.environ.get("PYAGING_DATA_REVISION", DEFAULT_REVISION)


class PyAgingHubError(RuntimeError):
    """Base error for failures at the pyaging Hugging Face boundary."""


class PyAgingResourceNotFoundError(PyAgingHubError):
    """Raised when a requested data file does not exist."""


class PyAgingRepositoryError(PyAgingHubError):
    """Raised when the pyaging data repository cannot be found."""


class PyAgingAuthenticationError(PyAgingHubError):
    """Raised when Hugging Face rejects authentication or authorization."""


class PyAgingRateLimitError(PyAgingHubError):
    """Raised when Hugging Face rate-limits a download."""


class PyAgingDownloadError(PyAgingHubError):
    """Raised when a data file cannot be downloaded."""


def download_hf_file(
    filename: str, dir: str = "pyaging_data", logger=None, indent_level: int = 1, repo_id: str = REPO_ID
) -> str:
    """Download a pyaging data file to the standard Hugging Face cache.

    ``dir`` is retained for backward compatibility but is no longer used.
    """
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=get_data_revision(),
        )
    except LocalEntryNotFoundError as exc:
        raise PyAgingDownloadError(f"Could not download data file '{filename}' and no local copy is available") from exc
    except EntryNotFoundError as exc:
        raise PyAgingResourceNotFoundError(f"Data file '{filename}' was not found in repository '{repo_id}'") from exc
    except RepositoryNotFoundError as exc:
        if getattr(exc.response, "status_code", None) in (401, 403):
            raise PyAgingAuthenticationError(f"Hugging Face denied access to data file '{filename}'") from exc
        raise PyAgingRepositoryError(f"Hugging Face repository '{repo_id}' could not be found") from exc
    except HfHubHTTPError as exc:
        status_code = getattr(exc.response, "status_code", None)
        if status_code in (401, 403):
            raise PyAgingAuthenticationError(f"Hugging Face denied access to data file '{filename}'") from exc
        if status_code == 429:
            raise PyAgingRateLimitError(f"Hugging Face rate-limited the download of data file '{filename}'") from exc
        raise PyAgingDownloadError(f"Hugging Face could not download data file '{filename}'") from exc
    except OSError as exc:
        raise PyAgingDownloadError(f"Could not save data file '{filename}'") from exc

    if logger is not None:
        logger.info(f"Data available at {path}", indent_level=indent_level + 1)
    return path


def download_clock_weights(clock_name: str, dir: str = "pyaging_data", logger=None, indent_level: int = 1) -> str:
    """Download a clock's weight file, preferring its dedicated repository.

    Each clock lives in ``pyaging/<clock_name>``; the legacy shared data
    repository is used as a fallback for clocks that are not migrated yet.
    The repo's ``config.json`` is fetched alongside the weights - it carries
    the audited clock metadata and is the file the Hub counts as a download.
    """
    clock_repo = f"{CLOCKS_OWNER}/{clock_name}"
    filename = f"{clock_name}.pt"
    try:
        path = download_hf_file(filename, dir, logger, indent_level=indent_level, repo_id=clock_repo)
    except (PyAgingRepositoryError, PyAgingResourceNotFoundError):
        return download_hf_file(filename, dir, logger, indent_level=indent_level)
    with suppress(PyAgingHubError):
        download_hf_file("config.json", dir, repo_id=clock_repo)
    return path
