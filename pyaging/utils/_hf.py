"""Internal boundary for downloading pyaging data from Hugging Face."""

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
)

REPO_ID = "lucascamillomd/pyaging-data"
REVISION = "main"


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


def download_hf_file(filename: str, dir: str, logger=None, indent_level: int = 1) -> str:
    """Download a pyaging data file and return its resolved local path."""
    try:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            revision=REVISION,
            local_dir=str(dir),
        )
    except LocalEntryNotFoundError as exc:
        raise PyAgingDownloadError(f"Could not download data file '{filename}' and no local copy is available") from exc
    except EntryNotFoundError as exc:
        raise PyAgingResourceNotFoundError(f"Data file '{filename}' was not found in repository '{REPO_ID}'") from exc
    except RepositoryNotFoundError as exc:
        if getattr(exc.response, "status_code", None) in (401, 403):
            raise PyAgingAuthenticationError(f"Hugging Face denied access to data file '{filename}'") from exc
        raise PyAgingRepositoryError(f"Hugging Face repository '{REPO_ID}' could not be found") from exc
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
