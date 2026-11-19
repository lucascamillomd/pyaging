"""Internal boundary for downloading pyaging data from Hugging Face.

Clock weights live in one repository per clock under the ``pyaging``
organization (``pyaging/<clock_name>``); shared assets (example data, the
aggregate metadata file) live in the shared data repository, which is also
the fallback whenever a clock's own repository cannot be read.

Set the ``PYAGING_DATA_REVISION`` environment variable to pin downloads to a
specific existing revision of the data repositories for reproducibility. Defaults to ``main``, the live data release.

A clock's weight file is resolved against the Hub once per process and
revision; later loads of that clock reuse the resolved local path without a
request. A long-running process therefore sees a new push to a mutable
revision such as ``main`` only after it restarts.
"""

import contextvars
import os
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress

import httpx
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from huggingface_hub.utils import tqdm

from ..logger._live import _hf_bars_disabled

REPO_ID = "lucascamillomd/pyaging-data"
CLOCKS_OWNER = "pyaging"
DEFAULT_REVISION = "main"

# Hugging Face repository names: no leading/trailing "-" or ".", no "--" or "..".
_CLOCK_NAME = re.compile(r"(?!.*(?:--|\.\.))[a-z0-9](?:[a-z0-9_.-]{0,94}[a-z0-9_])?")
_PREFETCH_WORKERS = 8

# (clock name, revision) -> resolved weight path, for the life of the process.
_resolved_weights: dict[tuple[str, str], str] = {}


class _SilentProgressBar(tqdm):
    """A per-download bar that honors pyaging's silent display context."""

    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def get_data_revision() -> str:
    """Resolve the data-repo revision, honoring PYAGING_DATA_REVISION at call time.

    An unset, empty or whitespace-only value means :data:`DEFAULT_REVISION`.
    """
    return os.environ.get("PYAGING_DATA_REVISION", "").strip() or DEFAULT_REVISION


class PyAgingHubError(RuntimeError):
    """Base error for failures at the pyaging Hugging Face boundary."""


class PyAgingResourceNotFoundError(PyAgingHubError):
    """Raised when a requested data file does not exist."""


class PyAgingRevisionNotFoundError(PyAgingHubError):
    """Raised when a requested data repository revision does not exist."""


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
    revision = get_data_revision()
    progress_options = {"tqdm_class": _SilentProgressBar} if _hf_bars_disabled.get() else {}
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            **progress_options,
        )
    except LocalEntryNotFoundError as exc:
        raise PyAgingDownloadError(f"Could not download data file '{filename}' and no local copy is available") from exc
    except EntryNotFoundError as exc:
        raise PyAgingResourceNotFoundError(f"Data file '{filename}' was not found in repository '{repo_id}'") from exc
    except RevisionNotFoundError as exc:
        raise PyAgingRevisionNotFoundError(
            f"Data revision '{revision}' was not found in repository '{repo_id}' for file '{filename}'"
        ) from exc
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
    except httpx.RequestError as exc:
        raise PyAgingDownloadError(f"Could not transfer data file '{filename}' from Hugging Face") from exc
    except OSError as exc:
        raise PyAgingDownloadError(f"Could not save data file '{filename}'") from exc

    if logger is not None:
        logger.info(f"Data available at {path}", indent_level=indent_level + 1)
    return path


def _clear_resolved_weights() -> None:
    """Forget every weight path resolved in this process."""
    _resolved_weights.clear()


def _normalize_clock_name(clock_name: str) -> str:
    """Return the canonical clock id, rejecting names no clock repository can have."""
    name = clock_name.strip().lower()
    if not _CLOCK_NAME.fullmatch(name):
        raise PyAgingResourceNotFoundError(f"'{clock_name}' is not a valid clock name")
    return name


def _remembered_weights(key: tuple[str, str]) -> str | None:
    path = _resolved_weights.get(key)
    # The cache may have been pruned since the path was resolved.
    return path if path is not None and os.path.isfile(path) else None


def download_clock_weights(clock_name: str, dir: str = "pyaging_data", logger=None, indent_level: int = 1) -> str:
    """Download a clock's weight file, preferring its dedicated repository.

    Each clock lives in ``pyaging/<clock_name>``; the shared data
    repository is the fallback when the per-clock repository cannot be read,
    including when it lacks the requested revision or, offline, has no cached
    copy. Both use the same revision. The repo's ``config.json`` is fetched
    alongside the weights - it carries the audited clock metadata and is the
    file the Hub counts as a download.

    The name is stripped and lowercased; a name that cannot be a clock id
    raises :class:`PyAgingResourceNotFoundError` without contacting the Hub.
    The resolved path is remembered for the life of the process per clock and
    revision, so repeated loads make no request and a new push to a mutable
    revision such as ``main`` is seen only after the process restarts.
    """
    clock_name = _normalize_clock_name(clock_name)
    key = (clock_name, get_data_revision())
    path = _remembered_weights(key)
    if path is not None:
        if logger is not None:
            logger.info(f"Data available at {path}", indent_level=indent_level + 1)
        return path
    path = _resolve_clock_weights(clock_name, dir, logger, indent_level)
    _resolved_weights[key] = path
    return path


def _resolve_clock_weights(clock_name: str, dir: str, logger, indent_level: int) -> str:
    clock_repo = f"{CLOCKS_OWNER}/{clock_name}"
    filename = f"{clock_name}.pt"
    try:
        path = download_hf_file(filename, dir, logger, indent_level=indent_level, repo_id=clock_repo)
    except (
        PyAgingRepositoryError,
        PyAgingResourceNotFoundError,
        PyAgingRevisionNotFoundError,
        PyAgingAuthenticationError,
    ):
        # The Hub answers 401 (not 404) for nonexistent repos when the caller
        # has no token, which maps to the authentication error - for anonymous
        # users a typoed clock name must still reach the shared-repository fallback and
        # its clock-not-available message.
        return download_hf_file(filename, dir, logger, indent_level=indent_level)
    except PyAgingDownloadError as exc:
        if not isinstance(exc.__cause__, LocalEntryNotFoundError):
            raise
        # Offline (or unreachable) without a cached per-clock copy: the shared
        # repository may still hold this file cached at the same revision.
        try:
            return download_hf_file(filename, dir, logger, indent_level=indent_level)
        except PyAgingHubError:
            raise exc from exc.__cause__
    with suppress(PyAgingHubError):
        download_hf_file("config.json", dir, repo_id=clock_repo)
    return path


def _prefetch_one(clock_name: str, dir: str) -> None:
    with suppress(Exception):
        download_clock_weights(clock_name, dir)


def prefetch_clock_weights(clock_names, dir: str = "pyaging_data") -> None:
    """Resolve several clocks' weight files concurrently before they are loaded.

    Never raises: a clock that cannot be resolved here is reported by the
    later serial :func:`download_clock_weights` call in its usual way.
    """
    try:
        if isinstance(clock_names, str):
            clock_names = [clock_names]
        revision = get_data_revision()
        pending = []
        for clock_name in clock_names:
            with suppress(Exception):
                name = _normalize_clock_name(clock_name)
                if name not in pending and _remembered_weights((name, revision)) is None:
                    pending.append(name)
        if len(pending) < 2:
            return
        with ThreadPoolExecutor(max_workers=min(_PREFETCH_WORKERS, len(pending))) as executor:
            for name in pending:
                # Each worker needs its own copy: a context cannot be entered by two
                # threads at once, and it carries the caller's quiet-progress-bar setting.
                executor.submit(contextvars.copy_context().run, _prefetch_one, name, dir)
    except Exception:
        return
