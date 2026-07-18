from unittest.mock import Mock

import pytest
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
)

from pyaging.utils._hf import (
    PyAgingAuthenticationError,
    PyAgingDownloadError,
    PyAgingRateLimitError,
    PyAgingRepositoryError,
    PyAgingResourceNotFoundError,
    download_hf_file,
)


def test_download_hf_file_uses_pinned_repository_standard_cache_and_logs_path(monkeypatch, tmp_path):
    downloaded_path = str(tmp_path / "horvath2013.pt")
    hub_download = Mock(return_value=downloaded_path)
    logger = Mock()
    monkeypatch.setattr("pyaging.utils._hf.hf_hub_download", hub_download)

    result = download_hf_file("horvath2013.pt", str(tmp_path), logger, indent_level=2)

    assert result == downloaded_path
    hub_download.assert_called_once_with(
        repo_id="lucascamillomd/pyaging-data",
        filename="horvath2013.pt",
        revision="main",
    )
    logger.info.assert_called_once_with(f"Data available at {downloaded_path}", indent_level=3)


def test_download_hf_file_keeps_dir_argument_for_backward_compatibility(monkeypatch):
    hub_download = Mock(return_value="/hf-cache/horvath2013.pt")
    monkeypatch.setattr("pyaging.utils._hf.hf_hub_download", hub_download)

    download_hf_file("horvath2013.pt", "legacy-local-directory")

    assert "local_dir" not in hub_download.call_args.kwargs


def test_download_hf_file_maps_missing_entry_and_preserves_cause(monkeypatch, tmp_path):
    hub_error = EntryNotFoundError("missing entry")
    monkeypatch.setattr("pyaging.utils._hf.hf_hub_download", Mock(side_effect=hub_error))

    with pytest.raises(PyAgingResourceNotFoundError, match="horvath2013.pt") as error:
        download_hf_file("horvath2013.pt", str(tmp_path))

    assert error.value.__cause__ is hub_error


@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    [
        (401, PyAgingAuthenticationError),
        (403, PyAgingAuthenticationError),
        (429, PyAgingRateLimitError),
        (500, PyAgingDownloadError),
    ],
)
def test_download_hf_file_maps_http_errors_and_preserves_cause(
    monkeypatch,
    tmp_path,
    status_code,
    expected_error,
):
    response = Mock(status_code=status_code)
    hub_error = HfHubHTTPError("hub request failed", response=response)
    monkeypatch.setattr("pyaging.utils._hf.hf_hub_download", Mock(side_effect=hub_error))

    with pytest.raises(expected_error) as error:
        download_hf_file("horvath2013.pt", str(tmp_path))

    assert error.value.__cause__ is hub_error


@pytest.mark.parametrize(
    ("error_type", "status_code", "expected_error"),
    [
        (RepositoryNotFoundError, 401, PyAgingAuthenticationError),
        (GatedRepoError, 403, PyAgingAuthenticationError),
        (RepositoryNotFoundError, 404, PyAgingRepositoryError),
    ],
)
def test_download_hf_file_maps_repository_errors_by_status_and_preserves_cause(
    monkeypatch,
    tmp_path,
    error_type,
    status_code,
    expected_error,
):
    hub_error = error_type("repository request failed", response=Mock(status_code=status_code))
    monkeypatch.setattr("pyaging.utils._hf.hf_hub_download", Mock(side_effect=hub_error))

    with pytest.raises(expected_error) as error:
        download_hf_file("horvath2013.pt", str(tmp_path))

    assert error.value.__cause__ is hub_error


@pytest.mark.parametrize(
    "source_error",
    [LocalEntryNotFoundError("local cache miss"), OSError("disk unavailable")],
)
def test_download_hf_file_maps_local_failures_and_preserves_cause(monkeypatch, tmp_path, source_error):
    monkeypatch.setattr("pyaging.utils._hf.hf_hub_download", Mock(side_effect=source_error))

    with pytest.raises(PyAgingDownloadError) as error:
        download_hf_file("horvath2013.pt", str(tmp_path))

    assert error.value.__cause__ is source_error
