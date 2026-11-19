"""Clock-name validation, offline fallback and process-lifetime weight resolution."""

import os
import threading
from pathlib import Path
from unittest.mock import Mock

import huggingface_hub.constants
import pytest
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError, RepositoryNotFoundError

from pyaging.logger._live import quiet_hf_bars
from pyaging.utils._hf import (
    PyAgingDownloadError,
    PyAgingResourceNotFoundError,
    download_clock_weights,
    get_data_revision,
    prefetch_clock_weights,
)

SHARED_REPO = "lucascamillomd/pyaging-data"


@pytest.fixture(autouse=True)
def default_revision(monkeypatch):
    monkeypatch.delenv("PYAGING_DATA_REVISION", raising=False)


def _hub(tmp_path, fail=()):
    """A fake ``hf_hub_download`` that serves real files and records each request."""
    calls = []
    lock = threading.Lock()

    def download(repo_id, filename, revision, **kwargs):
        with lock:
            calls.append((repo_id, filename, revision, kwargs))
        for clock_repo, error in fail:
            if repo_id == clock_repo:
                raise error
        path = tmp_path / repo_id.replace("/", "--") / revision / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weights")
        return str(path)

    download.calls = calls
    return download


@pytest.mark.parametrize("value", ["", "   ", "\t"])
def test_blank_data_revision_means_the_default(monkeypatch, value):
    monkeypatch.setenv("PYAGING_DATA_REVISION", value)

    assert get_data_revision() == "main"


def test_data_revision_ignores_surrounding_whitespace(monkeypatch):
    monkeypatch.setenv("PYAGING_DATA_REVISION", " v0.5.7\n")

    assert get_data_revision() == "v0.5.7"


def test_clock_name_is_stripped_and_lowercased_before_reaching_the_hub(monkeypatch, tmp_path):
    hub = _hub(tmp_path)
    monkeypatch.setattr("pyaging.utils._hf.hf_hub_download", hub)

    download_clock_weights(" Horvath2013 ")

    assert {(repo, filename) for repo, filename, _, _ in hub.calls} == {
        ("pyaging/horvath2013", "horvath2013.pt"),
        ("pyaging/horvath2013", "config.json"),
    }


@pytest.mark.parametrize("name", ["Horvath 2013", "../hannum", "pyaging/horvath2013", "", "-x", "a--b", "a..b", "x."])
def test_implausible_clock_name_is_reported_missing_without_a_request(monkeypatch, name):
    hub = Mock(side_effect=AssertionError("must not contact the Hub"))
    monkeypatch.setattr("pyaging.utils._hf.hf_hub_download", hub)

    with pytest.raises(PyAgingResourceNotFoundError):
        download_clock_weights(name)

    hub.assert_not_called()


def test_load_clock_turns_implausible_name_into_clock_not_available(monkeypatch):
    from pyaging.predict import load_clock

    monkeypatch.setattr("pyaging.utils._hf.hf_hub_download", Mock(side_effect=AssertionError("no request")))

    with pytest.raises(NameError, match="Clock horvath 2013 is not available"):
        load_clock("Horvath 2013", verbose=False)


def test_offline_cache_miss_in_clock_repo_falls_back_to_shared_repo(monkeypatch, tmp_path):
    offline_miss = LocalEntryNotFoundError("not cached")
    hub = _hub(tmp_path, fail=[("pyaging/altumage", offline_miss)])
    monkeypatch.setattr("pyaging.utils._hf.hf_hub_download", hub)
    monkeypatch.setenv("PYAGING_DATA_REVISION", "v0.5.7")

    path = download_clock_weights("altumage")

    assert path.endswith("altumage.pt")
    assert [(repo, revision) for repo, _, revision, _ in hub.calls] == [
        ("pyaging/altumage", "v0.5.7"),
        (SHARED_REPO, "v0.5.7"),
    ]


def test_offline_cache_miss_everywhere_raises_the_clock_repo_error(monkeypatch):
    clock_miss = LocalEntryNotFoundError("clock repo not cached")
    shared_miss = LocalEntryNotFoundError("shared repo not cached")
    monkeypatch.setattr("pyaging.utils._hf.hf_hub_download", Mock(side_effect=[clock_miss, shared_miss]))

    with pytest.raises(PyAgingDownloadError) as error:
        download_clock_weights("altumage")

    assert error.value.__cause__ is clock_miss


def test_online_download_failure_does_not_try_the_shared_repo(monkeypatch):
    server_error = HfHubHTTPError("server error", response=Mock(status_code=500))
    hub = Mock(side_effect=server_error)
    monkeypatch.setattr("pyaging.utils._hf.hf_hub_download", hub)

    with pytest.raises(PyAgingDownloadError):
        download_clock_weights("altumage")

    assert hub.call_count == 1


def test_offline_pinned_revision_uses_shared_repo_from_a_real_cache(monkeypatch, tmp_path):
    commit = "0" * 40
    repo_dir = tmp_path / "models--lucascamillomd--pyaging-data"
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "refs" / "v0.5.7").write_text(commit)
    snapshot = repo_dir / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (snapshot / "altumage.pt").write_bytes(b"weights")
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setenv("PYAGING_DATA_REVISION", "v0.5.7")

    path = download_clock_weights("altumage")

    assert Path(path).read_bytes() == b"weights"


def test_second_resolution_in_a_process_makes_no_request(monkeypatch, tmp_path):
    hub = _hub(tmp_path)
    monkeypatch.setattr("pyaging.utils._hf.hf_hub_download", hub)

    first = download_clock_weights("horvath2013")
    requests_after_first = len(hub.calls)
    second = download_clock_weights("Horvath2013")

    assert first == second
    assert requests_after_first == 2
    assert len(hub.calls) == 2


def test_config_is_requested_once_per_clock_and_revision(monkeypatch, tmp_path):
    hub = _hub(tmp_path)
    monkeypatch.setattr("pyaging.utils._hf.hf_hub_download", hub)

    download_clock_weights("horvath2013")
    download_clock_weights("horvath2013")
    monkeypatch.setenv("PYAGING_DATA_REVISION", "v0.5.7")
    download_clock_weights("horvath2013")
    download_clock_weights("horvath2013")

    configs = [revision for _, filename, revision, _ in hub.calls if filename == "config.json"]
    assert configs == ["main", "v0.5.7"]


def test_resolution_is_repeated_when_the_cached_file_disappears(monkeypatch, tmp_path):
    hub = _hub(tmp_path)
    monkeypatch.setattr("pyaging.utils._hf.hf_hub_download", hub)

    path = download_clock_weights("horvath2013")
    os.remove(path)
    assert download_clock_weights("horvath2013") == path
    assert [filename for _, filename, _, _ in hub.calls].count("horvath2013.pt") == 2


def test_failed_resolution_is_not_remembered(monkeypatch, tmp_path):
    hub = _hub(tmp_path)
    failure = HfHubHTTPError("server error", response=Mock(status_code=500))
    monkeypatch.setattr("pyaging.utils._hf.hf_hub_download", Mock(side_effect=failure))
    with pytest.raises(PyAgingDownloadError):
        download_clock_weights("horvath2013")

    monkeypatch.setattr("pyaging.utils._hf.hf_hub_download", hub)

    assert download_clock_weights("horvath2013").endswith("horvath2013.pt")


def test_prefetch_resolves_clocks_concurrently_and_later_loads_make_no_request(monkeypatch, tmp_path):
    names = ["horvath2013", "hannum", "altumage"]
    # Each weight request waits for the others: a serial prefetch breaks the barrier.
    barrier = threading.Barrier(len(names), timeout=5)
    hub = _hub(tmp_path)

    def concurrent_download(repo_id, filename, revision, **kwargs):
        if filename.endswith(".pt"):
            barrier.wait()
        return hub(repo_id=repo_id, filename=filename, revision=revision, **kwargs)

    monkeypatch.setattr("pyaging.utils._hf.hf_hub_download", concurrent_download)

    prefetch_clock_weights(names)
    requests_after_prefetch = len(hub.calls)
    for name in names:
        download_clock_weights(name)

    assert requests_after_prefetch == 2 * len(names)
    assert len(hub.calls) == requests_after_prefetch


def test_prefetch_swallows_failures_and_the_serial_call_still_raises(monkeypatch, tmp_path):
    missing = RepositoryNotFoundError("missing", response=Mock(status_code=404))
    hub = _hub(tmp_path, fail=[("pyaging/nosuchclock", missing), (SHARED_REPO, missing)])
    monkeypatch.setattr("pyaging.utils._hf.hf_hub_download", hub)

    prefetch_clock_weights(["nosuchclock", "horvath2013", "Horvath 2013"])

    with pytest.raises(Exception, match="could not be found"):
        download_clock_weights("nosuchclock")
    calls_before = len(hub.calls)
    download_clock_weights("horvath2013")
    assert len(hub.calls) == calls_before


@pytest.mark.parametrize("clock_names", [[], ["horvath2013"], "horvath2013", ["horvath2013", "HORVATH2013 "]])
def test_prefetch_does_nothing_for_fewer_than_two_unresolved_clocks(monkeypatch, clock_names):
    hub = Mock(side_effect=AssertionError("must not contact the Hub"))
    monkeypatch.setattr("pyaging.utils._hf.hf_hub_download", hub)

    prefetch_clock_weights(clock_names)

    hub.assert_not_called()


def test_prefetch_skips_clocks_already_resolved(monkeypatch, tmp_path):
    hub = _hub(tmp_path)
    monkeypatch.setattr("pyaging.utils._hf.hf_hub_download", hub)
    download_clock_weights("horvath2013")
    download_clock_weights("hannum")
    hub.calls.clear()

    prefetch_clock_weights(["horvath2013", "hannum", "altumage"])

    assert hub.calls == []


@pytest.mark.parametrize("clock_names", [None, 42, [None, 3], [object(), object()]])
def test_prefetch_never_raises(monkeypatch, clock_names):
    monkeypatch.setattr("pyaging.utils._hf.hf_hub_download", Mock(side_effect=RuntimeError("boom")))

    assert prefetch_clock_weights(clock_names) is None


def test_prefetch_workers_keep_quiet_progress_bars(monkeypatch, tmp_path):
    hub = _hub(tmp_path)
    monkeypatch.setattr("pyaging.utils._hf.hf_hub_download", hub)

    with quiet_hf_bars(False):
        prefetch_clock_weights(["horvath2013", "hannum"])

    assert len(hub.calls) == 4
    assert all("tqdm_class" in kwargs for _, _, _, kwargs in hub.calls)
