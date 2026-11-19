import pickle
from unittest.mock import Mock

import pytest
import torch

import pyaging.logger._live as live_module
import pyaging.utils._utils as utils_module
from pyaging.utils import cite_clock, find_clock_by_doi, get_clock_metadata, load_clock_metadata

METADATA = {
    "horvath2013": {
        "clock_name": "horvath2013",
        "year": 2013,
        "doi": "https://doi.org/10.1186/gb-2013-14-10-r115",
        "citation": "Horvath, S. (2013).",
    },
    "hannum": {"clock_name": "hannum", "year": 2013, "doi": "https://doi.org/10.1016/j.molcel.2012.10.016"},
}


def _display_output():
    return live_module._console.file.getvalue()


@pytest.fixture
def stored_metadata(monkeypatch, tmp_path):
    """Serve METADATA from a real file, counting how often it is resolved."""
    path = tmp_path / "all_clock_metadata.pt"
    torch.save(METADATA, path)
    download_hf_file = Mock(return_value=str(path))
    monkeypatch.setattr(utils_module, "download_hf_file", download_hf_file)
    monkeypatch.delenv("PYAGING_DATA_REVISION", raising=False)
    return download_hf_file


@pytest.mark.parametrize(
    "search_doi",
    [
        "10.1186/gb-2013-14-10-r115",
        "https://doi.org/10.1186/gb-2013-14-10-r115",
        "http://dx.doi.org/10.1186/gb-2013-14-10-r115",
        "doi:10.1186/gb-2013-14-10-r115",
        " 10.1186/GB-2013-14-10-R115 ",
    ],
)
def test_find_clock_by_doi_accepts_bare_url_and_differently_cased_dois(stored_metadata, search_doi):
    find_clock_by_doi(search_doi)

    output = _display_output()
    assert "horvath2013" in output
    assert "no clocks found" not in output


def test_find_clock_by_doi_still_reports_no_match(stored_metadata):
    find_clock_by_doi("10.1000/xyz123")

    assert "no clocks found with DOI 10.1000/xyz123" in _display_output()


def test_get_clock_metadata_reports_unknown_clock_like_cite_clock(stored_metadata):
    get_clock_metadata("nosuchclock")

    assert "nosuchclock is not currently available in pyaging" in _display_output()


@pytest.mark.parametrize("helper", [get_clock_metadata, cite_clock])
def test_clock_helpers_ignore_case_and_surrounding_whitespace(stored_metadata, helper):
    helper(" Horvath2013 ")

    output = _display_output()
    assert "horvath2013" in output
    assert "not currently available" not in output


def test_load_clock_metadata_is_resolved_once_per_process_and_revision(stored_metadata, monkeypatch):
    logger = Mock()

    first = load_clock_metadata("pyaging_data", logger)
    second = load_clock_metadata("pyaging_data", logger)
    monkeypatch.setenv("PYAGING_DATA_REVISION", "v0.5.7")
    pinned = load_clock_metadata("pyaging_data", logger)

    assert first == second == pinned == METADATA
    assert stored_metadata.call_count == 2


def test_load_clock_metadata_cache_survives_caller_mutation(stored_metadata):
    first = load_clock_metadata("pyaging_data", Mock())
    first["horvath2013"]["year"] = 0
    first.pop("hannum")

    assert load_clock_metadata("pyaging_data", Mock()) == METADATA


def test_load_clock_metadata_resolves_again_when_cached_file_disappears(stored_metadata, tmp_path):
    load_clock_metadata("pyaging_data", Mock())
    replacement = tmp_path / "replacement.pt"
    torch.save({"hannum": METADATA["hannum"]}, replacement)
    (tmp_path / "all_clock_metadata.pt").unlink()
    stored_metadata.return_value = str(replacement)

    assert list(load_clock_metadata("pyaging_data", Mock())) == ["hannum"]


class _NotPlainData:
    pass


def test_load_clock_metadata_refuses_arbitrary_pickled_objects(monkeypatch, tmp_path):
    path = tmp_path / "all_clock_metadata.pt"
    torch.save({"horvath2013": {"payload": _NotPlainData()}}, path)
    monkeypatch.setattr(utils_module, "download_hf_file", Mock(return_value=str(path)))

    with pytest.raises(pickle.UnpicklingError):
        load_clock_metadata("pyaging_data", Mock())
