from unittest.mock import Mock, call

import pytest
import torch

from pyaging.predict._pred_utils import load_clock
from pyaging.utils._hf import PyAgingDownloadError, PyAgingResourceNotFoundError
from pyaging.utils._utils import load_clock_metadata


def test_load_clock_downloads_lowercase_hf_file_and_prepares_model(monkeypatch, tmp_path):
    returned_path = str(tmp_path / "resolved" / "horvath2013.pt")
    download_hf_file = Mock(return_value=returned_path)
    model = Mock()
    torch_load = Mock(return_value=model)
    logger = Mock()
    monkeypatch.setattr("pyaging.predict._pred_utils.download_hf_file", download_hf_file)
    monkeypatch.setattr("pyaging.predict._pred_utils.torch.load", torch_load)

    result = load_clock("Horvath2013", "cuda", str(tmp_path), logger, indent_level=2)

    assert result is model
    download_hf_file.assert_called_once_with("horvath2013.pt", str(tmp_path), logger, indent_level=2)
    torch_load.assert_called_once_with(returned_path, weights_only=False)
    assert model.to.call_args_list == [call(torch.float64), call("cuda")]
    model.eval.assert_called_once_with()


def test_load_clock_translates_only_missing_resource_to_chained_name_error(monkeypatch, tmp_path):
    missing_error = PyAgingResourceNotFoundError("missing clock")
    logger = Mock()
    monkeypatch.setattr(
        "pyaging.predict._pred_utils.download_hf_file",
        Mock(side_effect=missing_error),
    )

    with pytest.raises(NameError, match="Clock horvath2013 is not available") as error:
        load_clock("Horvath2013", "cpu", str(tmp_path), logger, indent_level=2)

    assert error.value.__cause__ is missing_error
    logger.error.assert_called_once()
    assert "Clock horvath2013 is not available on pyaging" in logger.error.call_args.args[0]
    assert logger.error.call_args.kwargs == {"indent_level": 3}


def test_load_clock_propagates_download_error_unchanged(monkeypatch, tmp_path):
    download_error = PyAgingDownloadError("network unavailable")
    logger = Mock()
    monkeypatch.setattr(
        "pyaging.predict._pred_utils.download_hf_file",
        Mock(side_effect=download_error),
    )

    with pytest.raises(PyAgingDownloadError) as error:
        load_clock("Horvath2013", "cpu", str(tmp_path), logger, indent_level=2)

    assert error.value is download_error
    logger.error.assert_not_called()


def test_load_clock_metadata_downloads_hf_file_and_loads_returned_path(monkeypatch, tmp_path):
    returned_path = str(tmp_path / "resolved" / "all_clock_metadata.pt")
    download_hf_file = Mock(return_value=returned_path)
    metadata = {"horvath2013": {"year": 2013}}
    torch_load = Mock(return_value=metadata)
    logger = Mock()
    monkeypatch.setattr("pyaging.utils._utils.download_hf_file", download_hf_file)
    monkeypatch.setattr("pyaging.utils._utils.torch.load", torch_load)

    result = load_clock_metadata(str(tmp_path), logger, indent_level=2)

    assert result is metadata
    download_hf_file.assert_called_once_with("all_clock_metadata.pt", str(tmp_path), logger, indent_level=2)
    torch_load.assert_called_once_with(returned_path, weights_only=False)
