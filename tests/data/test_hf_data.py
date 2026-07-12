from unittest.mock import Mock

import pytest

import pyaging.data as public_data
import pyaging.data._data as data_module


@pytest.mark.parametrize(
    ("data_type", "filename"),
    [
        ("GSE130735", "GSE130735_subset.pkl"),
        ("GSE193140", "GSE193140.pkl"),
        ("GSE139307", "GSE139307.pkl"),
        ("GSE223748", "GSE223748_subset.pkl"),
        ("ENCFF386QWG", "ENCFF386QWG.bigWig"),
        ("GSE65765", "GSE65765_CPM.pkl"),
        ("blood_chemistry_example", "blood_chemistry_example.pkl"),
    ],
)
def test_download_example_data_uses_hf_filename_and_caller_directory(
    monkeypatch,
    tmp_path,
    data_type,
    filename,
):
    logger = Mock()
    logger_manager = Mock()
    logger_manager.gen_logger.return_value = logger
    download_hf_file = Mock()
    caller_dir = str(tmp_path / "example-data")
    monkeypatch.setattr(data_module, "LoggerManager", logger_manager)
    monkeypatch.setattr(data_module, "download_hf_file", download_hf_file)

    data_module.download_example_data(data_type, dir=caller_dir, verbose=True)

    logger_manager.gen_logger.assert_called_once_with("download_example_data")
    download_hf_file.assert_called_once_with(filename, caller_dir, logger, indent_level=1)
    logger.done.assert_called_once_with()


def test_example_data_filename_mapping_is_not_public():
    assert not hasattr(public_data, "EXAMPLE_DATA_FILENAMES")


def test_download_example_data_rejects_unknown_data_type(monkeypatch):
    logger = Mock()
    logger_manager = Mock()
    logger_manager.gen_logger.return_value = logger
    monkeypatch.setattr(data_module, "LoggerManager", logger_manager)

    with pytest.raises(ValueError):
        data_module.download_example_data("not-implemented")
