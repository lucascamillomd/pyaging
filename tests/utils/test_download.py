from unittest.mock import Mock

import pytest

import pyaging.utils as utils
import pyaging.utils._utils as utils_module


def test_download_is_public_and_uses_cached_flat_basename(monkeypatch, tmp_path):
    cached_path = tmp_path / "metadata.csv"
    cached_path.write_text("cached")
    logger = Mock()
    urlretrieve = Mock()
    monkeypatch.setattr(utils_module, "urlretrieve", urlretrieve)

    utils.download(
        "https://data.example.org/nested/metadata.csv",
        str(tmp_path),
        logger,
        indent_level=2,
    )

    logger.info.assert_called_once_with(f"Data found in {cached_path}", indent_level=3)
    urlretrieve.assert_not_called()


def test_interrupted_download_does_not_poison_cache(monkeypatch, tmp_path):
    def interrupted_download(url, filename, reporthook):
        from pathlib import Path

        Path(filename).write_bytes(b"partial")
        raise OSError("connection lost")

    monkeypatch.setattr(utils_module, "urlretrieve", interrupted_download)

    with pytest.raises(OSError, match="connection lost"):
        utils.download("https://data.example.org/metadata.csv", str(tmp_path), Mock())

    assert list(tmp_path.iterdir()) == []


def test_download_uses_url_path_and_creates_nested_directory(monkeypatch, tmp_path):
    def complete_download(url, filename, reporthook):
        from pathlib import Path

        Path(filename).write_bytes(b"complete")

    monkeypatch.setattr(utils_module, "urlretrieve", complete_download)
    destination = tmp_path / "nested" / "data"

    utils.download("https://data.example.org/metadata.csv?download=1#table", str(destination), Mock())

    assert (destination / "metadata.csv").read_bytes() == b"complete"
    assert len(list(destination.iterdir())) == 1


def test_download_rejects_directory_at_destination(tmp_path):
    (tmp_path / "metadata.csv").mkdir()

    with pytest.raises(IsADirectoryError):
        utils.download("https://data.example.org/metadata.csv", str(tmp_path), Mock())
