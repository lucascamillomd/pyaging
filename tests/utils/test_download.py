from unittest.mock import Mock

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
