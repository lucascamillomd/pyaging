from unittest.mock import Mock

import pandas as pd

import pyaging.preprocess._preprocess_utils as preprocess_utils


def test_load_ensembl_metadata_uses_downloaded_hf_path_and_filters_chromosomes(monkeypatch, tmp_path):
    metadata_path = tmp_path / "returned-by-helper.csv"
    pd.DataFrame(
        {
            "gene_id": ["keep", "drop"],
            "chr": ["1", "MT"],
        }
    ).to_csv(metadata_path, index=False)
    logger = Mock()
    download_hf_file = Mock(return_value=str(metadata_path))
    caller_dir = str(tmp_path / "ensembl-data")
    monkeypatch.setattr(preprocess_utils, "download_hf_file", download_hf_file)

    genes = preprocess_utils.load_ensembl_metadata(caller_dir, logger)

    download_hf_file.assert_called_once_with(
        "Ensembl-105-EnsDb-for-Homo-sapiens-genes.csv",
        caller_dir,
        logger,
        indent_level=1,
    )
    assert genes.index.tolist() == ["keep"]
    assert genes.loc["keep", "chr"] == "1"
