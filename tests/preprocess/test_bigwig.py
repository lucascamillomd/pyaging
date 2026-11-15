import numpy as np
import pandas as pd
import pytest

from pyaging.preprocess import _preprocess


def test_bigwig_signals_and_missing_regions_keep_input_order(monkeypatch, tmp_path):
    pybigwig = pytest.importorskip("pyBigWig")
    path = tmp_path / "sample.bw"
    with pybigwig.open(str(path), "w") as bw:
        bw.addHeader([("chr1", 100)])
        bw.addEntries(["chr1", "chr1"], [0, 10], ends=[10, 20], values=[2.0, 4.0])
    genes = pd.DataFrame({"gene_id": ["mean", "missing"], "chr": ["1", "2"], "start": [1, 1], "end": [20, 10]})
    monkeypatch.setattr(_preprocess, "load_ensembl_metadata", lambda *args, **kwargs: genes)
    result = _preprocess.bigwig_to_df([str(path), str(path)], verbose=False)
    assert result.index.tolist() == [str(path), str(path)]
    assert result.columns.tolist() == ["mean", "missing"]
    np.testing.assert_allclose(result.to_numpy(), [[np.arcsinh(3.0), 0.0]] * 2)


def test_empty_bigwig_input_is_rejected_before_downloading_metadata(monkeypatch):
    monkeypatch.setattr(_preprocess, "PYBIGWIG_AVAILABLE", True)

    def unexpected_download(*args, **kwargs):
        pytest.fail("empty inputs should not download genomic metadata")

    monkeypatch.setattr(_preprocess, "load_ensembl_metadata", unexpected_download)
    with pytest.raises(ValueError, match="at least one"):
        _preprocess.bigwig_to_df([], verbose=False)
