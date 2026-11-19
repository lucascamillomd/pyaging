import warnings

import numpy as np
import pandas as pd
import pytest

from pyaging.preprocess import _preprocess


def _write_bigwig(path, header, entries):
    pybigwig = pytest.importorskip("pyBigWig")
    with pybigwig.open(str(path), "w") as bw:
        bw.addHeader(header)
        for chromosome, starts, ends, values in entries:
            bw.addEntries([chromosome] * len(starts), starts, ends=ends, values=values)
    return str(path)


def _use_genes(monkeypatch, genes):
    monkeypatch.setattr(_preprocess, "load_ensembl_metadata", lambda *args, **kwargs: genes)


GENES = pd.DataFrame(
    {
        "gene_id": ["g1", "g1_zero", "gX", "g1_uncovered"],
        "chr": ["1", "1", "X", "1"],
        "start": [1, 11, 1, 31],
        "end": [20, 20, 10, 40],
    }
)


def test_bigwig_signals_and_missing_regions_keep_input_order(monkeypatch, tmp_path):
    path = _write_bigwig(
        tmp_path / "sample.bw",
        [("chr1", 100)],
        [("chr1", [0, 10], [10, 20], [2.0, 4.0])],
    )
    genes = pd.DataFrame({"gene_id": ["mean", "missing"], "chr": ["1", "2"], "start": [1, 1], "end": [20, 10]})
    _use_genes(monkeypatch, genes)
    with pytest.warns(UserWarning, match="sample.bw"):
        result = _preprocess.bigwig_to_df([path, path], verbose=False)
    assert result.index.tolist() == [path, path]
    assert result.columns.tolist() == ["mean", "missing"]
    # A chromosome absent from the file is unknown, not zero signal.
    np.testing.assert_allclose(result.to_numpy(), [[np.arcsinh(3.0), np.nan]] * 2)


@pytest.mark.parametrize("prefix", ["chr", ""])
def test_ucsc_and_ensembl_chromosome_names_give_the_same_signal(monkeypatch, tmp_path, prefix):
    path = _write_bigwig(
        tmp_path / f"named{prefix}.bw",
        [(prefix + "1", 100), (prefix + "X", 50)],
        [
            (prefix + "1", [0, 10], [10, 20], [2.0, 0.0]),
            (prefix + "X", [0], [10], [5.0]),
        ],
    )
    _use_genes(monkeypatch, GENES)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = _preprocess.bigwig_to_df(path, verbose=False)
    # bigWigs often omit zero-signal intervals, so an uncovered region is 0 like reported zero signal.
    np.testing.assert_allclose(result.to_numpy(), [[np.arcsinh(1.0), 0.0, np.arcsinh(5.0), 0.0]])


def test_missing_chromosome_is_nan_with_one_warning_per_file(monkeypatch, tmp_path):
    path = _write_bigwig(tmp_path / "noX.bw", [("chr1", 100)], [("chr1", [0], [20], [3.0])])
    _use_genes(monkeypatch, GENES)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _preprocess.bigwig_to_df([path], verbose=False)
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert len(messages) == 1
    assert "noX.bw" in messages[0] and "X" in messages[0]
    assert np.isnan(result.loc[path, "gX"])
    assert result.loc[path, "g1"] == pytest.approx(np.arcsinh(3.0))


def test_bigwig_without_any_gene_chromosome_is_rejected(monkeypatch, tmp_path):
    path = _write_bigwig(tmp_path / "scaffold.bw", [("scaffold_1", 100)], [("scaffold_1", [0], [20], [3.0])])
    _use_genes(monkeypatch, GENES)
    with pytest.raises(ValueError, match="scaffold.bw"):
        _preprocess.bigwig_to_df(path, verbose=False)


def test_mitochondrial_chromosome_names_are_resolved(monkeypatch, tmp_path):
    path = _write_bigwig(tmp_path / "mito.bw", [("chrM", 100)], [("chrM", [0], [20], [3.0])])
    genes = pd.DataFrame({"gene_id": ["mt"], "chr": ["MT"], "start": [1], "end": [20]})
    _use_genes(monkeypatch, genes)
    result = _preprocess.bigwig_to_df(path, verbose=False)
    assert result.loc[path, "mt"] == pytest.approx(np.arcsinh(3.0))


def test_regions_past_the_chromosome_end_are_clipped(monkeypatch, tmp_path):
    path = _write_bigwig(tmp_path / "short.bw", [("chr1", 30)], [("chr1", [0, 20], [20, 30], [3.0, 6.0])])
    genes = pd.DataFrame(
        {"gene_id": ["overhang", "beyond"], "chr": ["1", "1"], "start": [21, 31], "end": [500, 60]},
    )
    _use_genes(monkeypatch, genes)
    result = _preprocess.bigwig_to_df(path, verbose=False)
    assert result.loc[path, "overhang"] == pytest.approx(np.arcsinh(6.0))
    assert np.isnan(result.loc[path, "beyond"])


def test_empty_bigwig_input_is_rejected_before_downloading_metadata(monkeypatch):
    monkeypatch.setattr(_preprocess, "PYBIGWIG_AVAILABLE", True)

    def unexpected_download(*args, **kwargs):
        pytest.fail("empty inputs should not download genomic metadata")

    monkeypatch.setattr(_preprocess, "load_ensembl_metadata", unexpected_download)
    with pytest.raises(ValueError, match="at least one"):
        _preprocess.bigwig_to_df([], verbose=False)
