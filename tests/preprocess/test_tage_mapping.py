from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyaging.preprocess._tage import _map_to_mouse_entrez

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests/data/tage"
MAPPING_ASSET = REPO_ROOT / "clocks/weights/tage_gene_mapping.csv.gz"

MAPPING = pd.DataFrame(
    {
        "species": ["mouse", "mouse", "human", "human", "human"],
        "source_id": ["CDKN1A", "12575", "CDKN1A", "ENSG00000124762", "LGALS3"],
        "source_type": ["symbol", "entrez", "symbol", "ensembl", "symbol"],
        "mouse_entrez": ["12575", "12575", "12575", "12575", "16854"],
    }
)


def _frame(columns, rows=2):
    return pd.DataFrame([[float(i + j) for j in range(len(columns))] for i in range(rows)], columns=columns)


def test_mouse_symbols_map_to_entrez():
    out = _map_to_mouse_entrez(_frame(["Cdkn1a"]), "mouse", MAPPING)
    assert list(out.columns) == ["12575"]


def test_human_ensembl_maps_via_orthologs():
    out = _map_to_mouse_entrez(_frame(["ENSG00000124762"]), "human", MAPPING)
    assert list(out.columns) == ["12575"]


def test_unmapped_genes_dropped():
    out = _map_to_mouse_entrez(_frame(["LGALS3", "NOT_A_GENE"]), "human", MAPPING)
    assert list(out.columns) == ["16854"]


def test_many_to_one_collapses_by_summing():
    # The reference sums duplicate hits (R/genes.R .apply_gene_mapping:
    # `tapply(x, mapped_ids, sum)`), so the two source columns add together.
    frame = _frame(["CDKN1A", "ENSG00000124762"])
    out = _map_to_mouse_entrez(frame, "human", MAPPING)
    assert list(out.columns) == ["12575"]
    assert out.shape == (2, 1)
    assert out["12575"].tolist() == [1.0, 3.0]


def test_integer_entrez_columns_are_accepted():
    # Entrez IDs read from a CSV arrive as integers, not strings.
    out = _map_to_mouse_entrez(_frame([12575]), "mouse", MAPPING)
    assert list(out.columns) == ["12575"]


def test_other_species_mappings_are_ignored():
    # CDKN1A is a mouse row too, but a human frame must use the human rows only.
    out = _map_to_mouse_entrez(_frame(["ENSG00000124762"]), "mouse", MAPPING)
    assert out.shape[1] == 0


def test_unknown_species_raises():
    with pytest.raises(ValueError, match="species"):
        _map_to_mouse_entrez(_frame(["Cdkn1a"]), "dog", MAPPING)


@pytest.mark.skipif(not MAPPING_ASSET.exists(), reason="mapping asset not built locally")
def test_matches_reference_after_mapping_fixture():
    """Replay the reference pipeline's mapping stage against its own R-produced output.

    This is the only test that exercises the real mapping asset, and the only one
    that can catch a wrong collapse rule: R sums duplicate hits, and 5 of these
    genes really do receive two Ensembl IDs each, so a mean or first-wins collapse
    would not reproduce these values.
    """
    mapping = pd.read_csv(MAPPING_ASSET, dtype=str)
    # Fixtures are genes x samples (see tests/data/tage/README.md); the function
    # takes samples x genes, so both ends need a transpose.
    counts = pd.read_csv(FIXTURES / "input_expression.csv.gz", index_col=0)
    expected = pd.read_csv(FIXTURES / "after_mapping.csv.gz", index_col=0)
    expected.index = expected.index.astype(str)

    # filter_genes(count_threshold=10, percent_threshold=20) runs first upstream.
    kept = (counts >= 10).sum(axis=1) >= 0.2 * counts.shape[1]
    out = _map_to_mouse_entrez(counts[kept].T, "mouse", mapping)

    assert out.shape == (24, 15991)
    assert set(out.columns) == set(expected.index)
    # Counts are integers summed exactly, so the R output is reproduced bit for bit.
    assert np.array_equal(out.loc[expected.columns, expected.index].to_numpy(), expected.to_numpy().T)
