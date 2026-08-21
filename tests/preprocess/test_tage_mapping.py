import pandas as pd
import pytest

from pyaging.preprocess._tage import _map_to_mouse_entrez

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
