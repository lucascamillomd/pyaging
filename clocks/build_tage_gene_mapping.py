#!/usr/bin/env python3
"""Build the tAge gene/ortholog mapping asset from the reference package's tables.

Run once; the output ships to Hugging Face alongside the tAge weights:

    uv run python clocks/build_tage_gene_mapping.py "$SCRATCH/tAge" clocks/weights/tage_gene_mapping.csv.gz

Output contract: a four-column frame with ``species`` (mouse|rat|macaque|human),
``source_id`` (upper-cased symbol / Ensembl / Entrez string), ``source_type``
(symbol|ensembl|entrez) and ``mouse_entrez`` (string). ``source_id`` is unique
within a species, so a consumer can build one lookup dict per species.

The mapping mirrors ``R/genes.R`` of the reference package (``map_genes``),
which resolves IDs in up to two stages:

1. ``.load_gene_mapping`` turns the source ID into the *source species'* Entrez
   ID via ``Gene_table_<species>.csv``, dropping rows with no Entrez and keeping
   the first occurrence of each duplicated source ID. Macaque is the exception
   (``.create_monkey_gene_mapping``): it chains source ID -> macaque Ensembl ->
   mouse Entrez through ``Orthologs_monkey_to_mouse_5.0.csv`` and so is done
   after this stage.
2. ``.apply_ortholog_mapping`` turns the source species' Entrez into mouse
   Entrez via ``Table_of_orthologs.csv`` (human and rat only), again keeping the
   first occurrence of each duplicated source Entrez.

Composing the two stages into the single hop stored here is exact rather than an
approximation, because every ortholog stage is injective on the reference
tables: after the reference's own deduplication no two source Entrez (human,
rat) and no two macaque Ensembl IDs share a mouse target. This script asserts
that, so a future table revision that broke it would fail the build instead of
silently changing predictions. It matters because the reference collapses
many-to-one *gene table* hits by summing counts but resolves many-to-one
*ortholog* hits by keeping the first row, and only injectivity makes those two
rules agree on a flattened map.

The reference itself accepts only Ensembl and Gene.Symbol input. ``entrez`` rows
are pyaging's addition: mouse Entrez maps to itself, and the other species enter
at stage 2 (or, for macaque, via the gene table's own Entrez column).
"""

import sys
from pathlib import Path

import pandas as pd

GENE_TABLES = {
    "mouse": "Gene_table_mouse.csv",
    "rat": "Gene_table_rat.csv",
    "macaque": "Gene_table_monkey.csv",
    "human": "Gene_table_human.csv",
}
# Column in Table_of_orthologs.csv holding each species' own Entrez ID.
ORTHOLOG_COLUMNS = {"rat": "Entrez.Rat", "human": "Entrez.Human"}
SOURCE_COLUMNS = {"ensembl": "Ensembl", "symbol": "Gene.Symbol", "entrez": "Entrez"}


def _first_wins(frame: pd.DataFrame, key: str, value: str) -> pd.Series:
    """Map ``key`` -> ``value``, dropping empty pairs and keeping the first duplicate.

    Mirrors the reference's ``gene_table[!duplicated(gene_table[[type]]), ]``.
    """
    columns = [key] if key == value else [key, value]
    pairs = frame[columns].dropna().drop_duplicates(subset=[key], keep="first")
    return pd.Series(pairs[value].values, index=pairs[key].values)


def _ortholog_map(meta: Path, species: str) -> pd.Series:
    """Source-species Entrez -> mouse Entrez, as ``.apply_ortholog_mapping`` builds it."""
    orthologs = pd.read_csv(meta / "Table_of_orthologs.csv", dtype=str)
    return _first_wins(orthologs, ORTHOLOG_COLUMNS[species], "Entrez.Mouse")


def _macaque_map(meta: Path) -> pd.Series:
    """Macaque Ensembl -> mouse Entrez, as ``.create_monkey_gene_mapping`` builds it."""
    orthologs = pd.read_csv(meta / "Orthologs_monkey_to_mouse_5.0.csv", dtype=str)
    return _first_wins(orthologs, "Ensembl.macaca", "Entrez.mouse")


def build(clone: Path, gene_list: list[str]) -> pd.DataFrame:
    meta = clone / "inst/extdata/metadata"
    frames = []
    for species, fname in GENE_TABLES.items():
        table = pd.read_csv(meta / fname, dtype=str)
        if species == "macaque":
            # Stage 1 lands on macaque Ensembl, stage 2 crosses to mouse.
            to_mouse = _macaque_map(meta)
            assert not to_mouse.duplicated().any(), "macaque ortholog map is not injective"
            hop = "Ensembl"
        elif species in ORTHOLOG_COLUMNS:
            to_mouse = _ortholog_map(meta, species)
            assert not to_mouse.duplicated().any(), f"{species} ortholog map is not injective"
            hop = "Entrez"
        else:
            to_mouse = None  # mouse gene tables already carry mouse Entrez
            hop = "Entrez"

        for source_type, column in SOURCE_COLUMNS.items():
            if source_type == "entrez" and to_mouse is not None and hop == "Entrez":
                # Human and rat Entrez enter at stage 2, so every ID the ortholog
                # table knows is usable, not just those in the gene table.
                resolved = to_mouse
            else:
                resolved = _first_wins(table, column, hop)
                if to_mouse is not None:
                    resolved = resolved.map(to_mouse).dropna()
            frames.append(
                pd.DataFrame(
                    {
                        "species": species,
                        "source_id": resolved.index,
                        "source_type": source_type,
                        "mouse_entrez": resolved.values,
                    }
                )
            )

    # A mouse Entrez ID is its own mouse Entrez ID even when the gene table has
    # no row for it, which is true of 189 of the clock's features.
    frames.append(
        pd.DataFrame(
            {
                "species": "mouse",
                "source_id": gene_list,
                "source_type": "entrez",
                "mouse_entrez": gene_list,
            }
        )
    )

    mapping = pd.concat(frames, ignore_index=True).dropna(subset=["mouse_entrez"])
    mapping["source_id"] = mapping["source_id"].str.upper()
    mapping = mapping.drop_duplicates(subset=["species", "source_id", "source_type"], keep="first")
    return mapping.reset_index(drop=True)


def main() -> None:
    clone, out = Path(sys.argv[1]), Path(sys.argv[2])
    gene_list = [line.strip().strip('"') for line in (clone / "inst/extdata/Gene_list_all_4.6.txt").read_text().split()]
    gene_list = [gene for gene in gene_list if gene and gene != "x"]

    mapping = build(clone, gene_list)

    # One lookup dict per species requires source_id to be unambiguous across types.
    clashes = mapping.duplicated(subset=["species", "source_id"]).sum()
    assert clashes == 0, f"{clashes} source_id values are ambiguous within a species"

    # Every clock feature must be reachable when a user supplies mouse Entrez IDs.
    identity = mapping.query("species == 'mouse' and source_type == 'entrez'")
    identity = dict(zip(identity["source_id"], identity["mouse_entrez"], strict=True))
    unreachable = [gene for gene in gene_list if identity.get(gene) != gene]
    assert not unreachable, f"{len(unreachable)} clock features do not self-map: {unreachable[:5]}"

    out.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(out, index=False, compression="gzip")
    print(mapping.groupby(["species", "source_type"]).size().to_string())
    print(len(mapping), "rows ->", out)


if __name__ == "__main__":
    main()
