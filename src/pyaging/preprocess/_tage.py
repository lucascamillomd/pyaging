"""Cohort preprocessing for the tAge transcriptomic clocks (Tyshkovskiy 2026).

The tAge models live in mouse Entrez gene space, so expression from any of the
four supported species has to be translated before anything else happens.
``_map_to_mouse_entrez`` mirrors ``map_genes`` in the reference package's
``R/genes.R``: an unrecognised gene is dropped, and several source genes landing
on the same mouse Entrez ID have their values **summed** into one column
(``tapply(x, mapped_ids, sum)`` in ``.apply_gene_mapping``). Summing is correct
here because mapping happens on raw counts, before normalisation.

The reference resolves IDs in two hops (source ID -> source-species Entrez ->
mouse Entrez, the second hop skipped for mouse and macaque) and, unlike the
first hop, breaks ties in the second by keeping the first row rather than
summing. The mapping asset built by ``clocks/build_tage_gene_mapping.py``
flattens both hops into a single lookup, which is exact rather than an
approximation only because every ortholog hop is injective on the reference
tables; the build script asserts that invariant.
"""

import pandas as pd

TAGE_SPECIES = ("mouse", "rat", "macaque", "human")


def _map_to_mouse_entrez(df: pd.DataFrame, species: str, mapping: pd.DataFrame) -> pd.DataFrame:
    """Translate a samples x genes frame into samples x mouse-Entrez columns.

    Genes with no mapping are dropped; genes sharing a mouse Entrez ID are summed.
    """
    if species not in TAGE_SPECIES:
        raise ValueError(f"species must be one of {TAGE_SPECIES}, got {species!r}")

    rows = mapping[mapping["species"] == species]
    lookup = dict(zip(rows["source_id"], rows["mouse_entrez"], strict=True))

    # Entrez IDs read from a CSV arrive as integers, so normalise before lookup.
    targets = df.columns.astype(str).str.upper().map(lookup)
    kept = targets.notna()

    mapped = df.loc[:, kept].copy()
    mapped.columns = targets[kept]
    return mapped.T.groupby(level=0).sum().T
