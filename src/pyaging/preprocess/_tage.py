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

import numpy as np
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


def _rle_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Scale raw counts by edgeR's RLE factors, as ``RLE_normalization`` does.

    The reference (``R/preprocessing.R:111``) divides the counts by
    ``lib.size * calcNormFactors(counts, method = "RLE")`` and multiplies by
    1e7, so the result is counts per 10 million RLE-adjusted library size.

    edgeR's RLE factor for a sample is the median ratio of its counts to the
    per-gene geometric mean over the cohort, taken over the genes whose
    geometric mean is nonzero (``.calcFactorRLE``); ``calcNormFactors`` then
    divides by the library size and rescales the factors to a geometric mean of
    one. A gene that is zero in any sample has a zero geometric mean and so
    drops out, which also covers edgeR's up-front removal of all-zero genes.
    """
    counts = df.to_numpy(dtype=float)
    if counts.shape[1] == 0:
        raise ValueError("RLE normalization needs at least one gene")

    library_sizes = counts.sum(axis=1)
    with np.errstate(divide="ignore"):
        # log(0) -> -inf -> a zero geometric mean, which is what excludes the gene.
        geometric_means = np.exp(np.log(counts).mean(axis=0))
    usable = geometric_means > 0
    if not usable.any():
        raise ValueError("RLE normalization needs a gene expressed in every sample")

    ratios = counts[:, usable] / geometric_means[usable]
    factors = np.median(ratios, axis=1) / library_sizes
    factors /= np.exp(np.log(factors).mean())

    return df.div(library_sizes * factors, axis=0) * 1e7


def _log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ``log10(x + 1)``, matching ``log_transform`` (``R/preprocessing.R:218``)."""
    return np.log10(df + 1.0)


def _scale_genes(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each sample across its genes, as ``scale_eset`` does.

    ``scale_eset`` (``R/preprocessing.R:254``) calls base R's ``scale()`` on a
    genes x samples matrix, and ``scale()`` works down the columns -- so despite
    the stage's name the standardisation is per sample, over the gene axis, with
    the sample standard deviation (``ddof=1``). A sample with no variation
    yields ``NaN`` here exactly as it does in R; nothing fills it in.
    """
    with np.errstate(invalid="ignore"):
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1, ddof=1), axis=0)


def _center_against_reference(df: pd.DataFrame, reference_index=None) -> pd.DataFrame:
    """Subtract the reference group's per-gene median from every sample.

    Mirrors ``control_subtraction`` (``R/preprocessing.R:298``): with no
    reference group the median is taken over all samples, and the median skips
    missing values (``na.rm = TRUE``), so a gene the alignment stage padded with
    ``NaN`` stays ``NaN`` for the model's imputer to fill.

    ``reference_index`` is a list of sample labels; ``None`` centres on the whole
    cohort.
    """
    reference = df if reference_index is None else df.loc[reference_index]
    if reference.empty:
        raise ValueError("reference_index selects no samples")
    return df.sub(reference.median(axis=0), axis=1)
