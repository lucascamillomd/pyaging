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

import anndata
import numpy as np
import pandas as pd

from ..logger._live import live_step
from ..utils._hf import download_hf_file

TAGE_SPECIES = ("mouse", "rat", "macaque", "human")

# filter_genes' defaults in the reference pipeline (R/preprocessing.R:68).
COUNT_THRESHOLD = 10
PERCENT_THRESHOLD = 20


def _filter_genes(
    df: pd.DataFrame, count_threshold: int = COUNT_THRESHOLD, percent_threshold: float = PERCENT_THRESHOLD
) -> pd.DataFrame:
    """Drop genes not expressed in enough samples, as ``filter_genes`` does.

    The reference (``R/preprocessing.R:72``) keeps a gene when
    ``sum(x >= count_threshold, na.rm = TRUE) >= ncol * (percent_threshold/100)``.
    Both comparisons are inclusive, and ``na.rm`` means a missing count never
    counts toward the quota -- which ``NaN >= threshold`` reproduces, since that
    comparison is ``False``.

    The filter is not cosmetic: on the authors' own example data it takes 57 010
    genes down to 19 550, and running the mapper without it yields a different
    gene set (28 266 columns) and different RLE factors downstream.
    """
    passing = (df >= count_threshold).sum(axis=0) >= df.shape[0] * (percent_threshold / 100)
    return df.loc[:, passing]


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
    if np.isnan(counts).any():
        # calcNormFactors stops on NA counts rather than propagating them.
        raise ValueError("RLE normalization needs counts without missing values")

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


def _scale_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each sample across its genes, as ``scale_eset`` does.

    ``scale_eset`` (``R/preprocessing.R:254``) calls base R's ``scale()`` on a
    genes x samples matrix, and ``scale()`` works down the columns -- so despite
    the reference's name for the stage the standardisation is per sample, over
    the gene axis, with the sample standard deviation (``ddof=1``). A sample with
    no variation yields ``NaN`` here exactly as it does in R; nothing fills it in.
    """
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


def _load_mapping(dir: str) -> pd.DataFrame:
    """Fetch the flattened source-ID -> mouse-Entrez lookup from the Hub."""
    path = download_hf_file("tage_gene_mapping.csv.gz", dir, repo_id="pyaging/tage")
    return pd.read_csv(path, dtype=str)


def _resolve_reference_group(adata, reference_group) -> list | None:
    """Turn obs names or a boolean mask into a list of distinct sample labels.

    A repeated label is dropped rather than kept: naming a sample twice would
    otherwise double its weight in the per-gene median that centres the cohort.
    """
    if reference_group is None:
        return None
    reference = np.asarray(reference_group)
    if reference.dtype == bool:
        if reference.shape[0] != adata.n_obs:
            raise ValueError("boolean reference_group must have one entry per sample")
        names = list(adata.obs_names[reference])
    else:
        names = list(dict.fromkeys(str(name) for name in reference))
        missing = sorted(set(names) - set(adata.obs_names))
        if missing:
            raise ValueError(f"reference_group names not in adata.obs_names: {missing[:5]}")
    if len(names) == 0:
        raise ValueError("reference_group selects no samples")
    return names


def prepare_tage(
    adata: anndata.AnnData,
    species: str,
    reference_group=None,
    dir: str = "pyaging_data",
    verbose: bool = True,
) -> anndata.AnnData:
    """
    Run the tAge cohort preprocessing on a matrix of raw RNA-seq counts.

    Reproduces the ``scaled_diff`` branch of the reference package's
    ``tAge_preprocessing()`` (Tyshkovskiy 2026): filter genes -> map to mouse
    Entrez IDs -> RLE normalise -> ``log10(x + 1)`` -> z-score each sample ->
    subtract the reference group's per-gene median. Matching the clocks' feature
    list happens later, inside ``predict_age``.

    The tAge clocks are cohort-relative: every stage above uses statistics of the
    whole input, so predictions depend on which samples are prepared together. A
    single sample cannot be prepared, and a prediction is an age difference
    against the reference group rather than an absolute age.

    Two samples clear the hard minimum but are not enough to normalise against:
    the RLE factors, the reference median, and the gene filter are all cohort
    statistics, so a very small cohort (roughly fewer than ten samples, or a
    reference group that small) makes them noisy and the resulting predictions
    correspondingly unstable. Nothing here rejects such an input; interpret it
    with that in mind.

    Parameters
    ----------
    adata : anndata.AnnData
        Samples x genes raw counts. ``var_names`` must be gene identifiers of
        ``species`` (symbols, Ensembl, or Entrez IDs).
    species : str
        One of ``pyaging.preprocess.TAGE_SPECIES``.
    reference_group : list of str or numpy.ndarray or None, optional
        The samples to centre against, either as ``obs_names`` or as a boolean
        mask aligned to ``adata.obs_names``. Defaults to None, which centres on
        every sample.
    dir : str
        Directory the gene mapping is downloaded to. Defaults to "pyaging_data".
    verbose : bool
        Whether to log the output to console with the logger. Defaults to True.

    Returns
    -------
    anndata.AnnData
        A new object holding the centred matrix, with mouse Entrez IDs as
        ``var_names``, ``obs`` copied from the input, ``uns["tage_prepared"]``
        set, and the run's parameters in ``uns["tage_preparation"]``.

    Raises
    ------
    ValueError
        If ``species`` is unknown, fewer than two samples are given,
        ``reference_group`` selects no samples or names samples not in the
        input, or no gene survives filtering and mapping.

    Notes
    -----
    One deliberate divergence from the reference: when its ``control_group_label``
    matches no sample, ``control_subtraction`` falls back to centring on the whole
    cohort with only a printed message. That silently answers a different question
    than the one asked, so an empty or unrecognised ``reference_group`` raises here
    instead.

    Examples
    --------
    >>> adata = pya.pp.prepare_tage(counts, species="mouse")  # doctest: +SKIP
    >>> pya.pred.predict_age(adata, "tage")  # doctest: +SKIP

    """
    if species not in TAGE_SPECIES:
        raise ValueError(f"species must be one of {TAGE_SPECIES}, got {species!r}")
    if adata.n_obs < 2:
        raise ValueError("prepare_tage needs at least two samples: the tAge clocks are cohort-relative")
    reference_index = _resolve_reference_group(adata, reference_group)

    with live_step("preparing tAge input", verbose) as (step, pipeline_logger):
        if species != "mouse":
            pipeline_logger.warning(
                f"tage is calibrated in months of mouse age; a {species} cohort needs rescaling by its own "
                "maximum lifespan over 48 months (tagemortality, a log hazard ratio, is never rescaled)",
                indent_level=2,
            )

        frame = pd.DataFrame(np.asarray(adata.X, dtype=np.float64), index=adata.obs_names, columns=adata.var_names)
        filtered = _filter_genes(frame)
        if filtered.shape[1] == 0:
            raise ValueError(
                f"no gene reached {COUNT_THRESHOLD} counts in {PERCENT_THRESHOLD}% of samples; "
                "prepare_tage expects raw RNA-seq counts, not normalized expression"
            )

        step.update("mapping genes to mouse Entrez IDs")
        mapped = _map_to_mouse_entrez(filtered, species, _load_mapping(dir))
        if mapped.shape[1] == 0:
            raise ValueError(f"no input gene could be mapped to a mouse Entrez ID for species {species!r}")
        # Overlap is judged against the filtered genes, not the raw input: an
        # RNA-seq matrix carries tens of thousands of unexpressed genes that the
        # filter drops, so the raw fraction says more about annotation size than
        # about how well the input matches the mapping table.
        if mapped.shape[1] < 0.5 * filtered.shape[1]:
            pipeline_logger.warning(
                f"only {mapped.shape[1]} of {filtered.shape[1]} expressed genes mapped to mouse "
                f"Entrez IDs; check that var_names are {species} gene identifiers",
                indent_level=2,
            )

        step.update("normalizing and centering")
        centered = _center_against_reference(_scale_samples(_log_transform(_rle_normalize(mapped))), reference_index)

        out = anndata.AnnData(X=centered.to_numpy(dtype=np.float64), obs=adata.obs.copy())
        out.obs_names = adata.obs_names
        out.var_names = centered.columns
        out.uns["tage_prepared"] = True
        out.uns["tage_preparation"] = {
            "species": species,
            "n_input_genes": int(frame.shape[1]),
            "n_filtered_genes": int(filtered.shape[1]),
            "n_mapped_genes": int(mapped.shape[1]),
            "n_reference_samples": len(reference_index) if reference_index is not None else int(adata.n_obs),
            "reference_group": list(reference_index) if reference_index is not None else "all_samples",
        }
        step.done(f"tAge input: {out.n_obs} samples × {out.n_vars} genes")

    return out
