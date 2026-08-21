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

import warnings

import numpy as np
import pandas as pd

from ..utils._hf import download_hf_file

TAGE_SPECIES = ("mouse", "rat", "macaque", "human")

# ``var_names`` that carry the cohort's species rather than a gene, one 0/1
# value per sample. The clocks reuse the mammalian-array idiom, where a
# covariate such as ``female`` is simply another column of the matrix.
SPECIES_COLUMNS = TAGE_SPECIES

# ``obs`` column naming the samples to centre against.
REFERENCE_COLUMN = "tage_reference_group"

# filter_genes' defaults in the reference pipeline (R/preprocessing.R:68).
COUNT_THRESHOLD = 10
PERCENT_THRESHOLD = 20


def _warn(logger, message: str) -> None:
    """Raise a warning on both the predict display and Python's warnings channel.

    The display logger renders nothing at ``verbose=False``, so a warning that
    only went there would be invisible in exactly the scripted runs where it
    matters most -- notably the silent default to mouse, which can score a human
    cohort as a mouse one. Both channels are fed from one string so their
    wording cannot drift apart.

    ``stacklevel=2`` blames the pipeline stage that called this, not this line.
    Reaching the user's ``predict_age`` call would mean hard-coding the depth of
    ``predict_age -> apply_cohort_transform -> _prepare_tage``, which is more
    fragile than it is worth.
    """
    logger.warning(message, indent_level=2)
    warnings.warn(message, UserWarning, stacklevel=2)


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


def _resolve_species(frame: pd.DataFrame, logger) -> tuple[str, pd.DataFrame]:
    """Read the cohort's species off its indicator columns and drop them.

    A species indicator is a column of the matrix, not a gene: one 0/1 value per
    sample, the same for every sample, named after the species. Exactly one
    indicator set to 1 picks the species; none at all -- or all of them zero --
    means mouse, which is what the clocks were trained in. Names are matched
    case-insensitively, so a ``Human`` column can never be read as an unlabelled
    mouse cohort.

    The indicators are dropped whether or not they were used. The gene mapping
    would drop them anyway, but only after the gene filter had counted them, and
    a column of ones is not a gene.
    """
    indicators = [name for name in frame.columns if str(name).strip().lower() in SPECIES_COLUMNS]
    selected = []
    for name in indicators:
        values = frame[name].to_numpy(dtype=float)
        unique = np.unique(values)
        if unique.size != 1:
            raise ValueError(
                f"the {name!r} species indicator must hold the same value for every sample, "
                f"got {unique.size} distinct values"
            )
        if unique[0] not in (0.0, 1.0):
            raise ValueError(f"the {name!r} species indicator must be 0 or 1, got {unique[0]:g}")
        if unique[0] == 1.0:
            selected.append(str(name).strip().lower())

    if len(selected) > 1:
        raise ValueError(f"more than one species indicator is set to 1: {sorted(selected)}")

    remaining = frame.drop(columns=indicators)
    if not selected:
        _warn(
            logger,
            "no species indicator column found; defaulting to mouse (add a column named "
            f"{'/'.join(SPECIES_COLUMNS)}, set to 1 for every sample, to say otherwise)",
        )
        return "mouse", remaining
    return selected[0], remaining


def _resolve_reference_group(adata) -> list | None:
    """Read the samples to centre against from ``obs[REFERENCE_COLUMN]``.

    Absent, the cohort centres on every sample -- the reference pipeline's own
    default. Present, the truthy rows are the reference group.
    """
    if REFERENCE_COLUMN not in adata.obs.columns:
        return None
    column = adata.obs[REFERENCE_COLUMN]
    values = np.asarray(column)
    if values.dtype == bool:
        mask = values
    elif np.issubdtype(values.dtype, np.number):
        mask = values != 0
    else:
        raise ValueError(f"adata.obs[{REFERENCE_COLUMN!r}] must be boolean (or 0/1), got dtype {values.dtype}")
    names = list(adata.obs_names[mask])
    if len(names) == 0:
        raise ValueError(f"adata.obs[{REFERENCE_COLUMN!r}] selects no samples")
    return names


def _prepare_tage(adata, dir: str = "pyaging_data", logger=None) -> pd.DataFrame:
    """
    Run the tAge cohort preprocessing on a matrix of raw RNA-seq counts.

    Reproduces the ``scaled_diff`` branch of the reference package's
    ``tAge_preprocessing()`` (Tyshkovskiy 2026): drop the species indicator
    columns -> filter genes -> map to mouse Entrez IDs -> RLE normalise ->
    ``log10(x + 1)`` -> z-score each sample -> subtract the reference group's
    per-gene median. Matching the clocks' feature lists happens afterwards, back
    in ``predict_age``.

    ``predict_age`` calls this for any clock declaring ``cohort_transform =
    "tage"``, once per call however many such clocks are requested; it is not a
    step users perform themselves. The species and the reference group are read
    off the input rather than passed as arguments, because by the time this runs
    there is no user call to pass them to: the species comes from a 0/1
    indicator column among ``var_names``, and the reference group from
    ``obs["tage_reference_group"]``.

    The tAge clocks are cohort-relative: every stage above uses statistics of the
    whole input, so predictions depend on which samples are predicted together. A
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
        Samples x genes raw counts. ``var_names`` must be gene identifiers
        (symbols, Ensembl, or Entrez IDs) of the cohort's species, optionally
        alongside a species indicator column.
    dir : str
        Directory the gene mapping is downloaded to. Defaults to "pyaging_data".
    logger : optional
        Internal pipeline logger; warnings surface on the predict display, and
        the two that would otherwise pass unnoticed -- the default to mouse and
        a poor gene overlap -- are also raised as ``UserWarning`` so they survive
        ``verbose=False``.

    Returns
    -------
    pandas.DataFrame
        The centred matrix, samples x mouse Entrez IDs. ``adata`` itself is left
        untouched apart from ``uns["tage_preparation"]``, which records the run's
        parameters and mapping statistics.

    Raises
    ------
    ValueError
        If the species indicators are inconsistent, fewer than two samples are
        given, the reference column selects no samples, or no gene survives
        filtering and mapping.

    Notes
    -----
    One deliberate divergence from the reference: when its ``control_group_label``
    matches no sample, ``control_subtraction`` falls back to centring on the whole
    cohort with only a printed message. That silently answers a different question
    than the one asked, so an empty reference column raises here instead.
    """
    if logger is None:
        logger = _NullLogger()
    if adata.n_obs < 2:
        raise ValueError("the tAge clocks are cohort-relative and need at least two samples")

    reference_index = _resolve_reference_group(adata)

    # ``to_df`` densifies a sparse matrix and labels the axes; the ``astype``
    # copies, so nothing downstream can reach the caller's counts and the other
    # clocks in the same predict_age call still see the original data.
    frame = adata.to_df().astype(np.float64)
    species, frame = _resolve_species(frame, logger)
    if species != "mouse":
        logger.warning(
            f"tage is calibrated in months of mouse age; a {species} cohort needs rescaling by its own "
            "maximum lifespan over 48 months (tagemortality, a log hazard ratio, is never rescaled)",
            indent_level=2,
        )

    filtered = _filter_genes(frame)
    if filtered.shape[1] == 0:
        raise ValueError(
            f"no gene reached {COUNT_THRESHOLD} counts in {PERCENT_THRESHOLD}% of samples; "
            "the tAge clocks expect raw RNA-seq counts, not normalized expression"
        )

    mapped = _map_to_mouse_entrez(filtered, species, _load_mapping(dir))
    if mapped.shape[1] == 0:
        raise ValueError(f"no input gene could be mapped to a mouse Entrez ID for species {species!r}")
    # Overlap is judged against the filtered genes, not the raw input: an
    # RNA-seq matrix carries tens of thousands of unexpressed genes that the
    # filter drops, so the raw fraction says more about annotation size than
    # about how well the input matches the mapping table.
    if mapped.shape[1] < 0.5 * filtered.shape[1]:
        _warn(
            logger,
            f"only {mapped.shape[1]} of {filtered.shape[1]} expressed genes mapped to mouse "
            f"Entrez IDs; check that var_names are {species} gene identifiers",
        )

    centered = _center_against_reference(_scale_samples(_log_transform(_rle_normalize(mapped))), reference_index)

    adata.uns["tage_preparation"] = {
        "species": species,
        "n_input_genes": int(frame.shape[1]),
        "n_filtered_genes": int(filtered.shape[1]),
        "n_mapped_genes": int(mapped.shape[1]),
        "n_reference_samples": len(reference_index) if reference_index is not None else int(adata.n_obs),
        "reference_group": list(reference_index) if reference_index is not None else "all_samples",
    }
    return centered


class _NullLogger:
    """Swallow the pipeline log when the transform is called without a display."""

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None

        return _noop
