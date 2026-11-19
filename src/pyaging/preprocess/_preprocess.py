import os
import re
import warnings

import anndata
import numpy as np
import pandas as pd

try:
    from pyBigWig import open as open_bw

    PYBIGWIG_AVAILABLE = True
except ImportError:
    PYBIGWIG_AVAILABLE = False

from ..logger._live import live_step
from ._preprocess_utils import (
    add_metadata_to_anndata,
    add_unstructured_data,
    create_anndata_object,
    impute_missing_values,
    load_ensembl_metadata,
    log_data_statistics,
)


def _bigwig_chromosome(chromosome: str, lengths: dict[str, int]) -> str | None:
    """Return the name the bigWig uses for an Ensembl chromosome, or None if it is absent."""
    bare = chromosome[3:] if chromosome.startswith("chr") else chromosome
    candidates = ["chr" + bare, bare]
    if bare in ("MT", "M"):
        candidates += ["chrM", "MT", "chrMT", "M"]
    return next((name for name in candidates if name in lengths), None)


def bigwig_to_df(bw_files: str | list[str], dir: str = "pyaging_data", verbose: bool = True) -> pd.DataFrame:
    """
    Convert bigWig files to a DataFrame, extracting signal data for genomic regions.

    This function processes a list of bigWig files, extracting signal data (such as chromatin accessibility
    or histone modification levels) for each gene based on genomic annotations from Ensembl. It computes the
    mean signal over the genomic region of each gene, applies an arcsinh transformation for normalization,
    and organizes the data into a DataFrame format.

    Parameters
    ----------
    bw_files: Union[str, List[str]]
        A list of bigWig file paths. If a single string is provided, it is converted to a list.

    dir : str
        Retained for backward compatibility. Hugging Face files use its standard cache.

    verbose: bool
        Whether to log the output to console with the logger. Defaults to True.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row represents a bigWig file and each column corresponds to a gene.
        The values in the DataFrame are the transformed signal data for each gene in each bigWig file.

    Raises
    ------
    ImportError
        If pyBigWig is not installed and the function is called.

    ValueError
        If a bigWig file contains none of the gene chromosomes.

    Notes
    -----
    The function utilizes Ensembl gene annotations and assumes the presence of genes on standard chromosomes
    (1-22, X). Non-standard chromosomes or regions outside annotated genes are not processed. The signal
    transformation uses the arcsinh function for normalization. Chromosomes may be named "chr1" (UCSC) or
    "1" (Ensembl) in the bigWig. Genes with no covered bases are 0. Genes on a chromosome the file lacks are
    NaN, and a warning lists those chromosomes; regions running past the chromosome end are clipped to it.
    This function requires pyBigWig to be installed.
    If pyBigWig is not available, an ImportError will be raised. To use this function, ensure you have installed
    pyaging with the 'histone' extra: pip install pyaging[histone]

    Examples
    --------
    >>> bigwig_files = ["sample1.bw", "sample2.bw"]
    >>> signals_df = bigwig_to_df(bigwig_files)
    # This returns a DataFrame where rows are bigWig files and columns are genes, with signal values.

    """
    if not PYBIGWIG_AVAILABLE:
        raise ImportError("pyBigWig is not installed. To use this function, please install it.")

    if isinstance(bw_files, str):
        bw_files = [bw_files]
    if not bw_files:
        raise ValueError("bw_files must contain at least one bigWig file.")

    with live_step("processing bigWig files", verbose) as (step, pipeline_logger):
        genes = load_ensembl_metadata(dir, pipeline_logger, indent_level=1)

        regions = list(zip(genes["chr"].astype(str), genes["start"], genes["end"], strict=True))
        gene_chromosomes = list(dict.fromkeys(chromosome for chromosome, _, _ in regions))
        # NaN marks a gene the file cannot describe (absent chromosome, or coordinates past
        # its end). A gene with no covered bases is 0: many bigWigs omit zero-signal intervals.
        signals = np.full((len(bw_files), len(regions)), np.nan)
        for index, bw_file in enumerate(bw_files):
            step.update(f"processing {os.path.basename(bw_file)} ({index + 1}/{len(bw_files)})")

            with open_bw(bw_file) as bw:
                lengths = bw.chroms()
                resolved = {chromosome: _bigwig_chromosome(chromosome, lengths) for chromosome in gene_chromosomes}
                absent = [chromosome for chromosome, name in resolved.items() if name is None]
                if absent and len(absent) == len(resolved):
                    raise ValueError(
                        f"bigWig file {bw_file!r} contains none of the gene chromosomes "
                        f"(it has {list(lengths)[:5]}); expected names such as 'chr1' or '1'."
                    )
                if absent:
                    warnings.warn(
                        f"bigWig file {bw_file!r} lacks chromosomes {absent}; their genes are set to NaN.",
                        UserWarning,
                        stacklevel=2,
                    )

                for gene_index, (chromosome, start, end) in enumerate(regions):
                    name = resolved[chromosome]
                    if name is None:
                        continue
                    # pyBigWig rejects intervals that run past the chromosome end.
                    end = min(end, lengths[name])
                    if start - 1 >= end:
                        continue
                    try:
                        signal = bw.stats(name, start - 1, end, type="mean", exact=True)[0]
                    except RuntimeError:
                        continue

                    signals[index, gene_index] = 0.0 if signal is None else np.arcsinh(signal)

        df_concat = pd.DataFrame(signals, index=bw_files, columns=genes.gene_id.tolist())

        plural = "s" if len(bw_files) != 1 else ""
        step.done(f"{len(bw_files)} bigWig file{plural} × {genes.shape[0]} genes")

    return df_concat


def df_to_adata(
    df: pd.DataFrame,
    metadata_cols: str | list[str] | None = None,
    imputer_strategy: str = "knn",
    verbose: bool = True,
) -> anndata.AnnData:
    """
    Converts a pandas DataFrame to an AnnData object.

    This function transforms a DataFrame containing biological data (such as gene expression
    levels, methylation data, etc.) into an AnnData object. It includes steps for handling
    missing values, and logging data statistics. The function is particularly useful
    in preparing datasets for downstream analyses in bioinformatics and computational biology.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing biological data. Rows represent samples, and columns represent features.

    metadata_cols : str or list[str], optional
        The name, or a list-like of names, of the columns in 'df' which are part of the metadata. They
        will be added to adata.obs rather than adata.X.

    imputer_strategy : str, optional
        The strategy for imputing missing values in 'df'. Supported strategies include 'mean',
        'median', 'constant' (0 values), and 'knn'. Defaults to 'knn'.

    verbose: bool
        Whether to show the progress display and warnings. Animated in
        notebooks and terminals, a plain summary when output is captured,
        and fully silent when False. Defaults to True.

    Returns
    -------
    anndata.AnnData
        The AnnData object containing the processed data, metadata, and additional annotations.

    Raises
    ------
    TypeError
        If the input 'df' is not a pandas DataFrame.

    KeyError
        If any of 'metadata_cols' is not a column of 'df'.

    Notes
    -----
    The AnnData object produced by this function is ready for various computational biology analyses,
    such as differential expression analysis, clustering, or trajectory inference. The embedded annotations
    enhance data understanding and facilitate more robust analyses.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(np.random.rand(5, 3), columns=["gene1", "gene2", "gene3"])
    >>> adata = df_to_adata(df)
    # This returns an AnnData object with the imputed data from 'df'.

    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input df must be a pandas DataFrame.")

    with live_step("building AnnData object", verbose) as (step, pipeline_logger):
        if metadata_cols is None:
            metadata_cols = []
        elif isinstance(metadata_cols, str) or not pd.api.types.is_list_like(metadata_cols):
            metadata_cols = [metadata_cols]
        else:
            metadata_cols = list(metadata_cols)
        missing_cols = [column for column in metadata_cols if column not in df.columns]
        if missing_cols:
            raise KeyError(f"Metadata columns not found in df: {missing_cols}")
        if len(metadata_cols) > 0:
            metadata = df.loc[:, metadata_cols]
            df = df.drop(metadata_cols, axis=1)
        else:
            metadata = None

        adata = create_anndata_object(df, pipeline_logger)

        add_metadata_to_anndata(adata, metadata, pipeline_logger)

        missing_pct = log_data_statistics(adata.X, pipeline_logger)

        step.update(f"imputing missing values ({imputer_strategy})")
        impute_missing_values(adata, imputer_strategy, pipeline_logger)

        if "X_imputed" in adata.layers:
            add_unstructured_data(adata, imputer_strategy, pipeline_logger)

        # Keep full cohorts in host memory; prediction transfers batches.
        adata.X = np.asfortranarray(adata.X)

        if missing_pct == 0:
            missing = " · no missing values"
        else:
            shown = f"{missing_pct:.2f}%" if missing_pct >= 0.01 else "<0.01%"
            missing = f" · {shown} missing values imputed"
        step.done(f"AnnData: {adata.n_obs} samples × {adata.n_vars} features{missing}")

    return adata


_EPICV2_SUFFIX = re.compile(r"_[TB][CO]\d+$")


def epicv2_probe_aggregation(df: pd.DataFrame, verbose: bool = True):
    """
    Aggregates probes targeting the same CpG site in a DataFrame from the Illumina Methylation EPIC array v2.

    Probes targeting the same CpG site are identified by their shared prefix (e.g., "cgXXXXXXX"), and their
    values are averaged to create a single feature for each unique CpG site. This reduces the dimensionality
    of the data by consolidating multiple probes for the same CpG site into a single value.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing probe data. Each column represents a probe, and the column names are
        expected to follow the format "cgXXXXXXX_YYYY".

    verbose: bool
        Whether to log the output to console with the logger. Defaults to True.

    Returns
    -------
    pandas.DataFrame:
        A new DataFrame with averaged values for each unique CpG site. The columns of this DataFrame correspond
        to unique CpG sites, and the column names are the CpG site identifiers (e.g., "cgXXXXXXX").
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input df must be a pandas DataFrame.")

    with live_step("scanning for duplicated probes", verbose) as (step, pipeline_logger):
        # Only strip EPIC v2 replicate suffixes (e.g. "_TC21", "_BO11"); leave metadata such as
        # "tissue_type" untouched.
        cpg_sites = pd.Index(
            [_EPICV2_SUFFIX.sub("", column) if isinstance(column, str) else column for column in df.columns],
            name=df.columns.name,
        )
        n_duplicated_probes = int(cpg_sites.duplicated().sum())

        # Even singleton probes need canonical CpG names for clock matching.
        if n_duplicated_probes == 0:
            step.done(f"no duplicated probes across {df.shape[1]} columns")
            return df if df.columns.equals(cpg_sites) else df.set_axis(cpg_sites, axis=1)

        step.update(f"averaging {n_duplicated_probes} duplicated probes")
        # Group by position so repeated column labels are counted only once.
        # sort=False preserves the input's first-occurrence ordering.
        is_probe = pd.Index(df.columns).map(lambda c: isinstance(c, str) and bool(_EPICV2_SUFFIX.search(c)))
        # Average every column that maps to a probe CpG (including an unsuffixed copy of it);
        # pass all other columns through unchanged.
        is_probe = np.asarray(cpg_sites.isin(cpg_sites[np.asarray(is_probe, dtype=bool)]), dtype=bool)
        # For float64 probes, averaging a lone probe only renames it, so only duplicated CpGs (a
        # small fraction of an array) are grouped. For other dtypes the groupby's output dtype
        # depends on every probe column (and, for float32, on whether values are missing), so
        # all probes are grouped together.
        float64_probes = (df.dtypes[is_probe] == np.float64).all()
        grouped = is_probe & cpg_sites.duplicated(keep=False) if float64_probes else is_probe
        single = is_probe & ~grouped
        aggregated_probes = df.loc[:, grouped].T.groupby(cpg_sites[grouped], sort=False).mean().T
        single_probes = df.loc[:, single].set_axis(cpg_sites[single], axis=1)
        others = df.loc[:, ~is_probe].set_axis(cpg_sites[~is_probe], axis=1)
        aggregated_df = pd.concat([others, single_probes, aggregated_probes], axis=1)[cpg_sites.unique()]
        step.done(f"{df.shape[1]} probes aggregated into {aggregated_df.shape[1]} unique CpG sites")

    return aggregated_df
