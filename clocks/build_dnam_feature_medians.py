"""Build the offline human CpG beta reference from the original GEO matrices."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd

from pyaging.preprocess import epicv2_probe_aggregation

HANNUM_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE40nnn/GSE40279/matrix/GSE40279_series_matrix.txt.gz"
BOA_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE246nnn/GSE246337/suppl/GSE246337_betas.csv.gz"


def compute_cohort_medians(beta: pd.DataFrame, *, epicv2: bool) -> pd.Series:
    """Reduce a samples-by-probes beta block to one median per CpG."""
    valid = beta.where(beta.ge(0) & beta.le(1))
    if epicv2:
        valid = epicv2_probe_aggregation(valid, verbose=False)
    return valid.median().dropna()


def combine_cohort_medians(hannum: pd.Series, boa: pd.Series) -> pd.Series:
    """Average shared medians and keep medians available in a single cohort."""
    return pd.concat([hannum, boa], axis=1).mean(axis=1).dropna().sort_index()


def read_cohort(path: Path, *, epicv2: bool) -> tuple[pd.Series, dict]:
    """Read probe rows in blocks, carrying any split EPIC v2 group forward."""
    seen = set()
    results = []
    pending = pd.DataFrame()
    raw_probes = 0

    def summarize(block, sites):
        if seen.intersection(sites):
            raise ValueError("CpG groups must be contiguous in the source matrix")
        seen.update(sites)
        results.append(compute_cohort_medians(block.T, epicv2=epicv2))

    with gzip.open(path, "rt", encoding="utf-8-sig") as stream:
        if not epicv2:
            for line in stream:
                if line.strip() == "!series_matrix_table_begin":
                    break
            else:
                raise ValueError("Hannum series matrix has no beta table")
        with pd.read_csv(
            stream,
            sep="," if epicv2 else "\t",
            index_col=0,
            comment=None if epicv2 else "!",
            chunksize=4096,
            float_precision="round_trip",
            na_values=["null"],
        ) as chunks:
            for chunk in chunks:
                sample_count = len(chunk.columns)
                raw_probes += len(chunk)
                block = pd.concat([pending, chunk]) if not pending.empty else chunk
                block = block.loc[block.index.str.fullmatch(r"cg[0-9]{8}(?:_[^_]+)?")]
                if block.empty:
                    continue
                sites = block.index.str.split("_", n=1).str[0] if epicv2 else block.index
                carry = sites == sites[-1] if epicv2 else sites.isna()
                summarize(block.loc[~carry], sites[~carry])
                pending = block.loc[carry]
            if not pending.empty:
                summarize(pending, pending.index.str.split("_", n=1).str[0])
    if not results:
        raise ValueError(f"No CpG beta rows found in {path.name}")
    medians = pd.concat(results)
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return medians, {
        "url": BOA_URL if epicv2 else HANNUM_URL,
        "file": path.name,
        "sha256": digest,
        "samples": sample_count,
        "source_rows": raw_probes,
        "cpgs_after_aggregation": len(seen),
        "cpgs_with_medians": len(medians),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hannum", type=Path)
    parser.add_argument("boa", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    hannum, hannum_source = read_cohort(args.hannum, epicv2=False)
    boa, boa_source = read_cohort(args.boa, epicv2=True)
    combined = combine_cohort_medians(hannum, boa)
    shared = len(hannum.index.intersection(boa.index))
    provenance = {
        "hannum": hannum_source,
        "boa": boa_source,
        "shared": shared,
        "hannum_only": len(hannum) - shared,
        "boa_only": len(boa) - shared,
        "method": "Within-sample EPIC v2 replicate mean, then cohort medians of finite betas in [0, 1], "
        "then the equal-weight mean of available cohort medians.",
    }
    payload = {"schema_version": 1, "provenance": provenance, "features": combined.to_dict()}
    with (
        args.output.open("wb") as output,
        gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0) as compressed,
    ):
        compressed.write(json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8"))
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
