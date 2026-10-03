"""Prepare the real OrganAge example from the CC0 PRIDE deposit PAD000022.

Run from the repository root:
    uv run --with openpyxl python tutorials/data/prepare_pad000022.py \
        --source-dir /tmp/pad000022 --output-dir hf_static_data/repo
"""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

SOURCE_URL = "https://ftp.pride.ebi.ac.uk/pride/data/archive/2025/11/PAD000022/"
SOURCES = {
    "Olink3K_Data_npx.csv": "2c932134a20aed4fd705a684b28b093de541705a13b20dcf3c28e9649317c395",
    "Olink3K_sample_descriptions.xlsx": "e17f56833b858e6cf4dc6fb5ad63ceb135a98318852cc29ce329b094b2f347f1",
}
CLOCKS = [
    "organagechronologicalbrain",
    "organagechronologicalheart",
    "organagechronologicalkidney",
    "organagemortalitybrain",
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source_dir, output_dir):
    source_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, expected in SOURCES.items():
        path = source_dir / filename
        if not path.exists():
            with urlopen(SOURCE_URL + filename, timeout=120) as response:
                path.write_bytes(response.read())
        if sha256(path) != expected:
            raise ValueError(f"Source checksum mismatch: {filename}")

    assays_path = Path(__file__).with_name("PAD000022_assays.csv")
    assays = pd.read_csv(assays_path)
    assert len(assays) == 134 and assays.Assay.is_unique and assays.OlinkID.is_unique
    raw = pd.read_csv(source_dir / "Olink3K_Data_npx.csv", dtype={"SampleID": str})
    sample_map = pd.read_excel(
        source_dir / "Olink3K_sample_descriptions.xlsx",
        dtype={"Olink_SampleId": str, "SubjectID": str},
    )
    # The spreadsheet pads numeric Olink sample IDs with a zero; the NPX CSV does not.
    sample_map["SampleID"] = sample_map.Olink_SampleId.str.lstrip("0")
    sample_map = sample_map.loc[sample_map.analyzed_sample.isna() & ~sample_map.replicates].copy()
    assert sample_map.SampleID.is_unique and sample_map.SubjectID.is_unique
    selected = raw.merge(assays, on=["Assay", "OlinkID", "UniProt", "Panel"], validate="many_to_one")
    selected = selected.loc[selected.SampleID.isin(sample_map.SampleID)]
    assert not selected.duplicated(["SampleID", "Assay"]).any()
    valid = selected.QC_Warning.eq("PASS") & selected.Assay_Warning.eq("PASS") & np.isfinite(selected.NPX)
    eligible = (
        selected.assign(valid=valid).groupby("SampleID").agg(complete=("Assay", "nunique"), passed=("valid", "all"))
    )
    ids = eligible.index[(eligible.complete == len(assays)) & eligible.passed].sort_values()[:32]
    assert len(ids) == 32
    df = selected.loc[selected.SampleID.isin(ids)].pivot(index="SampleID", columns="Assay", values="NPX")
    df = df.loc[ids, assays.Assay].astype("float64")
    assert np.isfinite(df.to_numpy()).all()
    df.columns.name = None
    df.insert(0, "subject_id", sample_map.set_index("SampleID").loc[ids, "SubjectID"])
    # Object strings keep the pickle readable by supported pandas 2.x releases.
    df["subject_id"] = df.subject_id.astype(object)
    df.index = pd.Index(df.index.to_numpy(dtype=object), dtype=object, name="SampleID")
    df.columns = pd.Index(df.columns.to_numpy(dtype=object), dtype=object)
    df.attrs = {
        "accession": "PAD000022",
        "source_doi": "10.6019/PAD000022",
        "publication_doi": "10.1038/s42004-025-01665-1",
        "license": "CC0-1.0",
        "platform": "Olink Explore 3072",
        "unit": "NPX",
        "normalization": "Plate control, unchanged from the source CSV",
    }
    df.to_pickle(output_dir / "PAD000022_subset.pkl", protocol=4)
    df.to_csv(output_dir / "PAD000022_subset.csv")
    assays.to_csv(output_dir / "PAD000022_assays.csv", index=False)
    provenance = {
        **df.attrs,
        "creators": "Douglas Y. Kirsher, Shreya Chand, Aron Phong, Bich Nguyen, Balazs G. Szoke, Sara Ahadi",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "license_evidence": "https://www.ebi.ac.uk/pride/ws/archive/v2/projects/PAD000022",
        "sources": [{"url": SOURCE_URL + filename, "sha256": digest} for filename, digest in SOURCES.items()],
        "samples": len(df),
        "protein_assays": len(assays),
        "clock_names": CLOCKS,
        "selection": "First 32 sorted Olink SampleIDs among author-analyzed, non-replicate subjects "
        "with finite NPX and PASS sample/assay QC for all 134 selected assays.",
        "transformations": [
            "Strip leading zero padding from spreadsheet sample IDs to join the NPX CSV.",
            "Select assays listed in PAD000022_assays.csv, then pivot samples into rows.",
            "Retain source NPX values, including values below LOD; no imputation or scaling.",
            "Join the published de-identified SubjectID as subject_id metadata.",
        ],
        "sample_ids": df.index.tolist(),
        "limitations": [
            "Only 134 assays are included; use this subset for the listed OrganAge models.",
            "Exact ages are not provided; PAC, HPS and PAOPAC cannot be scored with this example.",
            "Plate-control NPX has not been harmonized to the clocks' UK Biobank training cohort.",
            "This QC-selected subset is for demonstrating scoring, not assessing clock accuracy or calibration.",
        ],
        "artifacts_sha256": {
            filename: sha256(output_dir / filename)
            for filename in ["PAD000022_subset.pkl", "PAD000022_subset.csv", "PAD000022_assays.csv"]
        },
    }
    (output_dir / "PAD000022_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Prepared {len(df)} samples x {len(assays)} measured proteins in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source_dir, args.output_dir)
