#!/usr/bin/env python3
"""Build replication clocks from checksum-pinned original-author artifacts.

CellDRIFT: clone https://github.com/MorganLevineLab/CellDRIFT at the pinned
commit, then run ``uv run --with rdata python clocks/build_replication.py
--celldrift-source /tmp/CellDRIFT``. rdata is a build-only dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csc_matrix

from pyaging.models._base_models import LinearModel
from pyaging.models._replication import CellDRIFT, MiAge
from pyaging.utils import resolve_feature_ranges

ROOT = Path(__file__).resolve().parents[1]
CELLDRIFT_COMMIT = "066b3e816795f3c0b372ff8d33f59185a490bd8b"
CELLDRIFT_HASHES = {
    "R/CellDRIFT.R": "6ee6af7dae221428823419cf7b78219cbf301a92f941adeda08bb6895c74bbaa",
    "data/CV.CellDRIFT.rda": "4965aef078d809f5ea9ac6b43078e95422ed246bb867f2cf3e5101c24c3009e0",
    "data/Fit.CellDRIFT.rda": "cabb4ffabbbb21ef61365a2c80fef0241c885ccb7dd9d91c5b0f94c0c9ea99a3",
    "data/PCA_tan.rda": "70ec2fa10a536862372fc257d901ca8ea298d0bfd93ec9ed8e14a859348f676b",
    "data/PCA_yellow.rda": "dc48b27642595d43a3d652ea452b02e1ce079ce724160370439da152d85d4448",
    "data/exampleDNAm.rda": "a3651b09ea232a73aac771a65d6350f7d430eb1b3f9f891135c66682cfa404aa",
    "data/tan_module_means.rda": "a85cacd677a377798412682a020a73f55f12133e7c33fbd8b730469682ebb173",
    "data/yellow_module_means.rda": "5796b864adbea94dc5ffe85be388d3761d13243c2995b99a9d7052c6ecc748e2",
}


def verify_sources(directory: Path, checksums: dict[str, str]) -> None:
    for relative, expected in checksums.items():
        actual = hashlib.sha256((directory / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Original source checksum mismatch: {relative}")


def build_celldrift(source: Path) -> CellDRIFT:
    """Fold the released 62-PC glmnet predictor into its equivalent CpG score."""
    import rdata

    verify_sources(source, CELLDRIFT_HASHES)

    def read(name):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Missing constructor for R class")
            return rdata.read_rda(source / "data" / f"{name}.rda")[name]

    fit, cv = read("Fit.CellDRIFT"), read("CV.CellDRIFT")
    indices = np.flatnonzero(fit["lambda"] == cv["lambda.min"].item())
    if len(indices) != 1:
        raise ValueError("Expected an exact original glmnet lambda.min match")
    index = int(indices[0])
    sparse_beta = fit["beta"]
    pc_weights = csc_matrix((sparse_beta.x, sparse_beta.i, sparse_beta.p), shape=tuple(sparse_beta.Dim)).toarray()[
        :, index
    ]
    bias = float(np.asarray(fit["a0"])[index])
    pieces = []
    for module, start in (("yellow", 0), ("tan", 31)):
        pca = read(f"PCA_{module}")
        means = read(f"{module}_module_means")
        if bool(np.asarray(pca["scale"]).item()):
            raise ValueError("Unexpected scaled PCA in the original CellDRIFT artifact")
        features = means.coords["dim_0"].values.astype(str)
        if not np.array_equal(features, pca["rotation"].coords["dim_0"].values):
            raise ValueError(f"{module} PCA and imputation feature orders differ")
        if not np.array_equal(features, pca["center"].coords["dim_0"].values):
            raise ValueError(f"{module} PCA center feature order differs")
        weights = np.asarray(pca["rotation"]) @ pc_weights[start : start + 31]
        bias -= float(np.asarray(pca["center"]) @ weights)
        pieces.append(pd.DataFrame({"weight": weights, "reference": np.asarray(means)}, index=features))
    parameters = pd.concat(pieces)
    if parameters.index.has_duplicates or parameters.shape[0] != 2322:
        raise ValueError("CellDRIFT must contain 2,025 yellow and 297 distinct tan CpGs")
    model = CellDRIFT().double()
    model.version = "0.5.4"
    model.features = parameters.index.tolist()
    model.feature_units = [record["unit"] for record in resolve_feature_ranges(model.features, "DNA methylation")]
    model.reference_values = parameters["reference"].tolist()
    model.base_model = LinearModel(len(model.features)).double()
    with torch.no_grad():
        model.base_model.linear.weight.copy_(torch.from_numpy(parameters["weight"].to_numpy(copy=True)[None]))
        model.base_model.linear.bias.fill_(bias)
    model.metadata = json.loads((ROOT / "clocks/metadata/replication_0.5.4.json").read_text())["celldrift"]
    model.preprocess_name = "supplied NaNs to zero; absent CpGs to module means"
    model.source_provenance = {
        "url": "https://github.com/MorganLevineLab/CellDRIFT",
        "commit": CELLDRIFT_COMMIT,
        "sha256": CELLDRIFT_HASHES,
        "lambda_min": float(cv["lambda.min"].item()),
        "license": "Unspecified in GitHub DESCRIPTION; Zenodo 7693699 lists Other (Open).",
    }
    return model.eval()


MIAGE_COMMIT = "d84d6adfc039355d5f9c962fc5faa7d1054f56b2"
MIAGE_HASHES = {
    "function_library.r": "142cf1297819d2982b84550e6ebe4051373e92c8deeca734c8d79c4161428d49",
    "site_specific_parameters.Rdata": "3c7f3aafcee9728b0b1daa566e93186eea8f736a80041667fb987253dd92863e",
    "Additional_File1.csv": "2997ec6de30ee60079773c05b1164baaab435112d10ee761dd20e8b95821534a",
    "README.txt": "09887418e7d45d47fadeb6676c6ffc445178657affd9d9ebc6502d60f2b66b40",
}


def build_miage(source: Path) -> MiAge:
    """Read the original authors' Rdata object preserved in a pinned archive."""
    import rdata

    verify_sources(source, MIAGE_HASHES)
    parameters = rdata.read_rda(source / "site_specific_parameters.Rdata")["methyl.age"]
    features = pd.read_csv(source / "Additional_File1.csv")["CpG_site_ID"].tolist()
    if len(features) != 268 or len(set(features)) != 268:
        raise ValueError("Expected the original 268 distinct MiAge clock CpGs")
    if any(np.asarray(value).shape != (268,) for value in parameters[:3]):
        raise ValueError("MiAge b/c/d parameters do not match the original CpG list")
    model = MiAge().double()
    model.version = "0.5.4"
    model.features = features
    model.feature_units = [record["unit"] for record in resolve_feature_ranges(model.features, "DNA methylation")]
    model.reference_values = [float("nan")] * len(features)
    model.set_parameters(*parameters[:3])
    model.metadata = json.loads((ROOT / "clocks/metadata/replication_0.5.4.json").read_text())["miage"]
    model.preprocess_name = "omit missing CpGs and supplied NaNs from the least-squares objective"
    model.source_provenance = {
        "original_author_url": "https://www.columbia.edu/~sw2206/softwares/mitotic_age_R_code.zip",
        "original_author_access": "HTTP403 on 2026-10-02; original ZIP bytes could not be verified",
        "archive_mirror_url": (
            "https://github.com/anilpsori/DNAm_pipelines_and_biomarkers/tree/" + MIAGE_COMMIT + "/biomarkers/MiAge"
        ),
        "commit": MIAGE_COMMIT,
        "sha256": MIAGE_HASHES,
        "implementation_source": (
            "Preserved original function_library.r and site_specific_parameters.Rdata; "
            "not the mirror's modified MiAge.R wrapper"
        ),
        "publisher_supplement": "https://doi.org/10.6084/m9.figshare.5620603.v2",
        "publisher_archive_sha256": "998a8b96cbe8261b9c2eb687020d7ac04f1df786d9f7bd2b98c0dbc1e524dbfb",
        "independent_check": (
            "All 268 CpG IDs, gene symbols, chromosomes, coordinates, and informativeness scores "
            "match publisher supplement in order"
        ),
        "license": (
            "No license declared in archived original code/weight files; "
            "publisher supplement CC BY 4.0 covers its distinct annotation table"
        ),
    }
    return model.eval()


def save_models(models, weights_dir: Path, metadata_path: Path) -> None:
    weights_dir.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for model in models:
        name = model.metadata["clock_name"]
        torch.save(model, weights_dir / f"{name}.pt")
        (weights_dir / f"{name}.provenance.json").write_text(json.dumps(model.source_provenance, indent=2) + "\n")
        metadata[name] = dict(model.metadata)
        metadata[name]["preprocess"] = model.preprocess_name
        if model.reference_values is not None:
            metadata[name]["reference_values"] = True
    torch.save(metadata, metadata_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--celldrift-source", type=Path)
    parser.add_argument("--miage-source", type=Path)
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "clocks/weights")
    parser.add_argument("--metadata-path", type=Path, default=ROOT / "clocks/metadata/replication_0.5.4.pt")
    args = parser.parse_args()
    models = []
    if args.celldrift_source:
        models.append(build_celldrift(args.celldrift_source))
    if args.miage_source:
        models.append(build_miage(args.miage_source))
    if not models:
        parser.error("provide at least one original-author source directory")
    save_models(models, args.weights_dir, args.metadata_path)
    for model in models:
        print(f"Built {model.metadata['clock_name']} with {len(model.features)} original-author CpGs")


if __name__ == "__main__":
    main()
