"""Build PCBrainAge and eight PCGrimAge proxies from author-derived coefficients.

Normal deterministic rebuild (no R or network required)::

    uv run python clocks/build_pc_extensions.py

Regenerate the compact coefficients and independent R oracle first::

    uv run python clocks/build_pc_extensions.py --source-dir /tmp/pc-sources \
        --rscript /path/to/Rscript --regenerate

The source directory must contain the two pinned author repository checkouts
``PC-Clocks`` and ``calcPCBrainAge`` and ``CalcAllPCClocks.RData`` downloaded
from the authors' Yale Box distribution. See tests/data/pc_extensions/README.md.
No methylCIPHER or Biolearn models are inputs to this builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pyaging.models import LinearModel, _pc_extensions
from pyaging.utils import resolve_feature_ranges

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data/pc_extensions"
METADATA = ROOT / "clocks/metadata/pc_extensions_0.5.4.json"
CLASSES = {
    name.lower(): getattr(_pc_extensions, name)
    for name in (
        "PCBrainAge",
        "PCGrimAgePackYrs",
        "PCGrimAgeADM",
        "PCGrimAgeB2M",
        "PCGrimAgeCystatinC",
        "PCGrimAgeGDF15",
        "PCGrimAgeLeptin",
        "PCGrimAgePAI1",
        "PCGrimAgeTIMP1",
    )
}


def build_model(name: str, coefficients: Path = DATA / "coefficients.npz"):
    """Load compact double-precision author coefficients into a runtime clock."""
    if name not in CLASSES:
        raise ValueError(f"Unknown PC extension: {name}")
    with np.load(coefficients, allow_pickle=False) as data:
        features = data[f"{name}_features"].tolist()
        weights = np.array(data[f"{name}_weights"], dtype=np.float64)
        reference = np.array(data[f"{name}_reference"], dtype=np.float64)
        bias = float(data[f"{name}_bias"].item())
    if not (len(features) == len(weights) == len(reference)) or len(set(features)) != len(features):
        raise ValueError(f"{name}: inconsistent feature/weight/reference arrays")
    covariate_indices = [i for i, feature in enumerate(features) if feature in {"female", "age"}]
    methylation_mask = np.ones(len(features), dtype=bool)
    methylation_mask[covariate_indices] = False
    if not (
        np.isfinite(weights).all()
        and np.isfinite(reference[methylation_mask]).all()
        and np.isnan(reference[covariate_indices]).all()
        and np.isfinite(bias)
    ):
        raise ValueError(f"{name}: coefficients must be finite")
    model = CLASSES[name]().double()
    model.features = features
    if covariate_indices:
        model.required_covariates = [features[i] for i in covariate_indices]
        model.required_covariate_indices = torch.tensor(covariate_indices, dtype=torch.long)
    model.reference_values = reference.tolist()
    model.base_model = LinearModel(len(features)).double()
    with torch.no_grad():
        model.base_model.linear.weight.copy_(torch.from_numpy(weights[None]))
        model.base_model.linear.bias.fill_(bias)
    model.metadata = json.loads(METADATA.read_text())[name]
    model.version = model.metadata["version"]
    if model.metadata["n_features"] != len(features):
        raise ValueError(f"{name}: metadata feature count does not match coefficients")
    ranges = resolve_feature_ranges(features, model.metadata["data_type"])
    model.feature_units = [entry["unit"] for entry in ranges]
    return model.eval()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def regenerate(source_dir: Path, rscript: str):
    """Verify pinned source files, run R, and write lossless compact fixtures."""
    manifest = json.loads((DATA / "provenance.json").read_text())
    for relative, expected in manifest["source_sha256"].items():
        source = source_dir / relative
        if _sha256(source) != expected:
            raise ValueError(f"Original source checksum mismatch: {relative}")
    with tempfile.TemporaryDirectory(prefix="pyaging-pc-export-") as temp:
        temp = Path(temp)
        subprocess.run(
            [rscript, "--vanilla", str(DATA / "export_author_models.R"), str(source_dir), str(temp)],
            check=True,
        )
        arrays = {}
        oracle = {}
        for name in CLASSES:
            frame = pd.read_csv(temp / f"{name}.tsv.gz", sep="\t", float_precision="round_trip")
            arrays[f"{name}_features"] = frame["feature"].to_numpy(dtype=str)
            arrays[f"{name}_weights"] = frame["weight"].to_numpy(dtype=np.float64)
            arrays[f"{name}_reference"] = frame["reference"].to_numpy(dtype=np.float64)
            arrays[f"{name}_bias"] = np.array(float((temp / f"{name}.bias").read_text()))
            oracle[name] = np.loadtxt(temp / f"{name}.oracle").tolist()
        np.savez_compressed(DATA / "coefficients.npz", **arrays)
        (DATA / "oracle.json").write_text(json.dumps(oracle, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--clock", choices=sorted(CLASSES))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "clocks/weights")
    args = parser.parse_args()
    if args.regenerate:
        if args.source_dir is None:
            parser.error("--regenerate requires --source-dir")
        regenerate(args.source_dir, args.rscript)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = [args.clock] if args.clock else sorted(CLASSES)
    aggregate = {}
    for name in names:
        model = build_model(name)
        target = args.output_dir / f"{name}.pt"
        torch.save(model, target)
        provenance = json.loads((DATA / "provenance.json").read_text())
        provenance["clock_name"] = name
        original_name = {
            "pcbrainage": "PCBrainAge_Model",
            "pcgrimagepackyrs": "PCPACKYRS",
            "pcgrimageadm": "PCADM",
            "pcgrimageb2m": "PCB2M",
            "pcgrimagecystatinc": "PCCystatinC",
            "pcgrimagegdf15": "PCGDF15",
            "pcgrimageleptin": "PCLeptin",
            "pcgrimagepai1": "PCPAI1",
            "pcgrimagetimp1": "PCTIMP1",
        }[name]
        provenance["source_model"] = original_name
        provenance["artifact_sha256"] = _sha256(target)
        provenance["coefficient_archive_sha256"] = _sha256(DATA / "coefficients.npz")
        target.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        if name == "pcbrainage":
            target.with_suffix(".LICENSE.txt").write_text(
                "The original calcPCBrainAge DESCRIPTION declares: License: MIT License.\n"
                "Source: https://github.com/MorganLevineLab/calcPCBrainAge/blob/"
                "8a4a0bf93fe715e48d495af46edf17f10d988afa/DESCRIPTION\n"
                "The pinned upstream repository does not include a separate full license file.\n"
                "Author: Kyra L. Thrush. This notice preserves the upstream declaration; "
                "it does not substitute the pyaging package license for upstream terms.\n"
            )
        aggregate[name] = {**model.metadata, "version": model.version, "reference_values": True}
        print(f"{name}: {len(model.features):,} features -> {target}")
    if args.clock is None and args.output_dir.resolve() == (ROOT / "clocks/weights").resolve():
        torch.save(aggregate, ROOT / "clocks/metadata/pc_extensions_0.5.4.pt")


if __name__ == "__main__":
    main()
