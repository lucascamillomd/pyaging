"""Build PAC from the checksum-pinned original Kuo et al. R release.

Run ``uv run python clocks/build_proteoclock.py`` from the repository root.
The Insilico proteoclock wrapper is not used as PAC's coefficient source: it
rounds Gompertz constants and applies optional, separate cohort preprocessing.
ipfP3GPT is not built because the authors withdrew its restricted UKB weights.
"""

import argparse
import hashlib
import json
import re
import tempfile
import urllib.request
from pathlib import Path

import torch

from pyaging.models._proteoclock import PAC

AUTHOR_COMMIT = "e15edb66800f876a928cec9c8856500048c46fea"
AUTHOR_REPOSITORY = "https://github.com/kuo-lab-uchc/PAC"
AUTHOR_URL = f"https://raw.githubusercontent.com/kuo-lab-uchc/PAC/{AUTHOR_COMMIT}/"
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = {
    "README.md": "011e265bbfbdfd102e06b2b669433921cf7e4bd87b97b23fa68421abc1201d9f",
    "pac_proteomic_age.R": "b87edc70f11e34740f0323e4b7c677eb1d53c5008d631ecf87a35cd22be1d2ff",
    "pac_example_data.csv": "00d7360bbac1c8a750e29332a4bda080d94780d180cbb1cbf5bcc7bd47612f9f",
}


def _read_source(path, source_dir=None):
    cache = Path(source_dir) if source_dir is not None else Path(tempfile.gettempdir()) / "pyaging-pac" / AUTHOR_COMMIT
    target = cache / path
    if not target.exists():
        if source_dir is not None:
            raise FileNotFoundError(f"Original-author file is absent: {target}")
        with urllib.request.urlopen(AUTHOR_URL + path, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256[path]:
            raise ValueError(f"Original-author source checksum mismatch: {path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    payload = target.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256[path]:
        raise ValueError(f"Original-author source checksum mismatch: {path}")
    return payload.decode("utf-8")


def _parse_author_parameters(source):
    """Extract factual fitted numbers without executing downloaded R code."""
    predictors = re.search(r"predictors=c\((.*?)\)", source, re.DOTALL)
    betas = re.search(r"betas=c\((.*?)\)", source, re.DOTALL)
    if predictors is None or betas is None:
        raise ValueError("PAC source must define predictors and betas")
    features = re.findall(r'"([^\"]+)"', predictors.group(1))
    coefficients = [float(value.strip()) for value in betas.group(1).split(",")]
    parameters = []
    for name in ("shape", "rate", "shape0", "rate0", "beta_age"):
        match = re.search(r"^\s*" + name + r"=([-+\d.eE]+)", source, re.MULTILINE)
        if match is None:
            raise ValueError(f"PAC source lacks {name}")
        parameters.append(float(match.group(1)))
    if len(features) != 129 or len(coefficients) != 129 or len(set(features)) != 129 or features[0] != "age":
        raise ValueError("PAC requires age followed by 128 unique original protein predictors")
    if not torch.isfinite(torch.tensor([*coefficients, *parameters], dtype=torch.float64)).all():
        raise ValueError("PAC source contains nonfinite parameters")
    if min(parameters) <= 0:
        raise ValueError("PAC Gompertz parameters must be positive")
    return features, coefficients, parameters


def clock_metadata():
    return {
        "clock_name": "pac",
        "data_type": "proteomics",
        "species": "Homo sapiens",
        "year": 2024,
        "approved_by_author": "⌛",
        "citation": (
            'Kuo, C.-L., et al. "Proteomic aging clock (PAC) predicts age-related outcomes in '
            'middle-aged and older adults." Aging Cell 23, e14195 (2024).'
        ),
        "doi": "https://doi.org/10.1111/acel.14195",
        "notes": (
            "Mortality-equivalent proteomic age from the original Kuo et al. R release, using all 128 Olink "
            "Explore 3072 plasma protein NPX values plus chronological age in years. Supply original lowercase "
            "identifiers, including fut3_fut5 and ntprobnp; there is no sex covariate. NPX is assay-normalized "
            "log2 relative abundance, not raw counts, concentrations, linear abundances or z-scores. "
            "NPX labeling alone does not establish comparability across panels, batches, plasma and serum. "
            "No cohort scaling, "
            "normalization, or imputation is applied during scoring. Every predictor column is required; "
            "supplied NaNs propagate. The paper used k-nearest-neighbor imputation (k=10) before model fitting; "
            "any upstream imputation is a separate caller decision. The full-precision original Gompertz constants "
            "are retained, without the later Insilico wrapper's rounding or optional distribution matching. "
            "Output is mortality-equivalent age in years, not a clinical diagnosis or an age-acceleration residual. "
            "Development cohort: UK Biobank adults aged 39-70. The original repository specifies no code license; "
            "the publication is CC BY 4.0. No author approval of this conversion is claimed."
        ),
        "research_only": True,
        "tissue": ["plasma"],
        "predicts": ["biological age"],
        "training_target": ["mortality"],
        "unit": ["years"],
        "model_type": "Gompertz proportional hazards model",
        "platform": ["Olink Explore 3072"],
        "population": "adults",
        "journal": "Aging Cell",
        "last_author": "Breno S. Diniz",
        "n_features": 129,
        "citations": 68,
        "citations_date": "2026-10-02",
    }


def build_clock(name="pac", source_dir=None):
    if name != "pac":
        raise ValueError(f"Unknown supported proteomic clock: {name}")
    features, coefficients, parameters = _parse_author_parameters(_read_source("pac_proteomic_age.R", source_dir))
    model = PAC()
    model.features = features
    model.base_model_features = list(features)
    model.feature_units = ["years"] + ["NPX"] * 128
    model.base_model = torch.nn.Linear(129, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.base_model.weight.copy_(torch.tensor([coefficients], dtype=torch.float64))
        model.gompertz_parameters.copy_(torch.tensor(parameters, dtype=torch.float64))
    model.metadata = clock_metadata()
    model.version = "0.5.5"
    model.license = "Original repository license unspecified; publication CC BY 4.0"
    model.license_text = (
        "PAC model parameters: Kuo et al., Aging Cell 23, e14195 (2024). DOI: 10.1111/acel.14195.\n"
        "The original author repository supplies no license file; its code is not relicensed here.\n"
        "The publication is licensed under Creative Commons Attribution 4.0 International.\n"
        "https://creativecommons.org/licenses/by/4.0/\n"
        "Pyaging's Torch implementation extracts fitted numerical parameters from the cited original release.\n"
    )
    model.provenance = {
        "repository": AUTHOR_REPOSITORY,
        "commit": AUTHOR_COMMIT,
        "coefficient_path": "pac_proteomic_age.R",
        "coefficient_sha256": SOURCE_SHA256["pac_proteomic_age.R"],
        "prediction_code_path": "pac_proteomic_age.R",
        "prediction_code_sha256": SOURCE_SHA256["pac_proteomic_age.R"],
        "paper": "https://doi.org/10.1111/acel.14195",
        "input_scale": "Olink NPX, with chronological age in years; no scoring-time scaling",
        "missingness": "All predictor columns required; supplied NaNs propagate; no reference-value imputation",
        "original_repository_license": "unspecified",
        "publication_license": "CC BY 4.0",
    }
    return model.eval()


def save_clock(model, output_dir=None):
    output = Path(output_dir) if output_dir is not None else REPO_ROOT / "clocks" / "weights"
    output.mkdir(parents=True, exist_ok=True)
    name = model.metadata["clock_name"]
    torch.save(model, output / f"{name}.pt")
    (output / f"{name}.provenance.json").write_text(json.dumps(model.provenance, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    model = build_clock(source_dir=args.source_dir)
    save_clock(model, output_dir=args.output_dir)
    metadata = {"pac": model.metadata}
    path = REPO_ROOT / "clocks" / "metadata" / "proteoclock_0.5.5.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(f"Built PAC from original author commit {AUTHOR_COMMIT}")


if __name__ == "__main__":
    main()
