"""Build Healthspan Proteomic Score from the pinned original Kuo lab R release.

Run ``uv run python clocks/build_hps.py`` from the repository root. Fitted
numerical parameters are extracted directly from the author release; neither
Biolearn nor MethylCYPHER is used as a source.
"""

import argparse
import hashlib
import json
import re
import tempfile
import urllib.request
from pathlib import Path

import torch

from pyaging.models._proteoclock import HPS

AUTHOR_COMMIT = "6eec9ac5a091e34fece69fa4652ebd9af22c1839"
AUTHOR_REPOSITORY = "https://github.com/kuo-lab-uchc/HPS"
AUTHOR_URL = f"https://raw.githubusercontent.com/kuo-lab-uchc/HPS/{AUTHOR_COMMIT}/"
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = {
    "README.md": "c5f29dd0c21773976c0dc0eea1395169a0009735a0e02b42a45f1b3d59ef3b24",
    "HPS.R": "e55df4dd50313869fa4d94f989391d36fba8b8594b8579e189be4cfdc930c4c3",
    "hps_example_data.csv": "af2bd95e5ead0ea9c07f0eb855fa463c71c90cd5467be68dfa97385b23c4b8df",
}


def _read_source(path, source_dir=None):
    cache = Path(source_dir) if source_dir is not None else Path(tempfile.gettempdir()) / "pyaging-hps" / AUTHOR_COMMIT
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
    """Read the published fitted numbers without executing downloaded code."""
    predictors = re.search(r"predictors=c\((.*?)\)", source, re.DOTALL)
    betas = re.search(r"betas=c\((.*?)\)", source, re.DOTALL)
    if predictors is None or betas is None:
        raise ValueError("HPS source must define predictors and betas")
    features = re.findall(r'"([^\"]+)"', predictors.group(1))
    coefficients = [float(value.strip()) for value in betas.group(1).split(",")]
    parameters = []
    for name in ("shape", "rate"):
        match = re.search(r"^\s*" + name + r"=([-+\d.eE]+)", source, re.MULTILINE)
        if match is None:
            raise ValueError(f"HPS source lacks {name}")
        parameters.append(float(match.group(1)))
    if len(features) != 87 or len(coefficients) != 87 or len(set(features)) != 87 or features[0] != "age":
        raise ValueError("HPS requires age followed by 86 unique original protein predictors")
    if not torch.isfinite(torch.tensor([*coefficients, *parameters], dtype=torch.float64)).all():
        raise ValueError("HPS source contains nonfinite parameters")
    if min(parameters) <= 0:
        raise ValueError("HPS Gompertz parameters must be positive")
    return features, coefficients, parameters


def clock_metadata():
    return {
        "clock_name": "hps",
        "data_type": "proteomics",
        "species": "Homo sapiens",
        "year": 2025,
        "approved_by_author": "⌛",
        "citation": (
            'Kuo, C.-L., et al. "A proteomic signature of healthspan." '
            "Proceedings of the National Academy of Sciences 122, e2414086122 (2025)."
        ),
        "doi": "https://doi.org/10.1073/pnas.2414086122",
        "notes": (
            "Healthspan Proteomic Score (HPS) from the original Kuo lab R release: 86 Olink Explore 3072 "
            "plasma protein NPX values plus chronological age in years. Returns the modeled probability "
            "of remaining free of the study's major-disease-or-death endpoint over the next 10 years, "
            "on a 0-1 scale; higher is healthier. It is not predicted years of life, biological age, "
            "or an age-adjusted residual. The endpoint includes cancer excluding nonmelanoma skin cancer, "
            "diabetes, heart failure, myocardial infarction, stroke, COPD, dementia, or death. Supply "
            "original lowercase identifiers, including separate nppb and ntprobnp assays. NPX is "
            "assay-normalized log2 relative abundance, not raw counts, concentrations, linear abundances "
            "or z-scores. The study used within-batch and across-batch intensity normalization; NPX labeling "
            "alone does not establish comparability across panels, batches, plasma and serum. No additional "
            "scaling, normalization or imputation occurs during scoring. All 87 predictor columns are "
            "required; supplied NaNs propagate. The study used upstream k-nearest-neighbor imputation "
            "(k=10, multiUS); any imputation is a separate caller decision. The Gompertz model was fit "
            "in 30,184 UK Biobank adults initially free of the specified endpoint conditions; "
            "the healthy cohort age range was 39-70. "
            "The August 2025 correction only changes author affiliations. Research use only. The original "
            "repository specifies no code license; the publication is CC BY-NC-ND 4.0. This independent "
            "Torch implementation uses the published fitted numerical parameters and does not relicense "
            "the authors' R code. No author approval of this conversion is claimed."
        ),
        "research_only": True,
        "tissue": ["plasma"],
        "predicts": ["healthspan probability"],
        "training_target": ["healthspan"],
        "unit": ["probability"],
        "model_type": "Gompertz proportional hazards model",
        "platform": ["Olink Explore 3072"],
        "population": "adults",
        "journal": "Proceedings of the National Academy of Sciences",
        "last_author": "Breno S. Diniz",
        "n_features": 87,
        "citations": 21,
        "citations_date": "2026-10-02",
    }


def build_clock(source_dir=None):
    features, coefficients, parameters = _parse_author_parameters(_read_source("HPS.R", source_dir))
    model = HPS()
    model.features = features
    model.base_model_features = list(features)
    model.feature_units = ["years"] + ["NPX"] * 86
    model.base_model = torch.nn.Linear(87, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.base_model.weight.copy_(torch.tensor([coefficients], dtype=torch.float64))
        model.gompertz_parameters.copy_(torch.tensor(parameters, dtype=torch.float64))
    model.metadata = clock_metadata()
    model.version = "0.5.6"
    model.license = "Original repository license unspecified; publication CC BY-NC-ND 4.0; research use only"
    model.provenance = {
        "repository": AUTHOR_REPOSITORY,
        "commit": AUTHOR_COMMIT,
        "coefficient_path": "HPS.R",
        "coefficient_sha256": SOURCE_SHA256["HPS.R"],
        "prediction_code_path": "HPS.R",
        "prediction_code_sha256": SOURCE_SHA256["HPS.R"],
        "paper": "https://doi.org/10.1073/pnas.2414086122",
        "correction": "https://doi.org/10.1073/pnas.2520058122",
        "correction_scope": "Author affiliations only; no changes to the model",
        "input_scale": "Olink NPX, with chronological age in years; no scoring-time scaling",
        "missingness": "All predictor columns required; supplied NaNs propagate; no reference-value imputation",
        "output": "10-year disease/death-free probability; 0-1, higher is healthier",
        "original_repository_license": "unspecified",
        "publication_license": "CC BY-NC-ND 4.0",
        "research_only": True,
    }
    return model.eval()


def save_clock(model, output_dir=None):
    output = Path(output_dir) if output_dir is not None else REPO_ROOT / "clocks" / "weights"
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model, output / "hps.pt")
    (output / "hps.provenance.json").write_text(json.dumps(model.provenance, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    save_clock(build_clock(args.source_dir), args.output_dir)
    print(f"Built HPS from original author commit {AUTHOR_COMMIT}")


if __name__ == "__main__":
    main()
