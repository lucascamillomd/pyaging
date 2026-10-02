"""Convert the public MIT PAOPAC Conventional release into portable Torch trees.

Build-only dependency: cryptography. The original Windows loader's public string
table supplies its packaging key. No Windows code or arbitrary pickle globals
are executed; a restricted reader retains inert records and NumPy arrays only.
Only fitted numerical trees are published, not tuning objects or training data.

Run ``uv run --with cryptography python clocks/build_paopac.py``.
"""

import argparse
import collections
import hashlib
import importlib
import io
import json
import pickle
import tempfile
import urllib.request
import zlib
from pathlib import Path

import numpy as np
import torch

from pyaging.models._paopac import PAOPAC, PAOPACTrees

AUTHOR_COMMIT = "a166bc763ffd1395a96c6557db889706fcc9ef85"
RELEASE_COMMIT = "d65afe8610063d6d5ec02decb501c90ef70030c2"
AUTHOR_REPOSITORY = "https://github.com/JackieHanLab/PAOPAC"
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = {
    "model.bin": "5b0c5dadbac0fc382c51151a8845ff6a7ac1da31f89c0579731b02338c10c3f1",
    "pre.cp39-win_amd64.pyd": "c8ecb6f06d0830f3c88dc18e05f5e838042762b3f9f186e125963e089015e85c",
    "LICENSE": "52b3413ff0be04732237f3ec067fb994d51139ddfa61594160ca7b445e7ff3fe",
}
MODEL_TEXT_SHA256 = "10eb9c389ff5fe8fc9adb9c0c27430fcc43dcf628c43ecf42b70e4f8dbfb31d5"


def _read_source(name, source_dir=None):
    cache = Path(source_dir) if source_dir else Path(tempfile.gettempdir()) / "pyaging-paopac" / AUTHOR_COMMIT
    path = cache / name
    if not path.exists():
        if source_dir is not None:
            raise FileNotFoundError(path)
        url = (
            f"{AUTHOR_REPOSITORY}/releases/download/v1.0.0/model.bin"
            if name == "model.bin"
            else f"https://raw.githubusercontent.com/JackieHanLab/PAOPAC/{AUTHOR_COMMIT}/{name}"
        )
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != SOURCE_SHA256[name]:
            raise ValueError(f"PAOPAC source checksum mismatch: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256[name]:
        raise ValueError(f"PAOPAC source checksum mismatch: {name}")
    return data


class _Record:
    """Inert storage for an original model/tuning object's serialized fields."""


class _RestrictedReader(pickle.Unpickler):
    def find_class(self, module, name):
        allowed_records = {
            ("shaphypetune.shaphypetune", "BoostBoruta"),
            ("lightgbm.sklearn", "LGBMRegressor"),
            ("lightgbm.basic", "Booster"),
            ("hyperopt.pyll.base", "Apply"),
            ("hyperopt.pyll.base", "Literal"),
        }
        if (module, name) in allowed_records:
            return _Record
        if module == "numpy.core.multiarray" and name in {"scalar", "_reconstruct"}:
            return getattr(importlib.import_module(module), name)
        if module == "numpy" and name in {"dtype", "ndarray", "argmin"}:
            return getattr(np, name)
        if module == "collections" and name in {"defaultdict", "OrderedDict"}:
            return getattr(collections, name)
        raise ValueError(f"Unsupported pickle global: {module}.{name}")


def extract_conventional(source_dir=None):
    from cryptography.fernet import Fernet

    loader = _read_source("pre.cp39-win_amd64.pyd", source_dir)
    # Standard Cython compressed string table in this checksum-pinned loader.
    public_strings = zlib.decompress(loader[0xA950:])
    key = public_strings[-44:]
    packed = Fernet(key).decrypt(_read_source("model.bin", source_dir))
    data = zlib.decompress(packed)
    if hashlib.sha256(data).hexdigest() != "e5db3a2d90cade9d8bec2b9f879635a275fc190c52e155eef243c48a771b9fa3":
        raise ValueError("Unexpected PAOPAC decoded model payload")
    models = _RestrictedReader(io.BytesIO(data)).load()
    booster = models["Conventional"].estimator_._Booster
    # Both legacy and current LightGBM state are supplied. Their only difference
    # is the format-version header; numerical trees and feature names coincide.
    text = booster._handle
    if booster.handle.replace("version=v3\n", "version=v4\n", 1) != text:
        raise ValueError("PAOPAC legacy and current tree parameters differ")
    if hashlib.sha256(text.encode()).hexdigest() != MODEL_TEXT_SHA256:
        raise ValueError("Unexpected PAOPAC Conventional tree payload")
    return text


def convert_trees(text):
    """Parse only the exact regression tree format present in the author model."""
    header, *chunks = text.split("\nTree=")
    fields = dict(line.split("=", 1) for line in header.splitlines() if "=" in line)
    if fields.get("objective") != "regression" or fields.get("num_class") != "1":
        raise ValueError("Only the released PAOPAC regression model is supported")
    names = fields["feature_names"].split()
    arrays = {k: [] for k in ("features", "thresholds", "left", "right", "values", "roots")}
    max_depth = 0
    for chunk in chunks:
        tree = dict(line.split("=", 1) for line in chunk.splitlines() if "=" in line)
        n_leaves = int(tree["num_leaves"])
        n_nodes = n_leaves - 1
        if int(tree["num_cat"]) != 0 or set(tree["decision_type"].split()) != {"2"}:
            raise ValueError("The pinned release must contain numeric, default-left splits only")
        split_features = list(map(int, tree["split_feature"].split()))
        thresholds = list(map(float, tree["threshold"].split()))
        left = list(map(int, tree["left_child"].split()))
        right = list(map(int, tree["right_child"].split()))
        values = list(map(float, tree["leaf_value"].split()))
        if any(len(a) != n_nodes for a in (split_features, thresholds, left, right)) or len(values) != n_leaves:
            raise ValueError("Malformed PAOPAC tree arrays")
        offset = len(arrays["features"])
        arrays["roots"].append(offset)
        arrays["features"].extend(split_features + [0] * n_leaves)
        arrays["thresholds"].extend(thresholds + [0.0] * n_leaves)
        arrays["values"].extend([0.0] * n_nodes + values)
        for key, children in (("left", left), ("right", right)):
            arrays[key].extend(offset + (c if c >= 0 else n_nodes - c - 1) for c in children)
            arrays[key].extend(range(offset + n_nodes, offset + n_nodes + n_leaves))

        def depth(node, left=left, right=right):
            return 0 if node < 0 else 1 + max(depth(left[node]), depth(right[node]))

        max_depth = max(max_depth, depth(0))
    if len(chunks) != 800 or len(names) != 190 or names[-1] != "TDI":
        raise ValueError("Unexpected PAOPAC Conventional model dimensions")
    return names, PAOPACTrees(**arrays, depth=max_depth)


def clock_metadata():
    return {
        "clock_name": "paopac",
        "data_type": "proteomics",
        "species": "Homo sapiens",
        "year": 2026,
        "approved_by_author": "⌛",
        "research_only": True,
        "citation": "Xu, H., et al. Proteome-aware organ proxy aging clocks. bioRxiv (2026).",
        "doi": "https://doi.org/10.64898/2026.04.24.720503",
        "notes": (
            "Conventional PAOPAC from the authors' MIT v1.0.0 release: 800 LightGBM trees with 189 Olink "
            "protein predictors and TDI, plus chronological age for cohort correction. Supply plasma Olink "
            "Explore 3072 NPX (assay-normalized log2 relative abundance), not linear abundance, concentrations, "
            "counts or pre-standardized values. NPX labeling alone does not establish equivalence across "
            "assays, panels, batches or plasma and serum. Exact duplicate protein columns are averaged on "
            "the NPX scale, then names uppercased; newly colliding names are rejected. The original interface "
            "applies 2**NPX, fills missing or absent values "
            "with zero on that linear scale, and fits StandardScaler over the complete supplied cohort. "
            "TDI is optional and read only from the protein matrix; it is exponentiated and standardized "
            "like the other inputs. TDI in sample metadata is ignored; absent TDI becomes zero. Required "
            "chronological age is supplied as an age column in years. Final output is raw predicted age "
            "minus LOWESS(raw predicted age minus chronological age, chronological age), with frac=0.75 "
            "and the original statsmodels defaults. Predictions depend on cohort composition, including "
            "cohort ages; this is not a fixed per-sample age or a portable age-acceleration residual. "
            "Very small or age-degenerate cohorts can produce trivial corrections. No fixed UKB scaler "
            "or LOWESS curve is supplied by the original release. Preserve NaNs when preparing AnnData "
            "to reproduce the author's missing-value rule. Nature Biotechnology trial supplement S2 "
            "identifies the Conventional endpoint; exact identity of that trial's private model artifact "
            "with this later public release has not been verified. MIT source; preprint CC BY 4.0. "
            "No author approval of this conversion is claimed."
        ),
        "tissue": ["plasma"],
        "predicts": ["chronological age"],
        "training_target": ["chronological age"],
        "unit": ["years"],
        "model_type": "LightGBM with cohort standardization and LOWESS correction",
        "platform": ["Olink Explore 3072"],
        "population": "adults",
        "journal": "bioRxiv",
        "last_author": "Jing-Dong J. Han",
        "n_features": 191,
        "citations": 1,
        "citations_date": "2026-10-02",
    }


def build_clock(source_dir=None):
    text = extract_conventional(source_dir)
    features, trees = convert_trees(text)
    model = PAOPAC()
    model.features = [*features, "age"]
    model.base_model_features = list(model.features)
    model.feature_units = ["NPX"] * 189 + ["Townsend deprivation index; author protein-matrix transform", "years"]
    model.reference_values = [float("nan")] * 191
    model.base_model = trees
    model.metadata = clock_metadata()
    model.preprocess_name = "author 2**NPX, linear-scale zero fill and cohort StandardScaler"
    model.postprocess_name = "author cohort LOWESS age-bias correction (frac=0.75)"
    model.version = "0.5.6"
    model.license = "MIT"
    model.license_text = _read_source("LICENSE", source_dir).decode()
    model.provenance = {
        "repository": AUTHOR_REPOSITORY,
        "commit": AUTHOR_COMMIT,
        "release": "v1.0.0",
        "release_commit": RELEASE_COMMIT,
        "source_sha256": SOURCE_SHA256,
        "model_text_sha256": MODEL_TEXT_SHA256,
        "model_key": "Conventional",
        "trees": 800,
        "tree_input_count": 190,
        "protein_count": 189,
        "cohort_age_dependency_count": 1,
        "preprocessing_oracle": "Unmodified pre.cp39-win_amd64.pyd predict() under CPython 3.9.13 / Wine 8.0",
        "packaging": "Public MIT loader key; Fernet then zlib; restricted inert-record pickle reader",
        "retained_state": "Fitted numeric split features, thresholds, children, leaf values only",
        "training_participant_records_retained": False,
        "citation_count_source": "https://api.openalex.org/works/https://doi.org/10.64898/2026.04.24.720503",
    }
    return model.eval()


def save_clock(model, output_dir=None):
    output = Path(output_dir) if output_dir else REPO_ROOT / "clocks" / "weights"
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model, output / "paopac.pt")
    (output / "paopac.provenance.json").write_text(json.dumps(model.provenance, indent=2) + "\n")
    (output / "paopac.LICENSE.txt").write_text(model.license_text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    model = build_clock(args.source_dir)
    save_clock(model, args.output_dir)


if __name__ == "__main__":
    main()
