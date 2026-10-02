"""Build IntrinClock370 and PhenoAge partitions from original author assets.

Run from the repository root:
    uv run --with rdata --with remotezip python clocks/build_author_extensions.py \
        --intrin-source /tmp/intrinclock-original --prc-source /tmp/pyaging-prc-original

Without a source directory the pinned files are downloaded into --cache-dir.
The aggregate metadata registry is updated separately during release integration.
"""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import torch

from pyaging.models._author_extensions import DNAmPhenoAgeNonPRC, DNAmPhenoAgePRC, IntrinClock370
from pyaging.models._base_models import LinearModel
from pyaging.utils import resolve_feature_ranges

ROOT = Path(__file__).resolve().parents[1]
PRC_COMMIT = "0af213c60035d8dd568d9d9ba2df90cfbca19596"
PRC_URL = f"https://raw.githubusercontent.com/MorganLevineLab/prcPhenoAge/{PRC_COMMIT}/"
INTRIN_URL = "https://zenodo.org/api/records/10426597/files/Tomusiak2023_code.zip/content"
PRC_SHA256 = "089e2a3661d604c2822f56710b4076d2e36e07e2d2ce64de61f2958b3e9d0043"
INTRIN_SHA256 = "79c2d1b624c494156c7ee213e2eb62dbb90c133f8ddf4938982a90c301b0c9fa"
INTRIN_DEMO_SHA256 = "90517825578d294ee76ef43cc384b6865cb30625b04d0f35df2299d33fe3b324"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources(cache, intrin_source=None, prc_source=None):
    intrin = Path(intrin_source) if intrin_source else cache / "intrinclock"
    prc = Path(prc_source) if prc_source else cache / "prcPhenoAge"
    intrin.mkdir(parents=True, exist_ok=True)
    prc.mkdir(parents=True, exist_ok=True)
    if not all((intrin / name).exists() for name in ("final_model_small.RData", "demo.R")):
        from remotezip import RemoteZip

        with RemoteZip(INTRIN_URL) as archive:
            for basename in ("final_model_small.RData", "demo.R"):
                matches = [name for name in archive.namelist() if name.split("/")[-1] == basename]
                if len(matches) != 1:
                    raise ValueError(f"Ambiguous original archive member: {basename}")
                (intrin / basename).write_bytes(archive.read(matches[0]))
    csv = prc / "data-raw/Clock_Levine_PRC.csv"
    if not csv.exists():
        csv.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(PRC_URL + "data-raw/Clock_Levine_PRC.csv", timeout=120) as response:
            csv.write_bytes(response.read())
    if sha256(csv) != PRC_SHA256:
        raise ValueError("The pinned author PhenoAge coefficient checksum does not match")
    if sha256(intrin / "final_model_small.RData") != INTRIN_SHA256:
        raise ValueError("The original IntrinClock model checksum does not match")
    if sha256(intrin / "demo.R") != INTRIN_DEMO_SHA256:
        raise ValueError("The original IntrinClock demo checksum does not match")
    return intrin, prc


def linear_model(cls, features, coefficients, intercept=0.0):
    model = cls()
    model.features = list(features)
    model.base_model_features = model.features
    model.feature_units = [record["unit"] for record in resolve_feature_ranges(model.features, "DNA methylation")]
    model.base_model = LinearModel(len(features)).double()
    with torch.no_grad():
        model.base_model.linear.weight.copy_(torch.tensor(np.asarray(coefficients)[None, :], dtype=torch.float64))
        model.base_model.linear.bias.fill_(intercept)
    return model.eval()


def build_models(intrin, prc):
    import rdata
    from scipy.sparse import csc_matrix

    original = rdata.read_rds(intrin / "final_model_small.RData")
    fit = original["glmnet.fit"]
    index = np.flatnonzero(fit["lambda"] == original["lambda.1se"][0])
    if len(index) != 1:
        raise ValueError("Author lambda.1se must uniquely identify an original fitted column")
    index = int(index[0])
    beta = fit["beta"]
    coefficients = csc_matrix((beta.x, beta.i, beta.p), shape=tuple(beta.Dim)).getcol(index).toarray().ravel()
    selected = coefficients != 0
    if selected.sum() != 370:
        raise ValueError("Expected 370 selected CpGs in the original lambda.1se fit")
    model = linear_model(IntrinClock370, beta.Dimnames[0][selected], coefficients[selected], float(fit["a0"][index]))
    model.postprocess_name = "anti_log_linear"
    model.provenance = {
        "source_url": INTRIN_URL,
        "source_sha256": sha256(intrin / "final_model_small.RData"),
        "demo_sha256": INTRIN_DEMO_SHA256,
        "selection": "cv.glmnet lambda.1se",
        "lambda": float(original["lambda.1se"][0]),
        "column_index_r": index + 1,
    }
    models = {"intrinclock370": model}
    table = pd.read_csv(prc / "data-raw/Clock_Levine_PRC.csv")
    is_prc = table["PRC"].fillna(0).eq(1)
    for name, cls, mask, count in (
        ("dnamphenoageprc", DNAmPhenoAgePRC, is_prc, 55),
        ("dnamphenoagenonprc", DNAmPhenoAgeNonPRC, ~is_prc, 458),
    ):
        subset = table.loc[mask]
        if len(subset) != count:
            raise ValueError(f"Incorrect original partition size for {name}")
        model = linear_model(cls, subset["CpG"], subset["Weight"])
        model.preprocess_name = "missing_beta_zero_contribution"
        model.provenance = {
            "source_url": PRC_URL + "data-raw/Clock_Levine_PRC.csv",
            "source_commit": PRC_COMMIT,
            "source_sha256": PRC_SHA256,
            "intercept": "none, matching author calcPRCPhenoAge/calcnonPRCPhenoAge",
            "imputation": "FALSE: absent CpGs and supplied NA beta values contribute zero",
        }
        models[name] = model
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/pyaging-author-extensions"))
    parser.add_argument("--intrin-source", type=Path)
    parser.add_argument("--prc-source", type=Path)
    args = parser.parse_args()
    intrin, prc = sources(args.cache_dir, args.intrin_source, args.prc_source)
    registry = json.loads((ROOT / "clocks/metadata/author_extensions_0.5.4.json").read_text())
    destination = ROOT / "clocks/weights"
    destination.mkdir(parents=True, exist_ok=True)
    for name, model in build_models(intrin, prc).items():
        model.metadata.update(registry[name])
        model.metadata["version"] = "0.5.4"
        model.version = "0.5.4"
        torch.save(model, destination / f"{name}.pt")
        (destination / f"{name}.provenance.json").write_text(json.dumps(model.provenance, indent=2) + "\n")
        print(f"Built {name}: {len(model.features)} features")


if __name__ == "__main__":
    main()
