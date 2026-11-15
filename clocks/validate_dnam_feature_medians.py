"""Compare package and zero fill on complete clock inputs with masked probes."""

import argparse
import csv
import gzip
import json
from pathlib import Path
from unittest.mock import patch

import anndata
import numpy as np
import pandas as pd
import torch

import pyaging as pya
from pyaging.predict._pred_utils import _load_dnam_feature_medians


def read_validation_betas(path, features, *, blood):
    ages = None
    parts = []
    with gzip.open(path, "rt") as stream:
        if blood:
            for line in stream:
                if line.startswith("!Sample_characteristics") and '"age: ' in line:
                    values = next(csv.reader([line], delimiter="\t"))[1:]
                    ages = [float(value.removeprefix("age: ")) for value in values]
                if line.startswith("!series_matrix_table_begin"):
                    break
        with pd.read_csv(
            stream,
            sep="\t" if blood else ",",
            index_col=0,
            comment="!" if blood else None,
            usecols=None if blood else lambda name: not name.endswith(" Detection Pval"),
            chunksize=4096,
        ) as chunks:
            for chunk in chunks:
                parts.append(chunk.loc[chunk.index.isin(features)])
    beta = pd.concat(parts).T
    if ages is not None:
        ages = pd.Series(ages, index=beta.index)
    return beta, ages


def compare(beta, ages, model, panel=None):
    clock = model.metadata["clock_name"]
    if model.reference_values is not None:
        raise ValueError(f"{clock} has model references and cannot exercise the fallback")
    if not set(model.features).issubset(beta.columns):
        return {"clock": clock, "missing_baseline_features": len(set(model.features).difference(beta.columns))}
    beta = beta[model.features]
    beta = beta.loc[(beta.ge(0) & beta.le(1)).all(axis=1)]
    if beta.empty:
        return {"clock": clock, "samples": 0}

    def predict(frame, *, use_medians):
        model.use_dnam_median_fallback = use_medians
        adata = anndata.AnnData(
            frame.to_numpy(), obs=pd.DataFrame(index=frame.index), var=pd.DataFrame(index=frame.columns)
        )
        # Use the public prediction path with a local, separately pinned weight file.
        with patch("pyaging.predict._pred.load_clock", return_value=model):
            pya.pred.predict_age(adata, clock, verbose=False)
        return adata.obs[clock].to_numpy()

    baseline = predict(beta, use_medians=False)
    np.testing.assert_array_equal(predict(beta, use_medians=True), baseline)
    medians = _load_dnam_feature_medians()
    rng = np.random.default_rng(600)
    masks = {
        f"random_{fraction:.0%}": rng.choice(beta.columns, size=round(fraction * beta.shape[1]), replace=False)
        for fraction in (0.05, 0.10, 0.25)
    }
    if panel is not None:
        masks["observed_panel"] = beta.columns.difference(panel)

    results = []
    for mask, masked in masks.items():
        if not 0 < len(masked) < beta.shape[1]:
            continue
        incomplete = beta.drop(columns=masked)
        for use_medians in (False, True):
            predicted = predict(incomplete, use_medians=use_medians)
            delta = np.abs(predicted - baseline)
            results.append(
                {
                    "clock": clock,
                    "samples": len(beta),
                    "mask": mask,
                    "masked_probes": len(masked),
                    "covered_probes": sum(feature in medians for feature in masked),
                    "fill": "median" if use_medians else "zero",
                    "median_abs_change_years": float(np.median(delta)),
                    "p95_abs_change_years": float(np.quantile(delta, 0.95)),
                    "chronological_mae_years": (
                        float(np.mean(np.abs(predicted - ages.loc[beta.index])))
                        if ages is not None and clock == "hannum"
                        else None
                    ),
                }
            )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("beta_matrix", type=Path)
    parser.add_argument("weights", type=Path, nargs="+")
    parser.add_argument("--blood", action="store_true", help="Read the GSE41169 GEO series matrix")
    parser.add_argument("--panel", type=Path, help="Use CpGs in the GSE224218 processed matrix as another mask")
    args = parser.parse_args()
    models = [
        torch.load(path, weights_only=False, map_location="cpu").to(torch.float64).eval() for path in args.weights
    ]
    features = set().union(*(model.features for model in models))
    beta, ages = read_validation_betas(args.beta_matrix, features, blood=args.blood)
    panel = set(pd.read_csv(args.panel, sep="\t", usecols=["ID_REF"])["ID_REF"]) if args.panel else None
    print(json.dumps([compare(beta, ages, model, panel) for model in models], indent=2))


if __name__ == "__main__":
    main()
