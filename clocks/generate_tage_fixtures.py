#!/usr/bin/env python3
"""Generate expected tAge predictions from the published sklearn models.

Run after generate_tage_fixtures.R:

    uv run python clocks/generate_tage_fixtures.py tests/data/tage

The prediction logic mirrors ``inst/python/tage_predict.py`` in the authors'
tAge package: load the joblib pipeline, orient the matrix so the model's genes
are columns, select exactly the model's features (missing genes stay NaN so the
pipeline's imputer fills them with training-set medians), predict, and rescale
only the chronological clock by the species maximum lifespan.
"""

import json
import sys
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

ZENODO = "https://zenodo.org/records/18763485/files/{}?download=1"
MODELS = {
    "tage": "EN_Chronoage_Multispecies_Multitissue_scaleddiff.pkl",
    "tagemortality": "EN_Mortality_Multispecies_Multitissue_scaleddiff.pkl",
}
# tAge's PREDICTIONS_SPECIES_ADJ: the chronological clocks predict a fraction of
# maximum lifespan, so mouse predictions are rescaled to months by this factor.
# Mortality clocks output log10(hazard ratio) and are never rescaled.
MOUSE_MAX_LIFESPAN_MONTHS = 48
LIFESPAN_SCALED = {"tage": True, "tagemortality": False}
UNITS = {
    "tage": "months of mouse chronological age",
    "tagemortality": "log10(hazard ratio)",
}


def load_clock(path: Path):
    """Load a published clock, returning (estimator, feature names)."""
    # joblib.load executes pickle bytecode; acceptable here because the files
    # come from the paper's official Zenodo record over HTTPS (fixed record id),
    # the same trust decision pyaging already makes for its torch.save'd clocks.
    clock = joblib.load(path)
    if isinstance(clock, Pipeline):
        for _, step in clock.steps:
            # Models were fitted with scikit-learn < 1.3, whose SimpleImputer
            # lacks the _fill_dtype attribute newer versions expect.
            if isinstance(step, SimpleImputer) and not hasattr(step, "_fill_dtype"):
                step._fill_dtype = getattr(step, "statistics_", np.array([], dtype=np.float64)).dtype
        return clock, list(clock.feature_names_in_)
    return clock, list(clock.feature_names)


def orient(matrix: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return the matrix as samples x genes, whichever way the R dump was written."""
    genes = set(features)
    if len(genes & set(matrix.index.map(str))) > len(genes & set(matrix.columns.map(str))):
        matrix = matrix.T
    matrix.columns = matrix.columns.map(str)
    return matrix


def main(out: Path) -> None:
    preds: dict[str, object] = {}
    for clock_name, filename in MODELS.items():
        local = out / filename
        if not local.exists():
            urllib.request.urlretrieve(ZENODO.format(filename), local)
        clock, features = load_clock(local)
        assert features, "model must carry feature names"
        for suffix in ("center_all", "center_refgroup"):
            stage = out / f"after_{suffix}.csv"
            if not stage.exists():  # fixtures are committed gzipped
                stage = stage.with_suffix(".csv.gz")
            matrix = orient(pd.read_csv(stage, index_col=0), features)
            missing = [f for f in features if f not in matrix.columns]
            assert not missing, f"matrix is missing {len(missing)} model features"
            aligned = matrix.loc[:, features]
            y = clock.predict(aligned)
            if LIFESPAN_SCALED[clock_name]:
                y = y * MOUSE_MAX_LIFESPAN_MONTHS
            preds[f"{clock_name}_{suffix}"] = [float(v) for v in y]
            preds["sample_ids"] = aligned.index.astype(str).tolist()
        preds[f"units_{clock_name}"] = UNITS[clock_name]
        preds[f"n_features_{clock_name}"] = len(features)

    preds["mouse_max_lifespan_months"] = MOUSE_MAX_LIFESPAN_MONTHS
    preds["reference_group_sample_ids"] = (out / "reference_group_sample_ids.txt").read_text().split()
    (out / "expected_predictions.json").write_text(json.dumps(preds, indent=1) + "\n")
    print(json.dumps({k: v for k, v in preds.items() if not isinstance(v, list)}, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
