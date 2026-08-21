"""Build the blood chemistry example dataset from NHANES IV.

The source is the public ``NHANES4`` table shipped with the R package BioAge
(Kwon & Belsky), which pools the 1999-2018 continuous NHANES cycles. Rows are
kept only where every variable a pyaging clinical clock needs is measured, and
only where every value falls inside the package's own plausibility range, so
the shipped example never trips a range warning of its own.

Values are converted to the units the pyaging feature registry declares. CRP is
carried raw in mg/dL - each clock applies its own transform internally - so no
pre-logged column ships.

Usage::

    uv run python clocks/build_blood_chemistry_example.py

Writes ``clocks/blood_chemistry_example.csv`` (reviewable) and
``clocks/blood_chemistry_example.pkl`` (the file uploaded to Hugging Face).
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

from pyaging.utils._feature_ranges import load_feature_range_registry

N_ROWS = 30
HERE = Path(__file__).resolve().parent

# BioAge NHANES4 column -> (pyaging feature name, multiplier to the registry unit)
CONVERSIONS = {
    "albumin_gL": ("albumin", 1.0),
    "creat_umol": ("creatinine", 1.0),
    "glucose_mmol": ("glucose", 1.0),
    "crp": ("c_reactive_protein", 1.0),  # raw mg/dL, not BioAge's lncrp
    "lymph": ("lymphocyte_percent", 1.0),
    "mcv": ("mean_cell_volume", 1.0),
    "rdw": ("red_cell_distribution_width", 1.0),
    "alp": ("alkaline_phosphatase", 1.0),
    "wbc": ("white_blood_cell_count", 1.0),
    "totchol": ("total_cholesterol", 0.02586),  # mg/dL -> mmol/L
    "bun": ("blood_urea_nitrogen", 0.357),  # mg/dL -> mmol/L
    "hba1c": ("hemoglobin_a1c", 1.0),
    "sbp": ("systolic_blood_pressure", 1.0),
    "fev": ("forced_expiratory_volume", 0.001),  # mL -> L
    "age": ("age", 1.0),
}

R_SCRIPT = """
suppressPackageStartupMessages({{library(BioAge); library(dplyr); library(readr)}})
NHANES4 %>%
  select(sampleID, gender, all_of(c({source_cols}))) %>%
  na.omit() %>%
  write_csv("{out}")
"""


def extract_nhanes(destination: Path) -> pd.DataFrame:
    """Pull the complete-case NHANES IV rows out of R's BioAge package."""
    source_cols = ", ".join(f'"{name}"' for name in CONVERSIONS)
    script = R_SCRIPT.format(source_cols=source_cols, out=destination)
    subprocess.run(["Rscript", "-e", script], check=True)
    return pd.read_csv(destination)


def to_pyaging_units(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename and rescale BioAge columns into pyaging feature names and units."""
    frame = pd.DataFrame(
        {feature: raw[source].to_numpy() * factor for source, (feature, factor) in CONVERSIONS.items()},
        index=[f"NHANES_{sample_id}" for sample_id in raw["sampleID"]],
    )
    frame["female"] = (raw["gender"] == 2).astype(float).to_numpy()
    return frame


def in_registry_range(frame: pd.DataFrame) -> pd.Series:
    """Mask of rows whose every value lies inside the package's plausibility range."""
    features = load_feature_range_registry()["features"]
    inside = pd.Series(True, index=frame.index)
    for column in frame.columns:
        entry = features[column]
        inside &= frame[column].between(entry["low"], entry["high"])
    return inside


def assert_defensible(frame: pd.DataFrame) -> None:
    """Fail loudly on a constant column or one that leaves its plausibility range.

    The dataset this replaces shipped a ``log_crp`` column with a standard
    deviation of exactly zero, so every published PhenoAge was computed against
    a placeholder.
    """
    features = load_feature_range_registry()["features"]
    problems = []
    for column in frame.columns:
        values = frame[column]
        if values.isna().any():
            problems.append(f"{column}: {int(values.isna().sum())} missing values")
        if values.std() == 0:
            problems.append(f"{column}: standard deviation is exactly 0 (constant {values.iloc[0]})")
        entry = features[column]
        outside = ~values.between(entry["low"], entry["high"])
        if outside.any():
            problems.append(
                f"{column}: {int(outside.sum())} values outside {entry['low']}-{entry['high']} {entry['unit']}"
            )
    if problems:
        raise SystemExit("Refusing to write the example dataset:\n  " + "\n  ".join(problems))


def main() -> None:
    raw = extract_nhanes(HERE / "nhanes4_complete_cases.csv")
    frame = to_pyaging_units(raw)
    frame = frame[in_registry_range(frame)].head(N_ROWS)

    if len(frame) < N_ROWS:
        raise SystemExit(f"Only {len(frame)} usable NHANES rows; expected {N_ROWS}")
    assert_defensible(frame)
    if frame["female"].nunique() < 2:
        raise SystemExit("The selected rows are all one sex; the sex-specific clocks need both")

    frame.to_csv(HERE / "blood_chemistry_example.csv")
    # pandas 3 defaults to a pyarrow-backed string dtype; pyaging does not depend on
    # pyarrow, so an arrow-backed index would make the pickle unreadable on a clean install.
    frame.index = frame.index.astype(object)
    frame.columns = frame.columns.astype(object)
    frame.to_pickle(HERE / "blood_chemistry_example.pkl")
    (HERE / "nhanes4_complete_cases.csv").unlink()

    summary = frame.describe().T[["mean", "std", "min", "max"]]
    print(f"{len(frame)} subjects x {frame.shape[1]} features", file=sys.stderr)
    print(summary.to_string(), file=sys.stderr)


if __name__ == "__main__":
    main()
