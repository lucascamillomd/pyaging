"""Run under Windows CPython3.9 with the unchanged author pre.pyd.

Usage: python generate_oracle.py SOURCE_DIR OUTPUT_DIR
SOURCE_DIR contains pre.cp39-win_amd64.pyd, original model.bin, and extracted
Conventional.txt (checksum checked). Only pickle.loads is substituted, providing
an object with the same fitted estimator backed by native LightGBM. The original
compiled predict() performs every input transform and the final LOWESS correction.
"""

import hashlib
import json
import pathlib
import pickle
import sys

import lightgbm
import numpy as np
import pandas as pd
import sklearn
import statsmodels

base, output = map(pathlib.Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
assert (
    hashlib.sha256((base / "pre.cp39-win_amd64.pyd").read_bytes()).hexdigest()
    == "c8ecb6f06d0830f3c88dc18e05f5e838042762b3f9f186e125963e089015e85c"
)
assert (
    hashlib.sha256((base / "Conventional.txt").read_bytes()).hexdigest()
    == "10eb9c389ff5fe8fc9adb9c0c27430fcc43dcf628c43ecf42b70e4f8dbfb31d5"
)


class Estimator:
    def __init__(self):
        self.booster = lightgbm.Booster(model_file=str(base / "Conventional.txt"))
        self.feature_name_ = self.booster.feature_name()
        self.n_classes_ = 1
        self.estimator_ = self
        self.raw = None
        self.scaled = None

    def predict(self, frame):
        self.scaled = frame.to_numpy().copy()
        self.raw = self.booster.predict(frame, num_threads=1)
        return self.raw


estimator = Estimator()
pickle.loads = lambda *args, **kwargs: {"Conventional": estimator}
sys.path.insert(0, str(base))
import pre  # noqa: E402 -- the original extension is loaded after installing the pickle boundary.

predictor = pre.OrganAgePredictor(str(base / "model.bin"))
rng = np.random.default_rng(23981)
columns = estimator.feature_name_[:-1]
protein = pd.DataFrame(rng.normal(0, 1, (24, len(columns))), columns=columns, index=[f"s{i}" for i in range(24)])
ages = pd.Series(rng.uniform(40, 75, 24), index=protein.index, name="Age")
cases = {}
cases["complete"] = protein.copy(), ages.copy()
x = protein.copy()
x["TDI"] = rng.uniform(-4, 5, 24)
cases["tdi"] = x, ages.copy()
x = protein.drop(columns=columns[:7]).copy()
x.iloc[1:5, 0] = np.nan
x[columns[10]] = np.nan
cases["missing"] = x, ages.copy()
x = protein.copy()
x.columns = x.columns.str.lower()
x["ABO"] = 5.0
cases["case_collision"] = x, ages.copy()
x = pd.concat([protein, protein[["ABO"]] * 0 + 5], axis=1)
cases["exact_duplicate"] = x, ages.copy()
x = x.copy()
x.iloc[0, 0] = np.nan
x.iloc[1, -1] = np.nan
x.iloc[2, [0, len(x.columns) - 1]] = np.nan
cases["duplicate_nan"] = x, ages.copy()
cases["singleton"] = protein.iloc[:1].copy(), ages.iloc[:1].copy()
cases["constant_age"] = protein.copy(), ages * 0 + 55.0
x = ages.copy()
x.iloc[3] = np.nan
cases["age_nan"] = protein.copy(), x
cases["all_nan"] = protein * np.nan, ages.copy()
archive = {}
errors = {}
for name, (frame, age) in cases.items():
    metadata = pd.DataFrame({"Age": age, "TDI": 20.0}, index=frame.index)
    try:
        result = predictor.predict(frame.copy(), metadata)
    except Exception as error:
        errors[name] = type(error).__name__ + ": " + str(error)
        continue
    archive[name + "_input"] = frame.to_numpy()
    archive[name + "_columns"] = np.array(frame.columns, dtype=str)
    archive[name + "_ages"] = age.to_numpy()
    archive[name + "_index"] = np.array(frame.index, dtype=str)
    archive[name + "_scaled"] = estimator.scaled
    archive[name + "_raw"] = estimator.raw
    archive[name + "_corrected"] = result.loc[frame.index, "Conventional"].to_numpy()
np.savez_compressed(output / "native_oracle.npz", **archive)
metadata = {
    "python": sys.version,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit-learn": sklearn.__version__,
    "statsmodels": statsmodels.__version__,
    "lightgbm": lightgbm.__version__,
    "cases": [name for name in cases if name not in errors],
    "errors": errors,
    "source": "JackieHanLab/PAOPAC v1.0.0; compiled pre.predict() unchanged",
    "pyd_sha256": hashlib.sha256((base / "pre.cp39-win_amd64.pyd").read_bytes()).hexdigest(),
    "model_text_sha256": hashlib.sha256((base / "Conventional.txt").read_bytes()).hexdigest(),
    "weights_interface": (
        "pickle.loads supplies native Booster-backed estimator with original feature_name_; "
        "final fitted estimator uses all 800 trees"
    ),
    "data": "Synthetic NumPy default_rng(23981), no participant data",
}
(output / "oracle_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))
