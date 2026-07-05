"""Build-time generator for the Clock Explorer data.

Downloads the public aggregate clock metadata from S3 and writes:
  - docs/_static/clocks.json  (array consumed by the Explorer front-end)
  - docs/_static/clock_glossary.csv  (download + no-JS fallback)
"""
import json
import os
from urllib.request import urlretrieve

import pandas as pd
import torch

URL = "https://pyaging.s3.amazonaws.com/clocks/metadata0.1.0/all_clock_metadata.pt"
STATIC = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "_static"))

FIELDS = [
    "data_type", "species", "predicts", "unit", "tissue", "platform",
    "population", "model_type", "n_features", "year", "citations",
    "citations_date", "last_author", "journal", "doi", "notes",
    "preprocess", "postprocess", "reference_values", "approved_by_author",
]


def _json_safe(o):
    if hasattr(o, "item"):          # numpy scalar
        return o.item()
    if hasattr(o, "tolist"):        # numpy array / tensor
        return o.tolist()
    return str(o)


def _shorten(v):
    # reference_values can be a long array; keep the file lean.
    if isinstance(v, (list, tuple)) and len(v) > 8:
        return "{} values".format(len(v))
    return v


def generate():
    os.makedirs(STATIC, exist_ok=True)
    pt_path = os.path.join(STATIC, "all_clock_metadata.pt")
    urlretrieve(URL, pt_path)
    meta = torch.load(pt_path, weights_only=False)

    rows = []
    for name, m in meta.items():
        row = {"clock_name": name}
        for f in FIELDS:
            row[f] = _shorten(m.get(f))
        row["notebook"] = "clock_notebooks/{}.html".format(name)
        rows.append(row)
    rows.sort(key=lambda r: r["clock_name"].lower())

    with open(os.path.join(STATIC, "clocks.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, default=_json_safe, separators=(",", ":"))

    # CSV mirror (human-friendly column order) for download + no-JS fallback
    df = pd.DataFrame(rows).drop(columns=["notebook"]).set_index("clock_name")
    df.to_csv(os.path.join(STATIC, "clock_glossary.csv"))
    return len(rows)


if __name__ == "__main__":
    print("generated {} clocks".format(generate()))
