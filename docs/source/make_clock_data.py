"""Build-time generator for the Clock Explorer data.

Downloads the public aggregate clock metadata from S3 and writes:
  - docs/_static/clocks.json  (array consumed by the Explorer front-end)
  - docs/_static/clock_glossary.csv  (download + no-JS fallback)
"""
import json
import math
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

# Human-readable CSV headers (no-JS fallback table + downloaded clock_glossary.csv).
LABELS = {
    "clock_name": "Clock name",
    "data_type": "Data type",
    "species": "Species",
    "predicts": "Predicts",
    "unit": "Unit",
    "tissue": "Tissue",
    "platform": "Platform",
    "population": "Population",
    "model_type": "Model type",
    "n_features": "N features",
    "year": "Year",
    "citations": "Citations",
    "citations_date": "Citations date",
    "last_author": "Last author",
    "journal": "Journal",
    "doi": "DOI",
    "notes": "Notes",
    "preprocess": "Preprocess",
    "postprocess": "Postprocess",
    "reference_values": "Reference values",
    "approved_by_author": "Approved by author(s)",
}


def _finite(v):
    # NaN/±Inf are not valid JSON tokens; browsers' JSON.parse rejects them.
    # Map any non-finite float to None (-> null) so clocks.json always parses.
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _approval(v):
    # Upstream marks author approval with an emoji (✅ approved, ⌛ pending);
    # normalize to a searchable/filterable string for the Explorer + CSV.
    return "approved" if str(v).strip() == "✅" else "not approved"


def _json_safe(o):
    # Prefer tolist(): it works for numpy scalars, numpy arrays, and torch
    # tensors, whereas .item() raises on multi-element arrays.
    if hasattr(o, "tolist"):
        return _finite(o.tolist())
    if hasattr(o, "item"):
        return _finite(o.item())
    return str(o)


def _shorten(v):
    # reference_values can be a long array (list/tuple/ndarray/tensor); collapse
    # it to keep the file lean and avoid embedding large numeric blobs.
    if not isinstance(v, str):
        try:
            n = len(v)
        except TypeError:
            n = None
        if n is not None and n > 8:
            return "{} values".format(n)
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
            row[f] = _finite(_shorten(m.get(f)))
        row["approved_by_author"] = _approval(m.get("approved_by_author"))
        row["notebook"] = "clock_notebooks/{}.html".format(name)
        rows.append(row)
    # Approved-by-author clocks first, then alphabetical by name.
    rows.sort(key=lambda r: (r["approved_by_author"] != "approved", r["clock_name"].lower()))

    with open(os.path.join(STATIC, "clocks.json"), "w", encoding="utf-8") as fh:
        json.dump(
            rows, fh, ensure_ascii=False, default=_json_safe,
            separators=(",", ":"), allow_nan=False,
        )

    # CSV mirror (human-friendly column order + headers) for download + no-JS fallback
    df = pd.DataFrame(rows).drop(columns=["notebook"]).set_index("clock_name")
    df.index.name = LABELS["clock_name"]
    df = df.rename(columns=LABELS)
    df.to_csv(os.path.join(STATIC, "clock_glossary.csv"))
    return len(rows)


if __name__ == "__main__":
    print("generated {} clocks".format(generate()))
