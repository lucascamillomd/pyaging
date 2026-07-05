"""Hermetic checks on the committed Clock Explorer artifacts.

These tests validate the checked-in docs/_static/clocks.json and
clock_glossary.csv WITHOUT downloading from S3 and WITHOUT writing any file.
generate() itself is exercised at build time by the conf.py `builder-inited`
hook, so it does not need to run here.
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.normpath(os.path.join(HERE, "..", "_static"))

REQUIRED = {
    "clock_name", "data_type", "species", "predicts", "unit", "tissue",
    "platform", "population", "model_type", "n_features", "year", "citations",
    "last_author", "journal", "doi", "notes", "approved_by_author", "notebook",
}

# Human-readable header of clock_glossary.csv (index column first, then FIELDS order).
EXPECTED_CSV_HEADER = [
    "Clock name", "Data type", "Species", "Predicts", "Unit", "Tissue",
    "Platform", "Population", "Model type", "N features", "Year", "Citations",
    "Citations date", "Last author", "Journal", "DOI", "Notes", "Preprocess",
    "Postprocess", "Reference values", "Approved by author(s)",
]


def _load_rows():
    with open(os.path.join(STATIC, "clocks.json"), encoding="utf-8") as fh:
        return json.load(fh)


def test_committed_clocks_json_is_valid():
    rows = _load_rows()
    assert isinstance(rows, list)
    assert len(rows) == 173, f"expected 173 clocks, got {len(rows)}"

    # required keys present on every row
    for row in rows:
        missing = REQUIRED - set(row)
        assert not missing, f"{row.get('clock_name')} missing {missing}"
        # notebook link points into the gallery
        assert row["notebook"].startswith("clock_notebooks/")

    # ordering: approved-by-author clocks first, then alphabetical by name
    keys = [(r["approved_by_author"] != "approved", r["clock_name"].lower()) for r in rows]
    assert keys == sorted(keys), "rows are not sorted approved-first then A-Z"

    # every approval value is one of the normalized strings
    assert {r["approved_by_author"] for r in rows} <= {"approved", "not approved"}

    # strict JSON: no NaN/Infinity leakage (browsers' JSON.parse would reject those)
    json.dumps(rows, allow_nan=False)


def test_committed_glossary_csv_matches_json():
    rows = _load_rows()
    with open(os.path.join(STATIC, "clock_glossary.csv"), encoding="utf-8", newline="") as fh:
        reader = list(csv.reader(fh))
    assert reader, "clock_glossary.csv is empty"
    header = reader[0]
    assert header == EXPECTED_CSV_HEADER, f"unexpected CSV header: {header}"
    # one data row per clock
    data_rows = reader[1:]
    assert len(data_rows) == len(rows), f"CSV has {len(data_rows)} rows, JSON has {len(rows)}"
