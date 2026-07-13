"""Hermetic checks for Clock Explorer generation and committed artifacts.

The generator test uses temporary input and output paths; committed-artifact
checks validate docs/_static without network access.
"""
import csv
import json
import os
from unittest.mock import Mock

import make_clock_data
import torch

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
    "Postprocess", "Reference values", "Verified",
]


def _load_rows():
    with open(os.path.join(STATIC, "clocks.json"), encoding="utf-8") as fh:
        return json.load(fh)


def test_generate_downloads_metadata_from_hugging_face(tmp_path, monkeypatch):
    source_metadata = tmp_path / "source_metadata.pt"
    torch.save(
        {
            "verified_clock": {"approved_by_author": "✅"},
            "pending_clock": {"approved_by_author": "⌛"},
        },
        source_metadata,
    )
    static = tmp_path / "static"
    download_hf_file = Mock(return_value=str(source_metadata))
    monkeypatch.setattr(make_clock_data, "STATIC", str(static))
    monkeypatch.setattr(make_clock_data, "download_hf_file", download_hf_file)

    count = make_clock_data.generate()

    assert count == 2
    download_hf_file.assert_called_once_with("all_clock_metadata.pt", str(static))
    assert (static / "clocks.json").exists()
    assert (static / "clock_glossary.csv").exists()
    rows = json.loads((static / "clocks.json").read_text(encoding="utf-8"))
    assert {row["approved_by_author"] for row in rows} == {"By authors", "Not yet"}


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

    # ordering: author-verified clocks first, then alphabetical by name
    keys = [(r["approved_by_author"] != "By authors", r["clock_name"].lower()) for r in rows]
    assert keys == sorted(keys), "rows are not sorted verified-first then A-Z"

    # every approval value is one of the normalized strings
    assert {r["approved_by_author"] for r in rows} <= {"By authors", "Not yet"}

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
