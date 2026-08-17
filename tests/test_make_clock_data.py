"""Hermetic checks for Clock Explorer generation and committed artifacts.

The generator test uses temporary input and output paths; committed-artifact
checks validate docs/_static without network access.
"""

import csv
import importlib.util
import json
import os
import sys
from unittest.mock import Mock

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(REPO_ROOT, "docs", "_static")

_spec = importlib.util.spec_from_file_location(
    "make_clock_data", os.path.join(REPO_ROOT, "docs", "source", "make_clock_data.py")
)
make_clock_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_clock_data)

REQUIRED = {
    "clock_name",
    "data_type",
    "species",
    "predicts",
    "training_target",
    "unit",
    "tissue",
    "platform",
    "population",
    "model_type",
    "n_features",
    "year",
    "citations",
    "last_author",
    "journal",
    "doi",
    "notes",
    "approved_by_author",
    "notebook",
}

# Human-readable header of clock_glossary.csv (index column first, then FIELDS order).
EXPECTED_CSV_HEADER = [
    "Clock name",
    "Data type",
    "Species",
    "Predicts",
    "Training target",
    "Unit",
    "Tissue",
    "Platform",
    "Population",
    "Model type",
    "N features",
    "Year",
    "Citations",
    "Citations date",
    "Last author",
    "Journal",
    "DOI",
    "Notes",
    "Preprocess",
    "Postprocess",
    "Reference values",
    "Verified",
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


def test_training_target_follows_predicts_in_generated_fields():
    predicts_index = make_clock_data.FIELDS.index("predicts")

    assert make_clock_data.FIELDS[predicts_index + 1] == "training_target"
    assert make_clock_data.LABELS["training_target"] == "Training target"


def test_generate_uses_explicit_local_metadata_without_downloading(tmp_path, monkeypatch):
    source_metadata = tmp_path / "source_metadata.pt"
    torch.save({"local_clock": {"approved_by_author": "✅"}}, source_metadata)
    static = tmp_path / "static"
    download_hf_file = Mock(side_effect=AssertionError("must not download"))
    monkeypatch.setattr(make_clock_data, "STATIC", str(static))
    monkeypatch.setattr(make_clock_data, "download_hf_file", download_hf_file)

    count = make_clock_data.generate(metadata_path=source_metadata)

    assert count == 1
    download_hf_file.assert_not_called()
    rows = json.loads((static / "clocks.json").read_text(encoding="utf-8"))
    assert [row["clock_name"] for row in rows] == ["local_clock"]


def test_generate_preserves_array_fields_in_json_and_joins_them_in_csv(tmp_path, monkeypatch):
    source_metadata = tmp_path / "source_metadata.pt"
    array_fields = {
        "predicts": ["chronological age", "mortality"],
        "training_target": ["chronological age", "time-to-death"],
        "unit": ["years", "unitless"],
        "tissue": ["whole blood", "cord blood"],
        "platform": ["Illumina 450K", "Illumina EPIC"],
    }
    torch.save(
        {
            "array_clock": {
                "approved_by_author": "✅",
                **array_fields,
            }
        },
        source_metadata,
    )
    static = tmp_path / "static"
    monkeypatch.setattr(make_clock_data, "STATIC", str(static))

    make_clock_data.generate(metadata_path=source_metadata)

    rows = json.loads((static / "clocks.json").read_text(encoding="utf-8"))
    assert {field: rows[0][field] for field in array_fields} == array_fields
    with (static / "clock_glossary.csv").open(encoding="utf-8", newline="") as fh:
        csv_row = next(csv.DictReader(fh))
    assert csv_row["Predicts"] == "chronological age | mortality"
    assert csv_row["Training target"] == "chronological age | time-to-death"
    assert csv_row["Unit"] == "years | unitless"
    assert csv_row["Tissue"] == "whole blood | cord blood"
    assert csv_row["Platform"] == "Illumina 450K | Illumina EPIC"


def test_generate_maps_nonfinite_array_field_scalar_to_json_null(tmp_path, monkeypatch):
    source_metadata = tmp_path / "source_metadata.pt"
    torch.save(
        {"nonfinite_clock": {"approved_by_author": "✅", "unit": float("nan")}},
        source_metadata,
    )
    static = tmp_path / "static"
    monkeypatch.setattr(make_clock_data, "STATIC", str(static))

    make_clock_data.generate(metadata_path=source_metadata)

    rows = json.loads((static / "clocks.json").read_text(encoding="utf-8"))
    assert rows[0]["unit"] is None


def test_main_passes_metadata_path_from_cli(monkeypatch, capsys):
    generate = Mock(return_value=173)
    monkeypatch.setattr(make_clock_data, "generate", generate)
    monkeypatch.setattr(sys, "argv", ["make_clock_data.py", "--metadata-path", "/tmp/metadata.pt"])

    make_clock_data.main()

    generate.assert_called_once_with(metadata_path="/tmp/metadata.pt")
    assert capsys.readouterr().out == "generated 173 clocks\n"


def test_committed_clocks_json_is_valid():
    rows = _load_rows()
    assert isinstance(rows, list)
    assert len(rows) == 173, f"expected 173 clocks, got {len(rows)}"
    array_fields = ("tissue", "platform", "predicts", "training_target", "unit")

    # required keys present on every row
    for row in rows:
        missing = REQUIRED - set(row)
        assert not missing, f"{row.get('clock_name')} missing {missing}"
        for field in array_fields:
            assert isinstance(row[field], list), f"{row.get('clock_name')}.{field} must be a controlled-term array"
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
