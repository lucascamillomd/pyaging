import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.normpath(os.path.join(HERE, "..", "_static"))
REQUIRED = {
    "clock_name", "data_type", "species", "predicts", "unit", "tissue",
    "platform", "population", "model_type", "n_features", "year", "citations",
    "last_author", "journal", "doi", "notes", "notebook",
}


def test_generate_writes_valid_json():
    sys.path.insert(0, HERE)
    import make_clock_data

    n = make_clock_data.generate()
    assert n >= 170, f"expected >=170 clocks, got {n}"

    with open(os.path.join(STATIC, "clocks.json"), encoding="utf-8") as fh:
        rows = json.load(fh)
    assert isinstance(rows, list) and len(rows) == n
    # required keys present on every row
    for row in rows:
        missing = REQUIRED - set(row)
        assert not missing, f"{row.get('clock_name')} missing {missing}"
    # sorted by clock_name (case-insensitive)
    names = [r["clock_name"].lower() for r in rows]
    assert names == sorted(names)
    # notebook link points into the gallery
    assert rows[0]["notebook"].startswith("clock_notebooks/")
    # JSON is serializable with no numpy leakage (all values are JSON scalars/None/list/str)
    json.dumps(rows)


def test_csv_also_written():
    assert os.path.exists(os.path.join(STATIC, "clock_glossary.csv"))
