import json
import sys
from pathlib import Path

import pytest
import torch

CLOCKS_DIR = Path(__file__).resolve().parents[1] / "clocks"
sys.path.insert(0, str(CLOCKS_DIR))

import patch_clocks_v050  # noqa: E402
from patch_clocks_v050 import (  # noqa: E402
    NotebookShapeError,
    main,
    patch_notebook,
    patch_weight_file,
    rename_features,
)

from pyaging.utils import resolve_feature_ranges  # noqa: E402

BASIC_TEST_SOURCE = [
    "torch.manual_seed(42)\n",
    "input = torch.randn(10, len(model.features), dtype=float)\n",
    "model.eval()\n",
    "model.to(float)\n",
    "pred = model(input)\n",
    "pred",
]
IMPORT_SOURCE = ["import os\n", "import json\n", "import torch\n", "import pandas as pd\n", "import pyaging as pya"]


def _notebook(path, cells):
    path.write_text(json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}, indent=1))
    return path


def _markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def _code(source):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source}


def _notebook_with_basic_test(path, heading="## Basic test"):
    return _notebook(path, [_code(IMPORT_SOURCE), _markdown(heading), _code(BASIC_TEST_SOURCE)])


def _cells(path):
    return json.loads(path.read_text())["cells"]


def _save_fake_clock(path, features, units=None):
    model = torch.nn.Module()
    model.features = features
    model.metadata = {"clock_name": path.stem, "data_type": "DNA methylation"}
    if units is not None:
        model.feature_units = units
    torch.save(model, path)
    return path


def _save_current_clock(path, features):
    """Save a clock that already carries both the harmonized names and the registry units."""
    units = [record["unit"] for record in resolve_feature_ranges(features, "DNA methylation")]
    return _save_fake_clock(path, features, units=units)


def test_rename_features_maps_legacy_names():
    assert rename_features(["cg001", "Female", "Age", "GrimAge"]) == [
        "cg001",
        "female",
        "age",
        "GrimAge",
    ]


def test_rename_features_leaves_already_correct_names_alone():
    assert rename_features(["female", "age"]) == ["female", "age"]


def test_rename_features_rejects_a_collision():
    with pytest.raises(ValueError, match="already present"):
        rename_features(["Female", "female"])


def test_patch_weight_file_renames_and_reports_change(tmp_path):
    model = torch.nn.Module()
    model.features = ["cg001", "Female", "Age"]
    model.metadata = {"clock_name": "fake", "data_type": "DNA methylation"}
    path = tmp_path / "fake.pt"
    torch.save(model, path)

    assert patch_weight_file(path) is True
    assert torch.load(path, weights_only=False).features == ["cg001", "female", "age"]


def test_patch_weight_file_is_idempotent(tmp_path):
    model = torch.nn.Module()
    model.features = ["female", "age"]
    model.metadata = {"clock_name": "fake", "data_type": "clinical biomarkers"}
    path = tmp_path / "fake.pt"
    torch.save(model, path)

    patch_weight_file(path)
    assert patch_weight_file(path) is False


def test_rename_features_rejects_a_repeated_legacy_name():
    with pytest.raises(ValueError, match="already present"):
        rename_features(["Female", "Female"])


def test_patch_weight_file_rejects_a_model_without_features(tmp_path):
    model = torch.nn.Module()
    model.metadata = {"clock_name": "fake", "data_type": "DNA methylation"}
    path = tmp_path / "fake.pt"
    torch.save(model, path)

    with pytest.raises(ValueError, match="no 'features' list"):
        patch_weight_file(path)


def test_patch_weight_file_leaves_no_temporary_files_behind(tmp_path):
    path = _save_fake_clock(tmp_path / "fake.pt", ["cg001", "Female", "Age"])

    assert patch_weight_file(path) is True
    assert [p.name for p in tmp_path.iterdir()] == ["fake.pt"]


def test_patch_weight_file_keeps_the_original_when_the_save_fails(tmp_path, monkeypatch):
    path = _save_fake_clock(tmp_path / "fake.pt", ["cg001", "Female", "Age"])
    original_bytes = path.read_bytes()

    def failing_save(obj, target, *args, **kwargs):
        Path(target).write_bytes(b"truncated")  # a real partial write, not a clean no-op
        raise OSError("disk full")

    monkeypatch.setattr(patch_clocks_v050.torch, "save", failing_save)
    with pytest.raises(OSError, match="disk full"):
        patch_weight_file(path)

    assert path.read_bytes() == original_bytes
    assert [p.name for p in tmp_path.iterdir()] == ["fake.pt"]


def test_main_reports_scanned_and_patched_counts(tmp_path, capsys, monkeypatch):
    _save_fake_clock(tmp_path / "stale.pt", ["cg001", "Female", "Age"])
    _save_current_clock(tmp_path / "current.pt", ["cg001", "female", "age"])
    monkeypatch.setattr(sys, "argv", ["patch_clocks_v050.py", str(tmp_path)])

    assert main() == 0
    assert capsys.readouterr().out.strip() == "scanned 2 clocks, patched 1: stale.pt"


def test_main_reports_none_when_every_clock_is_already_current(tmp_path, capsys, monkeypatch):
    _save_current_clock(tmp_path / "current.pt", ["cg001", "female", "age"])
    monkeypatch.setattr(sys, "argv", ["patch_clocks_v050.py", str(tmp_path)])

    assert main() == 0
    assert capsys.readouterr().out.strip() == "scanned 1 clocks, patched 0: none"


def test_main_fails_on_a_directory_with_no_clocks(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["patch_clocks_v050.py", str(tmp_path)])

    assert main() == 1
    assert "no .pt files found" in capsys.readouterr().err


def test_patch_weight_file_sets_feature_units(tmp_path):
    model = torch.nn.Module()
    model.features = ["albumin", "age"]
    model.metadata = {"clock_name": "fake", "data_type": "clinical biomarkers"}
    path = tmp_path / "fake.pt"
    torch.save(model, path)

    assert patch_weight_file(path) is True
    assert torch.load(path, weights_only=False).feature_units == ["g/L", "years"]


def test_patch_weight_file_is_idempotent_once_units_are_written(tmp_path):
    path = _save_fake_clock(tmp_path / "fake.pt", ["cg001", "Female", "Age"])

    assert patch_weight_file(path) is True
    assert patch_weight_file(path) is False


def test_patch_notebook_replaces_randn_with_in_range_values(tmp_path):
    path = _notebook_with_basic_test(tmp_path / "fake.ipynb")

    assert patch_notebook(path) is True
    joined = "".join(_cells(path)[-1]["source"])
    assert "torch.randn" not in joined
    assert "resolve_feature_ranges" in joined


def test_patch_notebook_adds_normal_feature_ranges_section(tmp_path):
    path = _notebook_with_basic_test(tmp_path / "fake.ipynb")
    patch_notebook(path)

    cells = _cells(path)
    headings = ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown"]
    assert any("Normal feature ranges" in heading for heading in headings)
    assert any("model.feature_units" in "".join(cell["source"]) for cell in cells)


def test_patch_notebook_adds_the_math_import_the_basic_test_needs(tmp_path):
    path = _notebook_with_basic_test(tmp_path / "fake.ipynb")
    patch_notebook(path)

    cells = _cells(path)
    assert "import math\n" in cells[0]["source"]
    assert "math." in "".join(cells[-1]["source"])


def test_patch_notebook_does_not_duplicate_an_existing_math_import(tmp_path):
    path = _notebook_with_basic_test(tmp_path / "fake.ipynb")
    cells = _cells(path)
    cells[0]["source"] = ["import math\n", *IMPORT_SOURCE]
    _notebook(path, cells)

    patch_notebook(path)
    assert "".join(_cells(path)[0]["source"]).count("import math") == 1


def test_patch_notebook_accepts_the_capitalised_basic_test_heading(tmp_path):
    path = _notebook_with_basic_test(tmp_path / "fake.ipynb", heading="## Basic Test")

    assert patch_notebook(path) is True
    assert "resolve_feature_ranges" in "".join(_cells(path)[-1]["source"])


def test_patch_notebook_gives_every_cell_it_writes_an_id(tmp_path):
    path = _notebook_with_basic_test(tmp_path / "fake.ipynb")
    cells = _cells(path)
    cells[2]["id"] = "keep-me"
    _notebook(path, cells)

    patch_notebook(path)
    ranges_markdown, ranges_code, _basic_test_heading, basic_test = _cells(path)[1:]
    assert ranges_markdown.get("id") and ranges_code.get("id")
    assert basic_test["id"] == "keep-me"


def test_patch_notebook_is_idempotent(tmp_path):
    path = _notebook_with_basic_test(tmp_path / "fake.ipynb")

    patch_notebook(path)
    assert patch_notebook(path) is False


def test_patch_notebook_refuses_a_notebook_without_a_basic_test(tmp_path):
    path = _notebook(tmp_path / "fake.ipynb", [_code(IMPORT_SOURCE), _markdown("## Save torch model")])

    with pytest.raises(NotebookShapeError, match="no '## Basic test' heading"):
        patch_notebook(path)


def test_patch_notebook_refuses_a_notebook_with_two_basic_test_headings(tmp_path):
    path = _notebook(
        tmp_path / "fake.ipynb",
        [
            _code(IMPORT_SOURCE),
            _markdown("## Basic test"),
            _code(BASIC_TEST_SOURCE),
            _markdown("## Basic test"),
            _code(BASIC_TEST_SOURCE),
        ],
    )

    with pytest.raises(NotebookShapeError, match="2 '## Basic test' headings"):
        patch_notebook(path)


def test_patch_notebook_refuses_when_the_cell_after_the_heading_is_unexpected(tmp_path):
    path = _notebook(
        tmp_path / "fake.ipynb",
        [_code(IMPORT_SOURCE), _markdown("## Basic test"), _code(["pred = model(my_own_input)"])],
    )

    with pytest.raises(NotebookShapeError, match="does not call torch.randn"):
        patch_notebook(path)


def test_patch_notebook_steps_over_prose_between_the_heading_and_the_code(tmp_path):
    path = _notebook(
        tmp_path / "fake.ipynb",
        [_code(IMPORT_SOURCE), _markdown("## Basic test"), _markdown("Some prose."), _code(BASIC_TEST_SOURCE)],
    )

    assert patch_notebook(path) is True
    cells = _cells(path)
    assert [cell["cell_type"] for cell in cells[1:]] == ["markdown", "code", "markdown", "markdown", "code"]
    assert "resolve_feature_ranges" in "".join(cells[-1]["source"])


def test_patch_notebook_reports_an_already_patched_notebook_with_prose_as_unchanged(tmp_path):
    """The four v0.5.0 clocks explain their smoke test in prose; they must not be double-patched."""
    path = _notebook(
        tmp_path / "fake.ipynb",
        [
            _code(IMPORT_SOURCE),
            _markdown("#### Normal feature ranges"),
            _markdown("## Basic test"),
            _markdown("Some prose."),
            _code(["records = pya.utils.resolve_feature_ranges(model.features, None)"]),
        ],
    )
    before = path.read_text()

    assert patch_notebook(path) is False
    assert path.read_text() == before


def test_patch_notebook_refuses_when_the_next_section_starts_before_any_code(tmp_path):
    path = _notebook(
        tmp_path / "fake.ipynb",
        [
            _code(IMPORT_SOURCE),
            _markdown("## Basic test"),
            _markdown("## Save torch model"),
            _code(["torch.save(model)"]),
        ],
    )

    with pytest.raises(NotebookShapeError, match="no code cell under"):
        patch_notebook(path)


def test_patch_notebook_refuses_a_half_patched_notebook(tmp_path):
    """A ranges section with an unpatched basic test is ambiguous, so never guess."""
    path = _notebook(
        tmp_path / "fake.ipynb",
        [
            _code(IMPORT_SOURCE),
            _markdown("#### Normal feature ranges"),
            _markdown("## Basic test"),
            _code(BASIC_TEST_SOURCE),
        ],
    )

    with pytest.raises(NotebookShapeError, match="already has a 'Normal feature ranges' section"):
        patch_notebook(path)


def test_patch_notebook_refuses_a_notebook_without_an_import_cell(tmp_path):
    path = _notebook(tmp_path / "fake.ipynb", [_markdown("## Basic test"), _code(BASIC_TEST_SOURCE)])

    with pytest.raises(NotebookShapeError, match="no import cell"):
        patch_notebook(path)


def test_patch_notebook_leaves_the_file_untouched_when_it_refuses(tmp_path):
    path = _notebook(tmp_path / "fake.ipynb", [_code(IMPORT_SOURCE), _markdown("## Save torch model")])
    before = path.read_text()

    with pytest.raises(NotebookShapeError):
        patch_notebook(path)
    assert path.read_text() == before


def test_main_patches_notebooks_when_given_a_notebooks_dir(tmp_path, capsys, monkeypatch):
    weights = tmp_path / "weights"
    notebooks = tmp_path / "notebooks"
    weights.mkdir()
    notebooks.mkdir()
    _save_fake_clock(weights / "fake.pt", ["cg001", "Female", "Age"])
    _notebook_with_basic_test(notebooks / "fake.ipynb")
    monkeypatch.setattr(sys, "argv", ["patch_clocks_v050.py", str(weights), "--notebooks-dir", str(notebooks)])

    assert main() == 0
    assert "scanned 1 notebooks, patched 1: fake.ipynb" in capsys.readouterr().out


def test_main_reports_skipped_notebooks_and_fails(tmp_path, capsys, monkeypatch):
    weights = tmp_path / "weights"
    notebooks = tmp_path / "notebooks"
    weights.mkdir()
    notebooks.mkdir()
    _save_current_clock(weights / "fake.pt", ["cg001", "female", "age"])
    _notebook_with_basic_test(notebooks / "good.ipynb")
    _notebook(notebooks / "odd.ipynb", [_code(IMPORT_SOURCE), _markdown("## Save torch model")])
    monkeypatch.setattr(sys, "argv", ["patch_clocks_v050.py", str(weights), "--notebooks-dir", str(notebooks)])

    assert main() == 1
    output = capsys.readouterr().out
    assert "skipped 1" in output
    assert "odd.ipynb" in output
    assert "resolve_feature_ranges" in "".join(_cells(notebooks / "good.ipynb")[-1]["source"])
