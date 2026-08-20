#!/usr/bin/env python3
"""One-off v0.5.0 migration for saved clock weights and their notebooks.

Renames legacy covariate features to their harmonized package-wide names,
writes each clock's ``feature_units`` from the package feature range registry,
and rewrites each notebook's smoke test to feed values drawn from the middle of
every feature's plausible range instead of standard normal noise.

Each file is rewritten individually, so an exception part-way through a
directory leaves the files already processed patched; the patch is idempotent,
so re-run it over the same directory.

It also brings each notebook's ``## Index`` cell back in step with the sections
the notebook actually has, including the ranges section this script adds.

The 178 clock notebooks were written over several years and only mostly share a
shape, so a notebook whose layout is not recognised is reported and left
untouched rather than patched on a guess.
"""

import argparse
import json
import os
import re
import sys
import tempfile
import uuid
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pyaging.utils import resolve_feature_ranges  # noqa: E402

LEGACY_FEATURE_NAMES = {"Female": "female", "Age": "age"}

RANGES_HEADING = "## Normal feature ranges"
RANGES_MARKDOWN = (
    f"{RANGES_HEADING}\n"
    "\n"
    "Units and plausible bounds come from `pyaging`'s feature range registry, keyed by feature "
    "name with a fallback to the default for the clock's `data_type`. `predict_age` warns when "
    "input values fall outside these bounds, which usually means the data is in different units "
    "than the clock expects."
)
RANGES_SOURCE = [
    "# Units and plausibility ranges come from the package registry, keyed by feature name.\n",
    'feature_ranges = pya.utils.resolve_feature_ranges(model.features, model.metadata["data_type"])\n',
    'model.feature_units = [record["unit"] for record in feature_ranges]\n',
    "pd.DataFrame.from_records(feature_ranges).head()",
]
BASIC_TEST_SOURCE = [
    "# Exercise the clock with values in the middle of each feature's expected range.\n",
    'records = pya.utils.resolve_feature_ranges(model.features, model.metadata["data_type"])\n',
    "midpoints = [\n",
    '    (record["low"] + record["high"]) / 2 if math.isfinite(record["high"]) else max(record["low"], 1.0)\n',
    "    for record in records\n",
    "]\n",
    "input = torch.tensor([midpoints] * 10, dtype=torch.float64)\n",
    "model.eval()\n",
    "model.to(torch.float64)\n",
    "pred = model(input)\n",
    "pred",
]


class NotebookShapeError(Exception):
    """Raised when a notebook does not have the layout :func:`patch_notebook` knows how to edit."""


def rename_features(features):
    """Replace legacy covariate names with their harmonized equivalents.

    Parameters
    ----------
    features : list of str
        Feature names as stored on a clock, in model input order.

    Returns
    -------
    list of str
        ``features`` with every legacy name mapped through
        :data:`LEGACY_FEATURE_NAMES`. Order and length are preserved.

    Raises
    ------
    ValueError
        If renaming would make a harmonized name appear more than once,
        either because both spellings are present (``["Female", "female"]``)
        or because a legacy name is repeated (``["Female", "Female"]``).
        Such a list cannot be renamed without losing a distinct feature.
    """
    renamed = [LEGACY_FEATURE_NAMES.get(feature, feature) for feature in features]
    for new in LEGACY_FEATURE_NAMES.values():
        if renamed.count(new) > 1:
            raise ValueError(f"cannot rename to {new!r}: it is already present {renamed.count(new)} times")
    return renamed


def patch_weight_file(path: Path) -> bool:
    """Apply the v0.5.0 migration to one saved clock.

    Features are renamed and ``feature_units`` is written from the registry, so
    a saved clock carries the units its values are expected in.

    The file is replaced atomically: the patched model is written to a
    temporary file in the same directory and then moved over the original, so
    a failed write can never truncate the existing weights.

    Parameters
    ----------
    path : pathlib.Path
        Path to a ``.pt`` clock saved by one of the ``clocks/notebooks``.

    Returns
    -------
    bool
        True if the file was rewritten, False if it already used the
        harmonized names and carried the registry units. Re-running over a
        patched file returns False.

    Raises
    ------
    ValueError
        If the saved object has no usable ``features`` list, or if its
        features cannot be renamed unambiguously.
    """
    model = torch.load(path, weights_only=False)
    features = getattr(model, "features", None)
    if features is None:
        raise ValueError(f"{path}: saved object has no 'features' list to migrate")

    renamed = rename_features(features)
    units = [record["unit"] for record in resolve_feature_ranges(renamed, model.metadata.get("data_type"))]
    if renamed == list(features) and getattr(model, "feature_units", None) == units:
        return False

    model.features = renamed
    model.feature_units = units
    descriptor, temporary_path = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    os.close(descriptor)
    try:
        torch.save(model, temporary_path)
        os.replace(temporary_path, path)
    except BaseException:
        Path(temporary_path).unlink(missing_ok=True)
        raise
    return True


def _markdown_cell(text):
    # nbformat 4.5 requires a cell id, and source is stored one line per entry.
    return {"cell_type": "markdown", "id": str(uuid.uuid4()), "metadata": {}, "source": text.splitlines(keepends=True)}


def _code_cell(source):
    return {
        "cell_type": "code",
        "id": str(uuid.uuid4()),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": list(source),
    }


def _source(cell):
    return "".join(cell["source"])


def _find_import_cell(cells) -> int:
    """Return the index of the notebook's import cell."""
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "code" and _source(cell).lstrip().startswith(("import ", "from ")):
            return index
    raise NotebookShapeError("no import cell to add 'import math' to")


def _add_math_import(cell) -> None:
    """Ensure ``import math`` is in an import cell, since the patched smoke test uses it."""
    lines = cell["source"]
    if any(line.strip() == "import math" for line in lines):
        return
    anchor = next((index for index, line in enumerate(lines) if line.startswith("import torch")), 0)
    lines.insert(anchor, "import math\n")


def _find_basic_test_heading(cells) -> int:
    """Return the index of the sole ``## Basic test`` markdown heading.

    The heading is matched case-insensitively because 27 notebooks spell it
    ``## Basic Test``.
    """
    matches = [
        index
        for index, cell in enumerate(cells)
        if cell["cell_type"] == "markdown" and "## basic test" in _source(cell).lower()
    ]
    if not matches:
        raise NotebookShapeError("no '## Basic test' heading")
    if len(matches) > 1:
        raise NotebookShapeError(f"{len(matches)} '## Basic test' headings, expected 1")
    return matches[0]


def _find_basic_test_cell(cells, heading: int) -> int:
    """Return the index of the code cell holding the smoke test.

    Markdown prose between the heading and the code belongs to the section and
    is stepped over, but a further ``## `` heading ends the section.
    """
    for index in range(heading + 1, len(cells)):
        cell = cells[index]
        if cell["cell_type"] == "code":
            return index
        if _source(cell).lstrip().startswith("## "):
            break
    raise NotebookShapeError("no code cell under the '## Basic test' heading")


def patch_notebook(path: Path) -> bool:
    """Give a clock notebook a feature-ranges section and an in-range smoke test.

    Only one notebook layout is accepted: a single ``## Basic test`` markdown
    heading whose section holds a code cell calling ``torch.randn``. Any other
    shape raises rather than being patched on a guess, and the file on disk is
    left untouched when it raises.

    Parameters
    ----------
    path : pathlib.Path
        Path to a notebook in ``clocks/notebooks``.

    Returns
    -------
    bool
        True if the notebook was rewritten, False if it was already patched.

    Raises
    ------
    NotebookShapeError
        If the notebook does not have the expected layout, including the
        half-patched case of a ranges section beside an unpatched smoke test.
    """
    notebook = json.loads(path.read_text())
    cells = notebook["cells"]

    heading = _find_basic_test_heading(cells)
    test_cell = _find_basic_test_cell(cells, heading)

    basic_test = _source(cells[test_cell])
    if "resolve_feature_ranges" in basic_test:
        return False
    if any("Normal feature ranges" in _source(cell) for cell in cells):
        raise NotebookShapeError("already has a 'Normal feature ranges' section but an unpatched basic test")
    if "torch.randn" not in basic_test:
        raise NotebookShapeError("the '## Basic test' code cell does not call torch.randn")

    _add_math_import(cells[_find_import_cell(cells)])
    replacement = _code_cell(BASIC_TEST_SOURCE)
    replacement["id"] = cells[test_cell].get("id", replacement["id"])  # keep the cell's identity stable
    cells[test_cell] = replacement
    cells[heading:heading] = [_markdown_cell(RANGES_MARKDOWN), _code_cell(RANGES_SOURCE)]
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    return True


INDEX_HEADING = "## Index"
_INDEX_ENTRY = re.compile(r"^\d+\. \[(?P<title>.+?)\]\((?P<anchor>#.+?)\)$")


def _find_index_cell(cells) -> int:
    """Return the index of the sole ``## Index`` markdown cell."""
    matches = [
        index
        for index, cell in enumerate(cells)
        if cell["cell_type"] == "markdown" and _source(cell).lstrip().startswith(INDEX_HEADING)
    ]
    if len(matches) != 1:
        raise NotebookShapeError(f"{len(matches)} '{INDEX_HEADING}' cells, expected 1")
    return matches[0]


def _section_headings(cells) -> list:
    """Return the notebook's ``## `` section titles in order, excluding the index itself."""
    headings = []
    for cell in cells:
        if cell["cell_type"] != "markdown":
            continue
        for line in _source(cell).splitlines():
            if line.startswith("## ") and line.strip() != INDEX_HEADING:
                headings.append(line[3:].strip())
    return headings


def _index_entries(body: str) -> list:
    """Parse an index cell body into ``(title, anchor)`` pairs."""
    entries = []
    for line in body.splitlines()[1:]:
        if not line.strip():
            continue
        match = _INDEX_ENTRY.match(line.strip())
        if match is None:
            raise NotebookShapeError(f"index cell holds a line that is not a numbered entry: {line.strip()!r}")
        entries.append((match["title"], match["anchor"]))
    return entries


def _index_body(entries, headings) -> str:
    """Rebuild an index that lists every section, keeping the wording of the entries already there.

    Existing entries are matched onto the headings in order and case-insensitively, because 27
    notebooks spell the heading ``## Basic Test`` while their index says ``Basic test``. A matched
    entry keeps its own title and anchor; a heading with no entry gets one. Numbering is then
    redone from 1, which is what the hand-maintained numbering had drifted away from.

    Raises
    ------
    NotebookShapeError
        If an existing entry has no heading to match, which means the index and the notebook
        disagree about more than the missing entries.
    """
    matched = {}
    position = 0
    for title, anchor in entries:
        while position < len(headings) and headings[position].lower() != title.lower():
            position += 1
        if position == len(headings):
            raise NotebookShapeError(f"index entry {title!r} has no '## ' heading to match")
        matched[position] = (title, anchor)
        position += 1

    lines = [INDEX_HEADING]
    for number, heading in enumerate(headings, start=1):
        title, anchor = matched.get(number - 1, (heading, "#" + heading.replace(" ", "-")))
        lines.append(f"{number}. [{title}]({anchor})")
    return "\n".join(lines)


def patch_notebook_index(path: Path) -> bool:
    """Make a clock notebook's ``## Index`` list every section, numbered from 1.

    Parameters
    ----------
    path : pathlib.Path
        Path to a notebook in ``clocks/notebooks``.

    Returns
    -------
    bool
        True if the index was rewritten, False if it already matched the notebook's sections.

    Raises
    ------
    NotebookShapeError
        If the notebook has no single index cell, if that cell holds anything but numbered
        entries, or if an entry cannot be matched to a section heading. The file on disk is
        left untouched when it raises.
    """
    notebook = json.loads(path.read_text())
    cells = notebook["cells"]

    index = _find_index_cell(cells)
    body = _source(cells[index])
    headings = _section_headings(cells)
    if not headings:
        raise NotebookShapeError("no '## ' section headings to index")

    rebuilt = _index_body(_index_entries(body), headings)
    if rebuilt == body.rstrip("\n"):
        return False

    cells[index]["source"] = rebuilt.splitlines(keepends=True)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    return True


def _patch_notebooks(notebooks_dir: Path) -> int:
    """Patch every notebook in a directory, reindex it, and report what changed.

    Returns
    -------
    int
        0 on success, 1 if the directory holds no notebooks or if any notebook
        was skipped because its layout was not recognised.
    """
    paths = sorted(notebooks_dir.glob("*.ipynb"))
    if not paths:
        print(f"no .ipynb files found in {notebooks_dir}", file=sys.stderr)
        return 1

    changed, reindexed, skipped = [], [], []
    for path in paths:
        try:
            if patch_notebook(path):
                changed.append(path.name)
            if patch_notebook_index(path):
                reindexed.append(path.name)
        except NotebookShapeError as error:
            skipped.append(f"{path.name} ({error})")

    print(f"scanned {len(paths)} notebooks, patched {len(changed)}: {', '.join(changed) if changed else 'none'}")
    print(f"reindexed {len(reindexed)}: {', '.join(reindexed) if reindexed else 'none'}")
    if skipped:
        print(f"skipped {len(skipped)} notebooks needing a hand: {'; '.join(skipped)}")
        return 1
    return 0


def main() -> int:
    """Patch every clock in a weights directory and report what changed.

    Returns
    -------
    int
        0 on success, 1 if the weights directory contains no ``.pt`` files
        (which almost always means the path is wrong rather than that there was
        nothing to do), or if any notebook was skipped as unrecognised.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights_dir", type=Path, help="directory of .pt clock weights")
    parser.add_argument("--notebooks-dir", type=Path, help="directory of clock notebooks to patch as well")
    arguments = parser.parse_args()

    paths = sorted(arguments.weights_dir.glob("*.pt"))
    if not paths:
        print(f"no .pt files found in {arguments.weights_dir}", file=sys.stderr)
        return 1

    changed = [path.name for path in paths if patch_weight_file(path)]
    print(f"scanned {len(paths)} clocks, patched {len(changed)}: {', '.join(changed) if changed else 'none'}")

    if arguments.notebooks_dir is None:
        return 0
    return _patch_notebooks(arguments.notebooks_dir)


if __name__ == "__main__":
    raise SystemExit(main())
