# Hugging Face Data Host Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move every current pyaging package and notebook data dependency from S3 to `lucascamillomd/pyaging-data`, then publish and verify `pyaging==0.3.0` on PyPI.

**Architecture:** A focused internal module wraps `hf_hub_download` and is the only Hub boundary used by package code. Runtime assets live at the HF repository root so `local_dir=dir` preserves flat paths such as `pyaging_data/horvath2013.pt`; notebook-only dependencies retain `supporting_files/...` paths. Maintainer-only Make targets publish weights before metadata, while CI stays hermetic and large online validation runs once before the immutable PyPI release.

**Tech Stack:** Python 3.9–3.13, PyTorch, `huggingface_hub`, pytest, uv/tox, Sphinx, Jupyter notebooks, GNU Make, GitHub Actions, Hugging Face Hub/Xet, PyPI.

---

## Execution prerequisite

Execute this plan in a dedicated `codex/hf-data-migration` worktree created from the commit containing this plan (a descendant of design commit `208bded`). Preserve the user's main checkout. Do not publish to HF, push a release tag, or upload to PyPI from an uncommitted or dirty worktree.

PyPI preflight performed while writing this plan:

- `https://pypi.org/pypi/pyaging/0.3.0/json` returned HTTP 404.
- PyPI reported `0.2.0` as the current latest release.
- GitHub CLI is authenticated with repository scope.
- The current local HF token can write as `lucascamillomd`, but it is broad rather than repository-scoped. Replace it with a fine-grained token before Task 9.
- The hermetic documentation baseline passes (`2 passed`). A repository-wide Ruff run has 40 pre-existing violations in unrelated legacy files; tasks lint new files and use `git diff --check` rather than expanding scope to legacy cleanup.

## File map

### New files

- `pyaging/utils/_hf.py` — sole package boundary for HF downloads and typed errors.
- `tests/utils/test_hf.py` — unit tests for HF arguments and error translation.
- `tests/predict/test_hf_loading.py` — clock and metadata download/load behavior.
- `tests/data/test_hf_data.py` — example-data filename mapping and flat paths.
- `tests/preprocess/test_hf_metadata.py` — Ensembl download/load behavior.
- `tests/integration/test_hf_smoke.py` — opt-in anonymous online smoke test.
- `tests/test_notebook_hosts.py` — source and docs notebook host guard.
- `tests/test_no_aws_dependencies.py` — active-source AWS dependency guard.
- `tests/test_release_configuration.py` — Makefile and workflow release safeguards.
- `tests/test_version.py` — `0.3.0` metadata consistency.
- `clocks/huggingface/README.md` — tracked source for the HF model repository card.
- `.github/workflows/ci.yml` — hermetic PR/main compatibility checks.

### Modified files

- `pyaging/utils/_utils.py` — metadata download, removal of S3 freshness code, and retention of the host-neutral notebook URL downloader.
- `pyaging/predict/_pred_utils.py` — HF clock download and precise missing-clock behavior.
- `pyaging/data/_data.py` — filename-based HF example downloads.
- `pyaging/preprocess/_preprocess_utils.py` — HF Ensembl download.
- `docs/source/make_clock_data.py` — HF metadata download for the catalog.
- `docs/source/test_make_clock_data.py` — generator routing test and host-neutral wording.
- `docs/Makefile` — host-neutral offline-fallback wording.
- 20 notebooks under `clocks/notebooks/` and their 20 copies under `docs/source/clock_notebooks/` — HF resolver URLs.
- `tests/predict/test_gold_standard.py` — mark the 25 GiB test as explicit full-catalog validation.
- `pyproject.toml` — HF dependency, pytest markers, tox dependencies/selection, version `0.3.0`.
- `uv.lock` — resolved dependency and project-version changes.
- `.gitignore` — local HF migration staging.
- `Makefile` — HF publishing targets and validated release order.
- `.github/workflows/release.yml` — tag-gated build, PyPI publish, and GitHub release.
- `pyaging/__init__.py` — version `0.3.0`.

### Removed files

- `.github/workflows/build.yml` — superseded by explicit CI/release workflows.
- `.github/workflows/publish.yml` — superseded by the tag-gated release workflow.
- `.github/workflows/test.yml` — superseded by `ci.yml` and release verification.

## Task 1: Add the internal Hugging Face download boundary

**Files:**
- Create: `pyaging/utils/_hf.py`
- Create: `tests/utils/test_hf.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [ ] **Step 1: Add failing unit tests for download arguments and typed errors**

Create `tests/utils/test_hf.py`:

```python
from unittest.mock import Mock

import pytest
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError

from pyaging.utils import _hf


def test_download_hf_file_uses_repo_main_and_flat_local_dir(monkeypatch, tmp_path):
    calls = {}

    def fake_download(**kwargs):
        calls.update(kwargs)
        return str(tmp_path / kwargs["filename"])

    monkeypatch.setattr(_hf, "hf_hub_download", fake_download)
    logger = Mock()

    path = _hf.download_hf_file("horvath2013.pt", str(tmp_path), logger, indent_level=2)

    assert path == str(tmp_path / "horvath2013.pt")
    assert calls == {
        "repo_id": "lucascamillomd/pyaging-data",
        "filename": "horvath2013.pt",
        "revision": "main",
        "local_dir": str(tmp_path),
    }
    logger.info.assert_called_once_with(
        f"Data available at {tmp_path / 'horvath2013.pt'}", indent_level=3
    )


def test_download_hf_file_maps_missing_entry(monkeypatch, tmp_path):
    def missing(**kwargs):
        raise EntryNotFoundError("missing")

    monkeypatch.setattr(_hf, "hf_hub_download", missing)

    with pytest.raises(_hf.PyAgingResourceNotFoundError) as exc_info:
        _hf.download_hf_file("missing.pt", str(tmp_path))

    assert "missing.pt" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, EntryNotFoundError)


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (401, _hf.PyAgingAuthenticationError),
        (403, _hf.PyAgingAuthenticationError),
        (429, _hf.PyAgingRateLimitError),
        (500, _hf.PyAgingDownloadError),
    ],
)
def test_download_hf_file_maps_http_errors(monkeypatch, tmp_path, status_code, expected):
    response = Mock(status_code=status_code)

    def failed(**kwargs):
        raise HfHubHTTPError("failed", response=response)

    monkeypatch.setattr(_hf, "hf_hub_download", failed)

    with pytest.raises(expected) as exc_info:
        _hf.download_hf_file("file.pt", str(tmp_path))

    assert isinstance(exc_info.value.__cause__, HfHubHTTPError)
```

- [ ] **Step 2: Run the new test and verify that it fails before implementation**

Run:

```bash
uv run --no-sync pytest tests/utils/test_hf.py -v
```

Expected: collection fails because `huggingface_hub` and `pyaging.utils._hf` are not yet available.

- [ ] **Step 3: Add the runtime dependency**

Run:

```bash
uv add 'huggingface_hub>=1.3,<2'
```

Expected: `pyproject.toml` and `uv.lock` add `huggingface_hub`; `requests` remains until Task 3.

- [ ] **Step 4: Implement the focused HF boundary**

Create `pyaging/utils/_hf.py`:

```python
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
)

REPO_ID = "lucascamillomd/pyaging-data"
REVISION = "main"


class PyAgingHubError(RuntimeError):
    """Base error for pyaging-owned Hugging Face resources."""


class PyAgingResourceNotFoundError(PyAgingHubError):
    """The requested file does not exist in the pyaging data repository."""


class PyAgingRepositoryError(PyAgingHubError):
    """The pyaging data repository is unavailable or misconfigured."""


class PyAgingAuthenticationError(PyAgingHubError):
    """Hugging Face rejected the current credentials."""


class PyAgingRateLimitError(PyAgingHubError):
    """Hugging Face rate-limited the download request."""


class PyAgingDownloadError(PyAgingHubError):
    """A remote, cache, or filesystem failure prevented the download."""


def download_hf_file(filename: str, dir: str, logger=None, indent_level: int = 1) -> str:
    """Download one public pyaging asset from the live HF repository."""
    try:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            revision=REVISION,
            local_dir=dir,
        )
    except LocalEntryNotFoundError as exc:
        raise PyAgingDownloadError(
            f"Could not download {filename} and no usable local HF cache entry exists."
        ) from exc
    except EntryNotFoundError as exc:
        raise PyAgingResourceNotFoundError(
            f"{filename} was not found in {REPO_ID} at {REVISION}."
        ) from exc
    except RepositoryNotFoundError as exc:
        raise PyAgingRepositoryError(
            f"The public data repository {REPO_ID} is unavailable."
        ) from exc
    except HfHubHTTPError as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code in {401, 403}:
            raise PyAgingAuthenticationError(
                f"Hugging Face rejected access while downloading {filename}."
            ) from exc
        if status_code == 429:
            raise PyAgingRateLimitError(
                f"Hugging Face rate-limited the download of {filename}."
            ) from exc
        raise PyAgingDownloadError(f"Hugging Face could not download {filename}.") from exc
    except OSError as exc:
        raise PyAgingDownloadError(f"Could not store {filename} under {dir}.") from exc

    if logger is not None:
        logger.info(f"Data available at {path}", indent_level=indent_level + 1)
    return path
```

- [ ] **Step 5: Run the focused tests**

Run:

```bash
uv run pytest tests/utils/test_hf.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit the download boundary**

```bash
git add pyaging/utils/_hf.py tests/utils/test_hf.py pyproject.toml uv.lock
git commit -m "feat: add Hugging Face data download boundary"
```

## Task 2: Route clock weights and aggregate metadata through HF

**Files:**
- Create: `tests/predict/test_hf_loading.py`
- Modify: `pyaging/predict/_pred_utils.py`
- Modify: `pyaging/utils/_utils.py`

- [ ] **Step 1: Write failing tests for clock and metadata routing**

Create `tests/predict/test_hf_loading.py`:

```python
from unittest.mock import Mock

import pytest

from pyaging.predict import _pred_utils
from pyaging.utils import _utils
from pyaging.utils._hf import PyAgingDownloadError, PyAgingResourceNotFoundError


def test_load_clock_uses_returned_hf_path(monkeypatch, tmp_path):
    expected_path = tmp_path / "horvath2013.pt"
    model = Mock()
    calls = {}

    def fake_download(filename, dir, logger, indent_level):
        calls.update(filename=filename, dir=dir, indent_level=indent_level)
        return str(expected_path)

    monkeypatch.setattr(_pred_utils, "download_hf_file", fake_download)
    monkeypatch.setattr(_pred_utils.torch, "load", Mock(return_value=model))
    logger = Mock()

    result = _pred_utils.load_clock("Horvath2013", "cpu", str(tmp_path), logger)

    assert result is model
    assert calls == {"filename": "horvath2013.pt", "dir": str(tmp_path), "indent_level": 2}
    _pred_utils.torch.load.assert_called_once_with(str(expected_path), weights_only=False)
    model.to.assert_any_call(_pred_utils.torch.float64)
    model.to.assert_any_call("cpu")
    model.eval.assert_called_once_with()


def test_load_clock_only_maps_missing_resource_to_name_error(monkeypatch, tmp_path):
    missing = PyAgingResourceNotFoundError("missing")
    monkeypatch.setattr(
        _pred_utils,
        "download_hf_file",
        Mock(side_effect=missing),
    )

    with pytest.raises(NameError) as exc_info:
        _pred_utils.load_clock("missing", "cpu", str(tmp_path), Mock())

    assert exc_info.value.__cause__ is missing


def test_load_clock_preserves_non_missing_download_error(monkeypatch, tmp_path):
    failed = PyAgingDownloadError("network failed")
    monkeypatch.setattr(
        _pred_utils,
        "download_hf_file",
        Mock(side_effect=failed),
    )

    with pytest.raises(PyAgingDownloadError, match="network failed"):
        _pred_utils.load_clock("horvath2013", "cpu", str(tmp_path), Mock())


def test_load_clock_metadata_uses_returned_hf_path(monkeypatch, tmp_path):
    expected_path = tmp_path / "all_clock_metadata.pt"
    monkeypatch.setattr(
        _utils,
        "download_hf_file",
        Mock(return_value=str(expected_path)),
    )
    monkeypatch.setattr(_utils.torch, "load", Mock(return_value={"clock": {}}))
    logger = Mock()

    result = _utils.load_clock_metadata(str(tmp_path), logger)

    assert result == {"clock": {}}
    _utils.download_hf_file.assert_called_once_with(
        "all_clock_metadata.pt", str(tmp_path), logger, indent_level=2
    )
    _utils.torch.load.assert_called_once_with(str(expected_path), weights_only=False)
```

- [ ] **Step 2: Run the focused tests and observe S3-path failures**

Run:

```bash
uv run pytest tests/predict/test_hf_loading.py -v
```

Expected: failures show that `download_hf_file` is not imported and existing code still constructs S3 URLs/local paths.

- [ ] **Step 3: Replace clock loading with the HF helper**

In `pyaging/predict/_pred_utils.py`, remove the unused `urlretrieve` import, stop importing `download`, import the HF helper/errors directly, and replace `load_clock` with:

```python
from ..utils._hf import PyAgingResourceNotFoundError, download_hf_file


@progress("Load clock")
def load_clock(clock_name: str, device: str, dir: str, logger, indent_level: int = 2) -> Tuple:
    """Load one aging clock from the public pyaging Hugging Face repository."""
    clock_name = clock_name.lower()
    try:
        weights_path = download_hf_file(
            f"{clock_name}.pt",
            dir,
            logger,
            indent_level=indent_level,
        )
    except PyAgingResourceNotFoundError as exc:
        message = (
            f"Clock {clock_name} is not available on pyaging. "
            "Please refer to the clock names in the clock glossary table "
            "at pyaging.readthedocs.io."
        )
        logger.error(message, indent_level=indent_level + 1)
        raise NameError(message) from exc

    clock = torch.load(weights_path, weights_only=False)
    clock.to(torch.float64)
    clock.to(device)
    clock.eval()
    return clock
```

Keep the public signature unchanged. Update its existing docstring sections to say Hugging Face rather than a constructed remote URL.

- [ ] **Step 4: Replace aggregate metadata loading with the HF helper**

In `pyaging/utils/_utils.py`, add `from ._hf import download_hf_file` and replace the function body with:

```python
@progress("Load all clock metadata")
def load_clock_metadata(dir: str, logger, indent_level: int = 2) -> dict:
    """Load aggregate clock metadata from the public pyaging HF repository."""
    metadata_path = download_hf_file(
        "all_clock_metadata.pt",
        dir,
        logger,
        indent_level=indent_level,
    )
    return torch.load(metadata_path, weights_only=False)
```

Update the existing long docstring to describe HF and the caller-selected directory without changing the public signature.

- [ ] **Step 5: Run focused and existing utility tests**

Run:

```bash
uv run pytest tests/utils/test_hf.py tests/predict/test_hf_loading.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit clock and metadata routing**

```bash
git add pyaging/predict/_pred_utils.py pyaging/utils/_utils.py tests/predict/test_hf_loading.py
git commit -m "feat: load clocks and metadata from Hugging Face"
```

## Task 3: Route example and Ensembl data, then remove S3-specific freshness code

**Files:**
- Create: `tests/data/test_hf_data.py`
- Create: `tests/preprocess/test_hf_metadata.py`
- Modify: `pyaging/data/_data.py`
- Modify: `pyaging/preprocess/_preprocess_utils.py`
- Modify: `pyaging/utils/_utils.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [ ] **Step 1: Write failing tests for flat example and Ensembl paths**

Create `tests/data/test_hf_data.py`:

```python
from unittest.mock import Mock

import pytest

from pyaging.data import _data


@pytest.mark.parametrize(
    ("data_type", "filename"),
    [
        ("GSE130735", "GSE130735_subset.pkl"),
        ("GSE193140", "GSE193140.pkl"),
        ("GSE139307", "GSE139307.pkl"),
        ("GSE223748", "GSE223748_subset.pkl"),
        ("ENCFF386QWG", "ENCFF386QWG.bigWig"),
        ("GSE65765", "GSE65765_CPM.pkl"),
        ("blood_chemistry_example", "blood_chemistry_example.pkl"),
    ],
)
def test_download_example_data_maps_name_to_root_hf_filename(monkeypatch, tmp_path, data_type, filename):
    download = Mock(return_value=str(tmp_path / filename))
    monkeypatch.setattr(_data, "download_hf_file", download)
    monkeypatch.setattr(_data.LoggerManager, "gen_logger", Mock(return_value=Mock()))

    _data.download_example_data(data_type, dir=str(tmp_path))

    download.assert_called_once()
    assert download.call_args.args[:2] == (filename, str(tmp_path))
```

Create `tests/preprocess/test_hf_metadata.py`:

```python
from unittest.mock import Mock

import pandas as pd

from pyaging.preprocess import _preprocess_utils


def test_load_ensembl_metadata_reads_returned_hf_path(monkeypatch, tmp_path):
    expected_path = tmp_path / "Ensembl-105-EnsDb-for-Homo-sapiens-genes.csv"
    pd.DataFrame(
        {
            "gene_id": ["keep", "drop"],
            "chr": ["1", "MT"],
        }
    ).to_csv(expected_path, index=False)
    download = Mock(return_value=str(expected_path))
    monkeypatch.setattr(_preprocess_utils, "download_hf_file", download)
    logger = Mock()

    result = _preprocess_utils.load_ensembl_metadata(str(tmp_path), logger)

    download.assert_called_once_with(
        "Ensembl-105-EnsDb-for-Homo-sapiens-genes.csv",
        str(tmp_path),
        logger,
        indent_level=1,
    )
    assert result.index.tolist() == ["keep"]
```

- [ ] **Step 2: Run the tests and verify current URL-based behavior fails**

Run:

```bash
uv run pytest tests/data/test_hf_data.py tests/preprocess/test_hf_metadata.py -v
```

Expected: failures because both modules still import/call `download` with S3 URLs.

- [ ] **Step 3: Replace example URL mapping with filename mapping**

In `pyaging/data/_data.py`, import `download_hf_file` from `..utils._hf` and define:

```python
_EXAMPLE_DATA_FILENAMES = {
    "GSE130735": "GSE130735_subset.pkl",
    "GSE193140": "GSE193140.pkl",
    "GSE139307": "GSE139307.pkl",
    "GSE223748": "GSE223748_subset.pkl",
    "ENCFF386QWG": "ENCFF386QWG.bigWig",
    "GSE65765": "GSE65765_CPM.pkl",
    "blood_chemistry_example": "blood_chemistry_example.pkl",
}
```

Keep validation/logging unchanged, but replace the URL lookup and final call with:

```python
filename = _EXAMPLE_DATA_FILENAMES[data_type]
download_hf_file(filename, dir, logger, indent_level=1)
```

Update the docstring to describe public HF-hosted examples.

- [ ] **Step 4: Replace Ensembl URL download with the returned HF path**

In `pyaging/preprocess/_preprocess_utils.py`, stop importing `download`, remove the now-unused `os` import, import `download_hf_file` from `..utils._hf`, and use:

```python
genes_path = download_hf_file(
    "Ensembl-105-EnsDb-for-Homo-sapiens-genes.csv",
    dir,
    logger,
    indent_level=1,
)
genes = pd.read_csv(genes_path)
```

Leave chromosome filtering and the public signature unchanged.

- [ ] **Step 5: Remove the S3 freshness helper and obsolete dependencies while retaining notebook compatibility**

From `pyaging/utils/_utils.py`, delete `is_newer_than_target` and remove its imports `datetime`, `pytz`, and `requests`. Retain the public, host-neutral `download(url, dir, logger, indent_level=1)` plus its `os` and `urlretrieve` imports because maintained clock notebooks use it to flatten resolver URLs into their working directories. Its docstring must not mention AWS or S3.

From `pyproject.toml`:

- remove `requests` from `[project].dependencies`;
- replace `requests` in the tox `deps` block with `huggingface_hub>=1.3,<2`.

Run:

```bash
uv lock
rg -n 'is_newer_than_target|import requests|import pytz|from datetime import datetime' pyaging pyproject.toml
rg -n '^def download\(' pyaging/utils/_utils.py
```

Expected: the forbidden freshness/dependency scan has no matches and exactly one host-neutral `download` definition remains for notebooks.

- [ ] **Step 6: Run focused tests and lint**

Run:

```bash
uv run pytest tests/utils/test_hf.py tests/predict/test_hf_loading.py tests/data/test_hf_data.py tests/preprocess/test_hf_metadata.py -v
uv run ruff check pyaging/utils/_hf.py tests/utils/test_hf.py tests/predict/test_hf_loading.py tests/data/test_hf_data.py tests/preprocess/test_hf_metadata.py
git diff --check
```

Expected: all tests and lint pass.

- [ ] **Step 7: Commit package-wide HF routing**

```bash
git add pyaging/data/_data.py pyaging/preprocess/_preprocess_utils.py pyaging/utils/_utils.py tests/data/test_hf_data.py tests/preprocess/test_hf_metadata.py pyproject.toml uv.lock
git commit -m "feat: migrate package data files to Hugging Face"
```

## Task 4: Migrate documentation metadata generation

**Files:**
- Modify: `docs/source/make_clock_data.py`
- Modify: `docs/source/test_make_clock_data.py`
- Modify: `docs/Makefile`

- [ ] **Step 1: Add a failing generator-routing test**

Add `Mock`, `torch`, and `make_clock_data` to the module's existing top-level import block, then append this test to `docs/source/test_make_clock_data.py`:

```python
def test_generate_downloads_metadata_from_hf(monkeypatch, tmp_path):
    source = tmp_path / "source_metadata.pt"
    static = tmp_path / "static"
    torch.save(
        {
            "example": {
                "approved_by_author": "✅",
                "data_type": "methylation",
            }
        },
        source,
    )
    download = Mock(return_value=str(source))
    monkeypatch.setattr(make_clock_data, "STATIC", str(static))
    monkeypatch.setattr(make_clock_data, "download_hf_file", download)

    count = make_clock_data.generate()

    assert count == 1
    download.assert_called_once_with("all_clock_metadata.pt", str(static))
    assert (static / "clocks.json").exists()
    assert (static / "clock_glossary.csv").exists()
```

Change the module-level test docstring from “WITHOUT downloading from S3” to “WITHOUT network access.”

- [ ] **Step 2: Run the docs test and verify it fails**

Run:

```bash
uv run pytest docs/source/test_make_clock_data.py -v
```

Expected: failure because `make_clock_data` does not expose/use `download_hf_file`.

- [ ] **Step 3: Route the generator through the internal HF helper**

In `docs/source/make_clock_data.py`:

- remove `urlretrieve` and the S3 `URL` constant;
- add `from pyaging.utils._hf import download_hf_file`;
- update the module docstring to say public Hugging Face metadata;
- replace the start of `generate()` with:

```python
def generate():
    os.makedirs(STATIC, exist_ok=True)
    pt_path = download_hf_file("all_clock_metadata.pt", STATIC)
    meta = torch.load(pt_path, weights_only=False)
```

Keep JSON/CSV generation unchanged.

In `docs/Makefile`, change the comment to “with a committed fallback if the remote host is unreachable.”

- [ ] **Step 4: Run documentation data tests**

Run:

```bash
uv run pytest docs/source/test_make_clock_data.py -v
```

Expected: all tests pass without network access.

- [ ] **Step 5: Commit documentation routing**

```bash
git add docs/source/make_clock_data.py docs/source/test_make_clock_data.py docs/Makefile
git commit -m "docs: load clock catalog data from Hugging Face"
```

## Task 5: Replace notebook S3 URLs with HF resolver URLs

**Files:**
- Create: `tests/test_notebook_hosts.py`
- Modify: 20 notebooks under `clocks/notebooks/`
- Modify: matching 20 notebooks under `docs/source/clock_notebooks/`

- [ ] **Step 1: Write a failing notebook-host guard**

Create `tests/test_notebook_hosts.py`:

```python
from pathlib import Path


NOTEBOOK_ROOTS = [
    Path("clocks/notebooks"),
    Path("docs/source/clock_notebooks"),
]


def test_clock_notebooks_do_not_reference_pyaging_s3():
    offenders = []
    for root in NOTEBOOK_ROOTS:
        for path in root.glob("*.ipynb"):
            if "pyaging.s3" in path.read_text(encoding="utf-8"):
                offenders.append(str(path))
    assert offenders == []
```

- [ ] **Step 2: Run the guard and verify the 40 copied notebooks fail**

Run:

```bash
uv run pytest tests/test_notebook_hosts.py -v
```

Expected: failure listing 20 source notebooks and their 20 documentation copies.

- [ ] **Step 3: Perform the mechanical resolver-host replacement**

Run:

```bash
files=$(rg -l 'https://pyaging\.s3\.amazonaws\.com' clocks/notebooks docs/source/clock_notebooks --glob '*.ipynb')
perl -pi -e 's#https://pyaging\.s3\.amazonaws\.com/#https://huggingface.co/lucascamillomd/pyaging-data/resolve/main/#g' $files
```

Also replace the regional endpoint used by `cpgptgrimage3.ipynb`:

```bash
files=$(rg -l 'https://pyaging\.s3\.us-east-1\.amazonaws\.com' clocks/notebooks docs/source/clock_notebooks --glob '*.ipynb')
perl -pi -e 's#https://pyaging\.s3\.us-east-1\.amazonaws\.com/#https://huggingface.co/lucascamillomd/pyaging-data/resolve/main/#g' $files
```

This preserves every `supporting_files/...` path and only changes the host/prefix.

- [ ] **Step 4: Verify notebook JSON and host routing**

Run:

```bash
uv run python -c 'import json, pathlib; [json.load(p.open()) for root in (pathlib.Path("clocks/notebooks"), pathlib.Path("docs/source/clock_notebooks")) for p in root.glob("*.ipynb")]'
uv run pytest tests/test_notebook_hosts.py -v
rg -n 'huggingface.co/lucascamillomd/pyaging-data/resolve/main/supporting_files' clocks/notebooks docs/source/clock_notebooks --glob '*.ipynb' | head
```

Expected: JSON parsing succeeds, the test passes, and HF resolver URLs are present.

- [ ] **Step 5: Commit notebook host migration**

```bash
git add clocks/notebooks docs/source/clock_notebooks tests/test_notebook_hosts.py
git commit -m "docs: migrate notebook dependencies to Hugging Face"
```

## Task 6: Make large and online tests explicit

**Files:**
- Create: `tests/integration/test_hf_smoke.py`
- Modify: `tests/predict/test_gold_standard.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [ ] **Step 1: Mark the full catalog test and register explicit markers**

In `tests/predict/test_gold_standard.py`, add after imports:

```python
pytestmark = pytest.mark.full_catalog
```

Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = "-m 'not full_catalog and not online'"
markers = [
    "full_catalog: downloads and validates the complete clock catalog (~25 GiB)",
    "online: accesses public remote services",
]
```

Change the tox command to:

```ini
commands = pytest -m "not full_catalog and not online" tests/
```

- [ ] **Step 2: Add the opt-in anonymous HF smoke test**

Create `tests/integration/test_hf_smoke.py`:

```python
import os

import pytest
import torch

from pyaging.utils._hf import download_hf_file

pytestmark = pytest.mark.online


def test_public_hf_repository_serves_small_runtime_assets(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.delenv("HF_TOKEN", raising=False)

    metadata_path = download_hf_file("all_clock_metadata.pt", str(tmp_path / "data"))
    clock_path = download_hf_file("horvath2013.pt", str(tmp_path / "data"))
    example_path = download_hf_file("blood_chemistry_example.pkl", str(tmp_path / "data"))

    assert os.path.basename(metadata_path) == "all_clock_metadata.pt"
    assert os.path.basename(clock_path) == "horvath2013.pt"
    assert os.path.basename(example_path) == "blood_chemistry_example.pkl"
    assert "horvath2013" in torch.load(metadata_path, weights_only=False)
    assert torch.load(clock_path, weights_only=False).metadata["clock_name"].lower() == "horvath2013"
```

- [ ] **Step 3: Verify default tests do not collect the 25 GiB catalog or online smoke**

Run:

```bash
uv lock
uv run pytest --collect-only -q
uv run pytest tests -q
```

Expected: collection output marks `full_catalog` and `online` tests deselected; hermetic tests pass without network downloads.

- [ ] **Step 4: Commit test selection safeguards**

```bash
git add tests/predict/test_gold_standard.py tests/integration/test_hf_smoke.py pyproject.toml uv.lock
git commit -m "test: make large HF validation opt in"
```

## Task 7: Add sole-maintainer HF publishing commands and repository card

**Files:**
- Create: `clocks/huggingface/README.md`
- Create: `tests/test_release_configuration.py`
- Modify: `.gitignore`
- Modify: `Makefile`

- [ ] **Step 1: Write failing Makefile/repository-card tests**

Create `tests/test_release_configuration.py`:

```python
from pathlib import Path


MAKEFILE = Path("Makefile").read_text(encoding="utf-8")
HF_README = Path("clocks/huggingface/README.md")


def test_makefile_uses_only_hf_publish_targets():
    assert "upload-to-s3" not in MAKEFILE
    assert "aws s3" not in MAKEFILE
    assert "HF_REPO_ID ?= lucascamillomd/pyaging-data" in MAKEFILE
    assert "verify-hf-auth:" in MAKEFILE
    assert "upload-clocks-to-hf:" in MAKEFILE
    assert "upload-static-data-to-hf:" in MAKEFILE


def test_release_uploads_only_after_validation():
    release_line = next(line for line in MAKEFILE.splitlines() if line.startswith("release:"))
    assert release_line.index("test") < release_line.index("upload-clocks-to-hf")
    assert release_line.index("docs") < release_line.index("upload-clocks-to-hf")


def test_hf_repository_card_documents_security_and_licensing():
    text = HF_README.read_text(encoding="utf-8")
    assert "license: other" in text
    assert "torch.load(..., weights_only=False)" in text
    assert "research-only" in text
    assert "lucascamillomd/pyaging" in text
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
uv run pytest tests/test_release_configuration.py -v
```

Expected: failures because the Makefile still uses AWS and the card does not exist.

- [ ] **Step 3: Create the tracked HF repository card source**

Create `clocks/huggingface/README.md`:

```markdown
---
license: other
library_name: pyaging
tags:
  - biology
  - aging
  - biological-age
  - pytorch
---

# pyaging data

This public repository contains the model weights and data files used by
[`lucascamillomd/pyaging`](https://github.com/lucascamillomd/pyaging).

## Contents

- Root-level `*.pt` files are the current pyaging clock models.
- `all_clock_metadata.pt` is the live aggregate clock catalog.
- Root-level example files support the pyaging tutorials.
- `supporting_files/` contains dependencies used to construct or document clocks.

Files used by the Python package are intentionally stored at the repository root so
`hf_hub_download(..., local_dir="pyaging_data")` preserves existing flat local paths.
The `main` branch is the live data release and may change independently of the Python
package version.

## Licensing and provenance

This is a mixed-provenance research collection, so the repository license is `other`.
The pyaging BSD license does not grant additional rights to third-party clock weights or
source datasets. Consult each clock's embedded metadata, cited publication, and notes
before use. Some clocks are marked research-only or have separate commercial terms.

## Security

Clock files are trusted Python/PyTorch objects loaded by pyaging with
`torch.load(..., weights_only=False)`. Loading a malicious pickle can execute code. Only
load these files from this official repository and review unexpected repository changes.

## Publishing policy

The repository is maintained solely by Lucas Paulo de Lima Camillo (`lucascamillomd`).
Weights are uploaded before aggregate metadata so the catalog never advertises a missing
clock file. Public users need no Hugging Face token to download files.
```

- [ ] **Step 4: Replace the AWS Make target with guarded HF targets**

At the top of `Makefile`, set:

```make
VERSION ?= v0.3.0
HF_REPO_ID ?= lucascamillomd/pyaging-data
HF_REPO_OWNER ?= lucascamillomd
HF_STATIC_DIR ?= hf_static_data
```

Update `.PHONY` to include `verify-hf-auth`, `create-hf-data-repo`, `upload-clocks-to-hf`, and `upload-static-data-to-hf`, and remove `upload-to-s3`.

Replace the S3 target with:

```make
verify-hf-auth:
	@account=$$(uv run hf auth whoami --format json | uv run python -c 'import json, sys; print(json.load(sys.stdin)["name"])'); \
	if [ "$$account" != "$(HF_REPO_OWNER)" ]; then \
		echo "Expected HF account $(HF_REPO_OWNER), got $$account"; \
		exit 1; \
	fi

create-hf-data-repo: verify-hf-auth
	uv run hf repos create $(HF_REPO_ID) --type model --exist-ok
	uv run hf upload $(HF_REPO_ID) clocks/huggingface/README.md README.md --type model --commit-message "Document pyaging data repository"

upload-clocks-to-hf: verify-hf-auth
	@echo "Uploading changed clock weights to Hugging Face..."
	uv run hf upload $(HF_REPO_ID) clocks/weights . --type model --commit-message "Update pyaging clock weights"
	@echo "Publishing aggregate metadata after weights..."
	uv run hf upload $(HF_REPO_ID) clocks/metadata/all_clock_metadata.pt all_clock_metadata.pt --type model --commit-message "Update aggregate clock metadata"
	@uv run hf models info $(HF_REPO_ID) --format json | uv run python -c 'import json, sys; print("HF revision:", json.load(sys.stdin)["sha"])'

upload-static-data-to-hf: verify-hf-auth
	@test -d "$(HF_STATIC_DIR)/repo" || { echo "Missing $(HF_STATIC_DIR)/repo staging directory"; exit 1; }
	uv run hf upload $(HF_REPO_ID) $(HF_STATIC_DIR)/repo . --type model --commit-message "Add current pyaging static data dependencies"
```

Change release prerequisites so validation precedes publication:

```make
release: version lint format update build install update-clocks-notebooks update-all-clocks process-tutorials test test-tutorials docs upload-clocks-to-hf commit tag
	@echo "Release $(VERSION) completed successfully"

release-slim: version lint format update build install update-all-clocks test docs upload-clocks-to-hf commit tag
	@echo "Release $(VERSION) (slim) completed successfully"
```

Append `hf_static_data/` to `.gitignore`.

- [ ] **Step 5: Run release-configuration tests without uploading**

Run:

```bash
uv run pytest tests/test_release_configuration.py -v
make -n upload-clocks-to-hf
make -n release-slim VERSION=v0.3.0 | tail -40
```

Expected: tests pass; dry-run output contains `hf upload`, places tests/docs before upload, and contains no `aws s3` command.

- [ ] **Step 6: Commit publishing infrastructure**

```bash
git add Makefile .gitignore clocks/huggingface/README.md tests/test_release_configuration.py
git commit -m "build: replace S3 publishing with Hugging Face"
```

## Task 8: Add the final active-source AWS guard and validate docs

**Files:**
- Create: `tests/test_no_aws_dependencies.py`

- [ ] **Step 1: Add the comprehensive active-source guard**

Create `tests/test_no_aws_dependencies.py`:

```python
from pathlib import Path


ROOTS = [
    Path("pyaging"),
    Path("clocks/notebooks"),
    Path("docs/source"),
    Path(".github/workflows"),
]
FILES = [Path("Makefile"), Path("docs/Makefile")]
SUFFIXES = {".py", ".ipynb", ".rst", ".md", ".yml", ".yaml"}


def active_files():
    yield from FILES
    for root in ROOTS:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in SUFFIXES:
                yield path


def test_active_sources_have_no_pyaging_s3_or_aws_cli_dependency():
    offenders = {}
    for path in active_files():
        text = path.read_text(encoding="utf-8")
        matches = [needle for needle in ("pyaging.s3.amazonaws.com", "aws s3") if needle in text]
        if matches:
            offenders[str(path)] = matches
    assert offenders == {}
```

- [ ] **Step 2: Run the guard and remove any remaining active wording/reference**

Run:

```bash
uv run pytest tests/test_no_aws_dependencies.py -v
rg -n 'pyaging\.s3\.amazonaws\.com|aws s3' pyaging clocks/notebooks docs/source Makefile docs/Makefile .github/workflows
```

Expected: test passes and `rg` returns no matches. Do not edit historical files under `docs/superpowers/`.

- [ ] **Step 3: Run hermetic package and documentation validation**

Run:

```bash
uv run pytest tests docs/source/test_make_clock_data.py -m 'not full_catalog and not online' -v
uv run ruff check pyaging/utils/_hf.py docs/source/make_clock_data.py docs/source/test_make_clock_data.py tests --exclude tests/predict/test_gold_standard.py
git diff --check
uv run sphinx-build -M html docs/source docs/_build
```

Expected: all tests/lint pass and Sphinx builds, using committed catalog artifacts if HF is not populated yet.

- [ ] **Step 4: Commit the AWS regression guard**

```bash
git add tests/test_no_aws_dependencies.py
git commit -m "test: prevent active AWS data dependencies"
```

## Task 9: Create and populate `lucascamillomd/pyaging-data`

**Files:**
- Local ignored staging: `hf_static_data/repo/`
- External writes: public HF model repository `lucascamillomd/pyaging-data`

- [ ] **Step 1: Verify the current owner credential before creating the repository**

Run without printing the token:

```bash
uv run hf auth whoami --format json | jq '{name, auth: {type: .auth.type, role: .auth.accessToken.role, fineGrained: .auth.accessToken.fineGrained}}'
```

Expected: account `lucascamillomd` with write permission. The existing broad owner token may be used only to create the empty user-owned repository and its README in Step 2.

- [ ] **Step 2: Create the public model repo and publish its card**

Run:

```bash
make create-hf-data-repo
uv run hf models info lucascamillomd/pyaging-data --format json | jq '{id, private, author}'
```

Expected: `id` is `lucascamillomd/pyaging-data`, `private` is `false`, and `author` is `lucascamillomd`.

- [ ] **Step 3: Switch to a repository-scoped token before uploading data**

Now that the repository exists, create a fine-grained token in Hugging Face settings with write access restricted to `lucascamillomd/pyaging-data`, then authenticate without putting the token on the command line:

```bash
uv run hf auth login
uv run hf auth whoami --format json | jq '{name, auth: {type: .auth.type, role: .auth.accessToken.role, fineGrained: .auth.accessToken.fineGrained}}'
```

Expected: account `lucascamillomd`, write permission, and a non-null fine-grained restriction. Stop before uploading any data if this is not true.

- [ ] **Step 4: Prepare the one-time current static dependency staging tree**

Run exactly this zsh block from the repository root:

```zsh
set -euo pipefail
base='https://pyaging.s3.amazonaws.com'
stage='hf_static_data/repo'
mkdir -p "$stage/supporting_files/cpgpt_grimage3_dependencies"

root_sources=(
  'example_data/GSE130735_subset.pkl'
  'example_data/GSE193140.pkl'
  'example_data/GSE139307.pkl'
  'example_data/GSE223748_subset.pkl'
  'example_data/ENCFF386QWG.bigWig'
  'example_data/GSE65765_CPM.pkl'
  'example_data/blood_chemistry_example.pkl'
  'supporting_files/Ensembl-105-EnsDb-for-Homo-sapiens-genes.csv'
)

support_sources=(
  'supporting_files/CalcAllPCClocks.RData'
  'supporting_files/ElasticNet_DNAmProtein_Vars_model4.csv'
  'supporting_files/datMiniAnnotation3_Gold.csv'
  'supporting_files/grimage2.csv'
  'supporting_files/grimage2_subcomponents.csv'
  'supporting_files/cpgpt_grimage3_dependencies/cpgpt_grimage3_weights.csv'
  'supporting_files/cpgpt_grimage3_dependencies/cpgpt_grimage3_weights_all_datasets.csv'
  'supporting_files/cpgpt_grimage3_dependencies/cpgpt_pcgrimage3_pca_components.npy'
  'supporting_files/cpgpt_grimage3_dependencies/cpgpt_pcgrimage3_weights.csv'
  'supporting_files/cpgpt_grimage3_dependencies/input_scaler_mean.npy'
  'supporting_files/cpgpt_grimage3_dependencies/input_scaler_mean_all_datasets.npy'
  'supporting_files/cpgpt_grimage3_dependencies/input_scaler_scale.npy'
  'supporting_files/cpgpt_grimage3_dependencies/input_scaler_scale_all_datasets.npy'
  'supporting_files/cpgpt_grimage3_dependencies/pca_scaler_mean.npy'
  'supporting_files/cpgpt_grimage3_dependencies/pca_scaler_scale.npy'
  'supporting_files/cpgpt_grimage3_dependencies/reliable/cpgpt_grimage3_weights_all_datasets_reliable.csv'
  'supporting_files/cpgpt_grimage3_dependencies/reliable/input_scaler_mean_all_datasets_reliable.npy'
  'supporting_files/cpgpt_grimage3_dependencies/reliable/input_scaler_scale_all_datasets_reliable.npy'
)

for source in $root_sources; do
  destination="$stage/${source:t}"
  curl --fail --location --retry 5 --continue-at - "$base/$source" --output "$destination"
done

for source in $support_sources; do
  destination="$stage/$source"
  mkdir -p "${destination:h}"
  curl --fail --location --retry 5 --continue-at - "$base/$source" --output "$destination"
done

test "$(find "$stage" -type f | wc -l | tr -d ' ')" = 26
du -sh "$stage"
find "$stage" -type f -print0 | sort -z | xargs -0 shasum -a 256 > /tmp/pyaging-hf-static-sha256.txt
```

Expected: 26 files and a temporary checksum audit at `/tmp/pyaging-hf-static-sha256.txt`.

- [ ] **Step 5: Regenerate all clock objects and aggregate metadata for v0.3.0**

Run:

```bash
(cd clocks && uv run python update_all_clocks.py v0.3.0)
uv run python - <<'PY'
import torch

metadata = torch.load("clocks/metadata/all_clock_metadata.pt", weights_only=False)
assert len(metadata) == 173
assert {entry["version"] for entry in metadata.values()} == {"v0.3.0"}
print("verified 173 metadata entries at v0.3.0")
PY
```

Expected: all 173 ignored local weight files are rewritten with version `v0.3.0`, aggregate metadata is regenerated, and model predictions remain unchanged when Task 10 runs the gold standards.

- [ ] **Step 6: Upload static dependencies**

Run:

```bash
make upload-static-data-to-hf HF_STATIC_DIR=hf_static_data
```

Expected: successful Xet-backed commits; no excluded legacy path is uploaded.

- [ ] **Step 7: Upload all current weights, then metadata**

Run:

```bash
test "$(find clocks/weights -maxdepth 1 -type f -name '*.pt' | wc -l | tr -d ' ')" = 173
make upload-clocks-to-hf
```

Expected: weight upload completes before the metadata commit; the target prints the resulting HF revision.

## Task 10: Verify repository completeness, checksums, and clock behavior

**Files:**
- Read-only local sources and HF repository
- Temporary audit output under `/tmp`

- [ ] **Step 1: Verify exact current inventory and explicit exclusions**

Run:

```bash
uv run python - <<'PY'
from huggingface_hub import HfApi
import torch

repo_id = "lucascamillomd/pyaging-data"
files = set(HfApi().list_repo_files(repo_id))
metadata = torch.load("clocks/metadata/all_clock_metadata.pt", weights_only=False)
expected_weights = {f"{name.lower()}.pt" for name in metadata}
remote_weights = {path for path in files if "/" not in path and path.endswith(".pt") and path != "all_clock_metadata.pt"}

required_static = {
    "GSE130735_subset.pkl", "GSE193140.pkl", "GSE139307.pkl",
    "GSE223748_subset.pkl", "ENCFF386QWG.bigWig", "GSE65765_CPM.pkl",
    "blood_chemistry_example.pkl", "Ensembl-105-EnsDb-for-Homo-sapiens-genes.csv",
    "supporting_files/CalcAllPCClocks.RData",
    "supporting_files/ElasticNet_DNAmProtein_Vars_model4.csv",
    "supporting_files/datMiniAnnotation3_Gold.csv",
    "supporting_files/grimage2.csv",
    "supporting_files/grimage2_subcomponents.csv",
    "supporting_files/cpgpt_grimage3_dependencies/cpgpt_grimage3_weights.csv",
    "supporting_files/cpgpt_grimage3_dependencies/cpgpt_grimage3_weights_all_datasets.csv",
    "supporting_files/cpgpt_grimage3_dependencies/cpgpt_pcgrimage3_pca_components.npy",
    "supporting_files/cpgpt_grimage3_dependencies/cpgpt_pcgrimage3_weights.csv",
    "supporting_files/cpgpt_grimage3_dependencies/input_scaler_mean.npy",
    "supporting_files/cpgpt_grimage3_dependencies/input_scaler_mean_all_datasets.npy",
    "supporting_files/cpgpt_grimage3_dependencies/input_scaler_scale.npy",
    "supporting_files/cpgpt_grimage3_dependencies/input_scaler_scale_all_datasets.npy",
    "supporting_files/cpgpt_grimage3_dependencies/pca_scaler_mean.npy",
    "supporting_files/cpgpt_grimage3_dependencies/pca_scaler_scale.npy",
    "supporting_files/cpgpt_grimage3_dependencies/reliable/cpgpt_grimage3_weights_all_datasets_reliable.csv",
    "supporting_files/cpgpt_grimage3_dependencies/reliable/input_scaler_mean_all_datasets_reliable.npy",
    "supporting_files/cpgpt_grimage3_dependencies/reliable/input_scaler_scale_all_datasets_reliable.npy",
}
excluded = {
    "cpgptgrimage3_before15-12-2025.pt",
    "altumage_data.pkl",
    "supporting_files/cpgpt_grimage3_dependencies/reliable/cpgptgrimage3_reliable.ipynb",
    "supporting_files/cpgpt_grimage3_dependencies/reliable/cpgptgrimage3_reliable.pt",
}

assert len(metadata) == 173
assert remote_weights == expected_weights, (expected_weights - remote_weights, remote_weights - expected_weights)
assert "all_clock_metadata.pt" in files
assert required_static <= files, required_static - files
assert excluded.isdisjoint(files), excluded & files
print(f"verified {len(remote_weights)} weights, metadata, and {len(required_static)} static dependencies")
PY
```

Expected: `verified 173 weights, metadata, and 26 static dependencies`.

- [ ] **Step 2: Verify local sources against remote HF checksums**

Run:

```bash
uv run hf cache verify lucascamillomd/pyaging-data --local-dir clocks/weights --fail-on-extra-files
uv run hf cache verify lucascamillomd/pyaging-data --local-dir clocks/metadata --fail-on-extra-files
uv run hf cache verify lucascamillomd/pyaging-data --local-dir hf_static_data/repo --fail-on-extra-files
```

Expected: every local source file in all three groups verifies against HF. Do not use `--fail-on-missing-files` because each local directory intentionally represents only one subset of the remote repository.

- [ ] **Step 3: Run the opt-in anonymous online smoke test**

Run:

```bash
HF_HOME="$(mktemp -d)" HF_TOKEN= uv run pytest tests/integration/test_hf_smoke.py -m online -o addopts='' -v
```

Expected: metadata, `horvath2013.pt`, and the small blood-chemistry example download anonymously and validate.

- [ ] **Step 4: Run the one-time full 173-clock validation**

Run:

```bash
HF_HOME="$(mktemp -d)" HF_TOKEN= uv run pytest tests/predict/test_gold_standard.py -m full_catalog -o addopts='' -v
```

Expected: all 173 gold-standard predictions pass. This is the deliberate one-time ~25 GiB validation; do not fan it out across CI jobs.

- [ ] **Step 5: Record the verified HF revision**

Run:

```bash
uv run hf models info lucascamillomd/pyaging-data --format json | jq '{id, sha, lastModified, private}' | tee /tmp/pyaging-hf-verified-revision.json
```

Expected: public repository metadata and an audit file containing the validated revision.

- [ ] **Step 6: Revoke the broad HF upload token if it has no other owner-approved use**

The current stored token is named `cpgpt-upload-write` and is not fine-grained. In Hugging Face token settings, revoke it if the user confirms it is not needed for another repository. Keep the new `lucascamillomd/pyaging-data`-scoped token active, then run:

```bash
uv run hf auth whoami --format json | jq '{name, role: .auth.accessToken.role, fineGrained: .auth.accessToken.fineGrained}'
```

Expected: `lucascamillomd`, write permission, and a non-null fine-grained restriction. Do not revoke an owner token whose separate use has not been confirmed.

## Task 11: Prepare version 0.3.0 and safe CI/release workflows

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `tests/test_version.py`
- Modify: `tests/test_release_configuration.py`
- Modify: `.github/workflows/release.yml`
- Modify: `pyaging/__init__.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Remove: `.github/workflows/build.yml`
- Remove: `.github/workflows/publish.yml`
- Remove: `.github/workflows/test.yml`

- [ ] **Step 1: Write failing version and workflow tests**

Create `tests/test_version.py`:

```python
from importlib.metadata import version

import pyaging


def test_package_version_is_0_3_0_everywhere():
    assert pyaging.__version__ == "0.3.0"
    assert version("pyaging") == "0.3.0"
```

Append to `tests/test_release_configuration.py`:

```python
def test_release_workflow_is_tag_gated_and_publishes_after_verify():
    workflow = Path(".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "tags:" in workflow
    assert "- 'v*'" in workflow
    assert "needs: verify" in workflow
    assert "pypa/gh-action-pypi-publish" in workflow
    assert "gh release create" in workflow


def test_ci_excludes_large_and_online_tests():
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "not full_catalog and not online" in workflow
    assert "pull_request:" in workflow
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
uv run pytest tests/test_version.py tests/test_release_configuration.py -v
```

Expected: version remains `0.2.0`; CI/release workflow assertions fail.

- [ ] **Step 3: Set version 0.3.0 consistently**

Change:

```toml
# pyproject.toml
version = "0.3.0"
```

```python
# pyaging/__init__.py
__version__ = "0.3.0"
```

Run `uv lock` after both edits.

- [ ] **Step 4: Create hermetic PR/main CI**

Create `.github/workflows/ci.yml`:

```yaml
name: ci

on:
  push:
    branches: [main]
  pull_request:

jobs:
  unit:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ["3.9", "3.10", "3.11", "3.12", "3.13"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: python -m pip install uv
      - run: uv sync --quiet
      - run: uv run pytest -m "not full_catalog and not online" tests docs/source/test_make_clock_data.py

  tutorials:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: python -m pip install uv
      - run: uv sync --quiet
      - run: uv run pytest --nbmake tutorials/ --ignore=tutorials/tutorial_cpgptgrimage3.ipynb
```

- [ ] **Step 5: Replace the chained workflows with one tag-gated release workflow**

Replace `.github/workflows/release.yml` with:

```yaml
name: release

on:
  push:
    tags:
      - 'v*'

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: python -m pip install uv
      - run: uv sync --quiet
      - name: Verify tag and package version
        run: |
          test "${GITHUB_REF_NAME}" = "v$(uv run python -c 'import pyaging; print(pyaging.__version__)')"
      - run: uv run pytest -m "not full_catalog and not online" tests docs/source/test_make_clock_data.py
      - run: uv build
      - run: uvx twine check dist/*
      - uses: actions/upload-artifact@v4
        with:
          name: distributions
          path: dist/*

  publish:
    needs: verify
    runs-on: ubuntu-latest
    permissions:
      id-token: write
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: distributions
          path: dist
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}

  github-release:
    needs: publish
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: distributions
          path: dist
      - name: Create GitHub release
        env:
          GH_TOKEN: ${{ github.token }}
        run: gh release create "${GITHUB_REF_NAME}" dist/* --repo "${GITHUB_REPOSITORY}" --verify-tag --generate-notes
```

Delete `.github/workflows/build.yml`, `.github/workflows/publish.yml`, and `.github/workflows/test.yml` so no older workflow can publish from an unverified `workflow_run`.

- [ ] **Step 6: Run version, workflow, and hermetic tests**

Run:

```bash
uv run pytest tests/test_version.py tests/test_release_configuration.py tests/test_no_aws_dependencies.py -v
uv run pytest tests docs/source/test_make_clock_data.py -m 'not full_catalog and not online' -q
uv run ruff check pyaging/utils/_hf.py docs/source/make_clock_data.py docs/source/test_make_clock_data.py tests --exclude tests/predict/test_gold_standard.py
git diff --check
```

Expected: all tests and lint pass.

- [ ] **Step 7: Commit the 0.3.0 release preparation**

```bash
git add pyaging/__init__.py pyproject.toml uv.lock .github/workflows tests/test_version.py tests/test_release_configuration.py
git commit -m "chore: prepare pyaging 0.3.0 release"
```

## Task 12: Final validation, merge, tag, publish, and clean-install verification

**Files:**
- No planned source edits after the release commit.
- External writes: GitHub branch/PR/main/tag/release and PyPI `0.3.0`.

- [ ] **Step 1: Confirm PyPI version remains unused immediately before release**

Run:

```bash
test "$(curl -sS -o /dev/null -w '%{http_code}' https://pypi.org/pypi/pyaging/0.3.0/json)" = "404"
```

Expected: success. If PyPI returns anything other than 404, stop and inspect the existing release.

- [ ] **Step 2: Run the complete pre-release suite and inspect artifacts**

Run:

```bash
git status --short
uv run pytest tests docs/source/test_make_clock_data.py -m 'not full_catalog and not online' -v
uv run ruff check pyaging/utils/_hf.py docs/source/make_clock_data.py docs/source/test_make_clock_data.py tests --exclude tests/predict/test_gold_standard.py
git diff --check
uv run sphinx-build -M html docs/source docs/_build
release_dist=$(mktemp -d)
uv build --out-dir "$release_dist"
uvx twine check "$release_dist"/*
unzip -l "$release_dist"/pyaging-0.3.0-*.whl | sed -n '1,120p'
tar -tzf "$release_dist"/pyaging-0.3.0.tar.gz | sed -n '1,120p'
```

Expected: clean worktree, all checks pass, and both wheel/sdist contain the HF implementation with version `0.3.0`.

- [ ] **Step 3: Push the implementation branch and merge it through GitHub**

Run from the implementation worktree:

```bash
gh secret list | rg '^PYPI_API_TOKEN\b'
git push -u origin codex/hf-data-migration
gh pr create --base main --head codex/hf-data-migration --title "Migrate pyaging data hosting to Hugging Face" --body "Moves runtime and notebook data dependencies to lucascamillomd/pyaging-data, adds guarded HF publishing, and prepares pyaging 0.3.0."
gh pr checks --watch
gh pr merge --merge --delete-branch
```

Expected: CI passes and the PR merges. Do not tag an unmerged branch.

- [ ] **Step 4: Update local main and create the immutable release tag**

From the main checkout:

```bash
git pull --ff-only origin main
test "$(git status --porcelain)" = ""
test "$(uv run python -c 'import pyaging; print(pyaging.__version__)')" = "0.3.0"
git tag -a v0.3.0 -m "Release v0.3.0"
git push origin v0.3.0
```

Expected: tag push triggers the consolidated `release` workflow.

- [ ] **Step 5: Watch the release workflow through PyPI publication**

Run:

```bash
run_id=''
for attempt in {1..30}; do
  run_id=$(gh run list --workflow release.yml --limit 10 --json databaseId,headBranch --jq '[.[] | select(.headBranch == "v0.3.0")][0].databaseId // empty')
  test -n "$run_id" && break
  sleep 2
done
test -n "$run_id"
gh run watch "$run_id" --exit-status
```

Expected: verify, PyPI publish, and GitHub release jobs all succeed. If publication fails, inspect `gh run view "$run_id" --log-failed`; never retag or attempt to overwrite an existing PyPI `0.3.0`.

- [ ] **Step 6: Verify PyPI metadata and distributions**

Run:

```bash
for attempt in {1..30}; do
  response=$(curl -sS https://pypi.org/pypi/pyaging/0.3.0/json) && break
  sleep 2
done
test -n "${response:-}"
printf '%s' "$response" | jq '{version: .info.version, files: [.urls[].filename]}'
```

Expected: version `0.3.0` with both wheel and source distribution filenames.

- [ ] **Step 7: Perform the clean anonymous install and prediction smoke test**

Run:

```bash
release_env=$(mktemp -d)
uv venv "$release_env/venv" --python 3.11
uv pip install --refresh --python "$release_env/venv/bin/python" 'pyaging==0.3.0'
HF_HOME="$release_env/hf-home" HF_TOKEN= "$release_env/venv/bin/python" - <<'PY'
import tempfile

import pandas as pd
import pyaging as pya

assert pya.__version__ == "0.3.0"
data_dir = tempfile.mkdtemp(prefix="pyaging-030-")
logger = pya.logger.Logger("release-smoke")
clock = pya.pred.load_clock("horvath2013", "cpu", data_dir, logger)
frame = pd.DataFrame([[0.5] * len(clock.features)], columns=clock.features)
adata = pya.pp.df_to_adata(frame, imputer_strategy="constant", verbose=False)
pya.pred.predict_age(adata, "horvath2013", dir=data_dir, verbose=False)
assert "horvath2013" in adata.obs
print("verified pyaging", pya.__version__, "prediction", float(adata.obs["horvath2013"].iloc[0]))
PY
```

Expected: the package installs from PyPI, downloads the clock anonymously from HF, and prints a finite prediction.

- [ ] **Step 8: Confirm final release state**

Run:

```bash
gh release view v0.3.0
git ls-remote --tags origin v0.3.0
curl -fsS https://pypi.org/pypi/pyaging/json | jq -r '.info.version'
```

Expected: GitHub release/tag exist and PyPI latest is `0.3.0`.

- [ ] **Step 9: Hand off the manual legacy-S3 retirement decision**

Report the verified HF revision, PyPI release, and the fact that old package versions or copied S3 URLs can still generate AWS egress. Do not delete objects or change bucket access automatically. Ask for separate explicit approval before disabling public S3 access, and include the recommended sequence: observe remaining S3 requests, choose a transition date, notify users, disable public reads, and confirm AWS egress falls to zero.
