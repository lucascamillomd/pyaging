# tAge Clocks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the two flagship transcriptomic clocks from Tyshkovskiy et al. (Nature 2026) — `tage` (chronological age) and `tagemortality` (log10 mortality hazard) — with a public `pa.pp.prepare_tage()` cohort-preprocessing function.

**Architecture:** A new preprocessing module performs the entire cohort-level tAge pipeline (ortholog mapping to mouse Entrez, RLE normalization, log transform, per-gene scaling, reference-group centering) once on the full AnnData; the clocks themselves are plain linear models that flow through the standard predict path. A `required_uns_flag` guard on the model base class stops raw counts from silently reaching a centered clock. Ground truth for tests is generated once from the authors' reference R package + sklearn pickles and committed as fixtures.

**Tech Stack:** Python (torch, anndata, numpy, pandas), scikit-learn + joblib (conversion notebooks only), R (fixture generation only, authors' `Gladyshev-Lab/tAge` package).

**Spec:** `docs/superpowers/specs/2026-08-21-tage-clocks-design.md`

## Global Constraints

- Clock names are exactly `tage` and `tagemortality` (lowercase, no underscores, matching pyaging convention).
- Only the Elastic Net, Multispecies Multi-Tissue, `scaleddiff` (Scaled) variants are converted. Zenodo files: `EN_Chronoage_Multispecies_Multitissue_scaleddiff.pkl` and `EN_Mortality_Multispecies_Multitissue_scaleddiff.pkl` from record 18763485.
- Both clocks set `metadata["research_only"] = True` (MGB Open Access License 1.0, non-commercial academic use only) and cite DOI `https://doi.org/10.1038/s41586-026-10542-3`.
- All new parity assertions use tolerance `1e-6` (tighten in-notebook if the data allows; never loosen without recording why in the notebook).
- Cohort-level math (RLE size factors, gene scaling, centering) happens only inside `prepare_tage`, never inside `model.preprocess` — predictions must be independent of `batch_size`.
- Committed test fixtures must stay under the pre-commit large-file limit; gzip them and subset **samples** (never genes) if needed, regenerating expected outputs on the subset.
- Follow repo TDD convention: failing test → run → implement → run → commit. Pre-commit runs ruff; keep files clean.
- The scratchpad for downloads/clones is `/private/tmp/claude-501/-Users-lucascamillo-pyaging/d69cf7a7-b1bb-4bb1-8e32-762894e62622/scratchpad` (referred to as `$SCRATCH` below).

---

### Task 1: Reference fixtures from the authors' pipeline

Generate committed ground-truth fixtures by running the authors' R preprocessing and sklearn models on their own bundled example data. Every later numeric task tests against these files.

**Files:**
- Create: `clocks/generate_tage_fixtures.R` (run once; output committed)
- Create: `clocks/generate_tage_fixtures.py` (run once; output committed)
- Create: `tests/data/tage/` fixtures (committed): `input_expression.csv.gz`, `input_metadata.csv`, `after_rle.csv.gz`, `after_log.csv.gz`, `after_scale.csv.gz`, `after_center_all.csv.gz`, `after_center_refgroup.csv.gz`, `expected_predictions.json`, `README.md`

**Interfaces:**
- Produces: the fixture files above. `expected_predictions.json` schema:
  `{"units_tage": "<recorded unit>", "sample_ids": [...], "tage_center_all": [...], "tage_center_refgroup": [...], "tagemortality_center_all": [...], "tagemortality_center_refgroup": [...], "reference_group_sample_ids": [...]}`

- [ ] **Step 1: Clone the reference repo and inspect the example data**

```bash
git clone --depth 1 https://github.com/Gladyshev-Lab/tAge "$SCRATCH/tAge"
head -3 "$SCRATCH/tAge/inst/extdata/Exprs_example.csv" | cut -c1-300
head -5 "$SCRATCH/tAge/inst/extdata/Metadata_example.csv"
wc -c "$SCRATCH/tAge/inst/extdata/Exprs_example.csv"
```

Record: gene ID type of the example rows, the metadata columns (age, species, tissue, any control/condition column usable as a reference group), and the file size.

- [ ] **Step 2: Write `clocks/generate_tage_fixtures.R`**

The script must (reading `R/preprocessing.R`, `R/genes.R`, `R/predict.R` in the clone for the exact exported function names and arguments — adjust the calls below to what the package actually exports, keeping the stage order fixed):

```r
#!/usr/bin/env Rscript
# Generate tAge parity fixtures from the authors' reference implementation.
# Run from the repo root:  Rscript clocks/generate_tage_fixtures.R <path-to-tAge-clone> tests/data/tage
args <- commandArgs(trailingOnly = TRUE)
tage_dir <- args[1]; out_dir <- args[2]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
devtools::load_all(tage_dir)   # or library(tAge) after install
exprs <- read.csv(file.path(tage_dir, "inst/extdata/Exprs_example.csv"), row.names = 1, check.names = FALSE)
meta  <- read.csv(file.path(tage_dir, "inst/extdata/Metadata_example.csv"), row.names = 1)

dump <- function(m, name) write.csv(as.data.frame(m), file.path(out_dir, paste0(name, ".csv")))

# Stage-wise export, mirroring tAge_preprocessing() for the scaleddiff space.
# (Exact function names/signatures come from the package; keep this order.)
mapped   <- <gene-mapping step if the example data is not already mouse Entrez>
rle      <- RLE_normalization(mapped);            dump(rle,    "after_rle")
logged   <- log_transform(rle);                   dump(logged, "after_log")
scaled   <- scale_eset(logged);                   dump(scaled, "after_scale")
cent_all <- <centering with no reference group>;  dump(cent_all, "after_center_all")
ref_ids  <- <sample ids of the chosen reference group from meta>
cent_ref <- <centering against ref_ids>;          dump(cent_ref, "after_center_refgroup")
writeLines(ref_ids, file.path(out_dir, "reference_group_sample_ids.txt"))
file.copy(file.path(tage_dir, "inst/extdata/Exprs_example.csv"),   file.path(out_dir, "input_expression.csv"))
file.copy(file.path(tage_dir, "inst/extdata/Metadata_example.csv"), file.path(out_dir, "input_metadata.csv"))
```

The `<...>` slots are resolved by reading the cloned R source while writing the script — they are lookups, not design decisions. If the example expression matrix is not already in mouse Entrez space, also dump the post-mapping matrix as `after_mapping.csv`.

- [ ] **Step 3: Run the R script**

```bash
Rscript clocks/generate_tage_fixtures.R "$SCRATCH/tAge" tests/data/tage
```

Expected: the stage CSVs exist and are finite numbers. If R deps are missing, install into a local library as `clocks/notebooks/kdmage.ipynb`'s embedded R does.

- [ ] **Step 4: Write `clocks/generate_tage_fixtures.py`**

Downloads the two EN pickles, applies them to the R-preprocessed matrices with sklearn (the model-half ground truth), writes `expected_predictions.json`:

```python
#!/usr/bin/env python3
"""Generate expected tAge predictions from the published sklearn models.

Run after generate_tage_fixtures.R:  python clocks/generate_tage_fixtures.py tests/data/tage
"""
import json, sys, urllib.request
from pathlib import Path
import joblib
import pandas as pd

out = Path(sys.argv[1])
ZENODO = "https://zenodo.org/records/18763485/files/{}?download=1"
MODELS = {
    "tage": "EN_Chronoage_Multispecies_Multitissue_scaleddiff.pkl",
    "tagemortality": "EN_Mortality_Multispecies_Multitissue_scaleddiff.pkl",
}
preds = {}
for clock, filename in MODELS.items():
    local = out / filename
    if not local.exists():
        urllib.request.urlretrieve(ZENODO.format(filename), local)
    # joblib.load executes pickle bytecode; acceptable here because the files
    # come from the paper's official Zenodo record over HTTPS (fixed record id),
    # the same trust decision pyaging already makes for its torch.save'd clocks.
    est = joblib.load(local)
    # If the pickle is a Pipeline, unwrap: est = est[-1] — but then the earlier
    # steps are part of the model and MUST be reproduced in the conversion
    # notebook; record what was found in tests/data/tage/README.md.
    feature_names = list(getattr(est, "feature_names_in_", []))
    assert feature_names, "model must carry feature_names_in_; if absent, find the gene order in the tAge package and record it"
    for suffix in ["center_all", "center_refgroup"]:
        matrix = pd.read_csv(out / f"after_{suffix}.csv", index_col=0)
        # R dump: genes x samples or samples x genes — orient so columns are features
        if list(matrix.index[:5]) and set(feature_names) & set(matrix.index):
            matrix = matrix.T
        aligned = matrix.reindex(columns=feature_names, fill_value=0.0)
        preds[f"{clock}_{suffix}"] = est.predict(aligned.values).tolist()
        preds["sample_ids"] = aligned.index.tolist()
preds["reference_group_sample_ids"] = (out / "reference_group_sample_ids.txt").read_text().split()
preds["units_tage"] = "TODO-REPLACE"  # fill from Step 5 before committing
(out / "expected_predictions.json").write_text(json.dumps(preds, indent=1))
```

- [ ] **Step 5: Determine `tage` output units and record them**

Compare `tage_center_all` predictions against the known ages in `input_metadata.csv` (metadata ages for mouse examples are typically months). Set `units_tage` accordingly in `expected_predictions.json` (edit the script's constant and re-run — no hand-edited outputs). Write `tests/data/tage/README.md` stating: fixture provenance (repo commit hash of the clone, Zenodo record), the resolved R function names used per stage, matrix orientation, the pickle's object type, and the units conclusion.

- [ ] **Step 6: Compress and commit**

```bash
gzip -k9 tests/data/tage/input_expression.csv tests/data/tage/after_*.csv
rm tests/data/tage/after_*.csv tests/data/tage/input_expression.csv tests/data/tage/*.pkl
du -sh tests/data/tage
git add clocks/generate_tage_fixtures.R clocks/generate_tage_fixtures.py tests/data/tage
git commit -m "test: add tAge reference fixtures from the published pipeline"
```

If `du` shows any single file above the pre-commit large-file limit, subset samples in the R script (keep every reference-group sample plus enough others to exercise both centerings), re-run both scripts, and note the subset in the README.

---

### Task 2: `required_uns_flag` guard on the predict path

**Files:**
- Modify: `src/pyaging/models/_base_models.py` (in `pyagingModel.__init__`)
- Modify: `src/pyaging/predict/_pred.py` (clock loop, after `load_clock`)
- Test: `tests/predict/test_required_uns_flag.py`

**Interfaces:**
- Produces: `pyagingModel.required_uns_flag: str | None` (default `None`). When set, `predict_age` raises `ValueError` unless `adata.uns.get(flag)` is truthy. Old saved models lack the attribute — the predict path must use `getattr(model, "required_uns_flag", None)`.

- [ ] **Step 1: Write the failing test**

```python
import anndata
import numpy as np
import pytest
import torch

import pyaging as pya
from pyaging.models._base_models import pyagingModel


class _GuardedClock(pyagingModel):
    def __init__(self):
        super().__init__()
        self.required_uns_flag = "tage_prepared"

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


def _minimal_adata():
    return anndata.AnnData(X=np.zeros((3, 2)), var={"var_names": ["g1", "g2"]})


def test_base_model_defaults_to_no_flag():
    class _Plain(pyagingModel):
        def preprocess(self, x):
            return x

        def postprocess(self, x):
            return x

    assert _Plain().required_uns_flag is None


def test_guard_raises_without_uns_marker(monkeypatch):
    model = _GuardedClock()
    model.metadata["clock_name"] = "guarded"
    model.features = ["g1", "g2"]
    model.base_model = torch.nn.Identity()
    monkeypatch.setattr(
        "pyaging.predict._pred.load_clock", lambda *a, **k: model
    )
    adata = _minimal_adata()
    with pytest.raises(ValueError, match="prepare_tage"):
        pya.pred.predict_age(adata, ["guarded"], verbose=False)


def test_guard_passes_with_uns_marker(monkeypatch):
    model = _GuardedClock()
    model.metadata["clock_name"] = "guarded"
    model.features = ["g1", "g2"]
    model.base_model = torch.nn.Linear(2, 1).double()
    monkeypatch.setattr(
        "pyaging.predict._pred.load_clock", lambda *a, **k: model
    )
    adata = _minimal_adata()
    adata.uns["tage_prepared"] = True
    pya.pred.predict_age(adata, ["guarded"], verbose=False)
    assert "guarded" in adata.obs.columns
```

Adjust the monkeypatch target and the `predict_age` signature details to the real module layout if they differ (check how `_pred.py` imports `load_clock`); the assertion behavior is the contract. If `add_pred_ages_and_clock_metadata_adata` requires more model attributes for the passing test, set them on the stub rather than weakening the assertion.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/predict/test_required_uns_flag.py -v`
Expected: FAIL (`AttributeError: required_uns_flag` / no ValueError raised).

- [ ] **Step 3: Implement**

In `pyagingModel.__init__` (next to `self.reference_values = None`):

```python
# Name of an ``adata.uns`` key that must be truthy before this clock may
# run; cohort-relative clocks set it so raw inputs fail loudly.
self.required_uns_flag = None
```

In `_pred.py`, immediately after `model = load_clock(...)`:

```python
required_flag = getattr(model, "required_uns_flag", None)
if required_flag is not None and not adata.uns.get(required_flag, False):
    raise ValueError(
        f"Clock '{clock_name}' needs cohort-preprocessed input: run "
        f"pyaging.preprocess.prepare_tage(...) first "
        f"(adata.uns['{required_flag}'] is missing)."
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/predict/test_required_uns_flag.py tests/predict -v`
Expected: new tests PASS; no regressions in the rest of `tests/predict`.

- [ ] **Step 5: Commit**

```bash
git add src/pyaging/models/_base_models.py src/pyaging/predict/_pred.py tests/predict/test_required_uns_flag.py
git commit -m "feat(predict): let clocks require a cohort-preprocessing marker"
```

---

### Task 3: Gene-mapping asset and `_map_to_mouse_entrez`

**Files:**
- Create: `clocks/build_tage_gene_mapping.py` (run once; output uploaded to HF with the weights in Task 7)
- Create: `src/pyaging/preprocess/_tage.py` (started here, extended in Tasks 4–5)
- Test: `tests/preprocess/test_tage_mapping.py`

**Interfaces:**
- Produces: `tage_gene_mapping.csv.gz` with columns `species` (mouse|rat|macaque|human), `source_id` (upper-cased symbol / Ensembl / Entrez string), `source_type` (symbol|ensembl|entrez), `mouse_entrez` (string).
- Produces: `_map_to_mouse_entrez(df: pd.DataFrame, species: str, mapping: pd.DataFrame) -> pd.DataFrame` — samples×genes in, samples×mouse-Entrez out; drops unmapped columns; many-to-one collapse mirrors the reference (`R/genes.R` in the Task 1 clone is the authority — sum for counts is expected; record what it does in the module docstring).

- [ ] **Step 1: Write `clocks/build_tage_gene_mapping.py`**

```python
#!/usr/bin/env python3
"""Build the tAge gene/ortholog mapping asset from the reference repo tables.

Usage: python clocks/build_tage_gene_mapping.py <path-to-tAge-clone> <out.csv.gz>
Sources: inst/extdata/metadata/Gene_table_{human,monkey,mouse,rat}.csv and
Table_of_orthologs.csv. Inspect each table's columns first and adapt the
melt below; the output contract is the four-column frame documented above.
"""
import sys
from pathlib import Path
import pandas as pd

clone, out = Path(sys.argv[1]), Path(sys.argv[2])
meta = clone / "inst/extdata/metadata"
frames = []
for fname, species in [
    ("Gene_table_mouse.csv", "mouse"),
    ("Gene_table_rat.csv", "rat"),
    ("Gene_table_monkey.csv", "macaque"),
    ("Gene_table_human.csv", "human"),
]:
    table = pd.read_csv(meta / fname, dtype=str)
    # Expected shape: per-species gene table with symbol/Ensembl/Entrez columns;
    # join non-mouse species to mouse Entrez through Table_of_orthologs.csv.
    ...
mapping = pd.concat(frames, ignore_index=True).dropna(subset=["mouse_entrez"])
mapping["source_id"] = mapping["source_id"].str.upper()
mapping.drop_duplicates(subset=["species", "source_id", "source_type"], keep="first", inplace=True)
mapping.to_csv(out, index=False, compression="gzip")
print(len(mapping), "rows ->", out)
```

The `...` body is a straight reshape of whatever columns the tables actually carry (look at them first: `head -2` each file); no inference beyond joining orthologs. Run it and sanity-check: every one of the two clocks' features (mouse Entrez, from Task 6) must appear as `species=mouse, source_type=entrez` rows mapping to themselves — add that assertion to the script once Task 6 exists, or assert non-empty per species now.

- [ ] **Step 2: Write the failing tests for `_map_to_mouse_entrez`**

```python
import pandas as pd
import pytest

from pyaging.preprocess._tage import _map_to_mouse_entrez

MAPPING = pd.DataFrame(
    {
        "species": ["mouse", "mouse", "human", "human", "human"],
        "source_id": ["CDKN1A", "12575", "CDKN1A", "ENSG00000124762", "LGALS3"],
        "source_type": ["symbol", "entrez", "symbol", "ensembl", "symbol"],
        "mouse_entrez": ["12575", "12575", "12575", "12575", "16854"],
    }
)


def _frame(columns, rows=2):
    return pd.DataFrame([[float(i + j) for j in range(len(columns))] for i in range(rows)], columns=columns)


def test_mouse_symbols_map_to_entrez():
    out = _map_to_mouse_entrez(_frame(["Cdkn1a"]), "mouse", MAPPING)
    assert list(out.columns) == ["12575"]


def test_human_ensembl_maps_via_orthologs():
    out = _map_to_mouse_entrez(_frame(["ENSG00000124762"]), "human", MAPPING)
    assert list(out.columns) == ["12575"]


def test_unmapped_genes_dropped():
    out = _map_to_mouse_entrez(_frame(["LGALS3", "NOT_A_GENE"]), "human", MAPPING)
    assert list(out.columns) == ["16854"]


def test_many_to_one_collapses_to_single_column():
    out = _map_to_mouse_entrez(_frame(["CDKN1A", "ENSG00000124762"]), "human", MAPPING)
    assert list(out.columns) == ["12575"]
    assert out.shape[1] == 1


def test_unknown_species_raises():
    with pytest.raises(ValueError, match="species"):
        _map_to_mouse_entrez(_frame(["Cdkn1a"]), "dog", MAPPING)
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/preprocess/test_tage_mapping.py -v`
Expected: FAIL with import error.

- [ ] **Step 4: Implement in `src/pyaging/preprocess/_tage.py`**

```python
"""Cohort preprocessing for the tAge transcriptomic clocks (Tyshkovskiy 2026)."""

import numpy as np
import pandas as pd

TAGE_SPECIES = ("mouse", "rat", "macaque", "human")


def _map_to_mouse_entrez(df: pd.DataFrame, species: str, mapping: pd.DataFrame) -> pd.DataFrame:
    if species not in TAGE_SPECIES:
        raise ValueError(f"species must be one of {TAGE_SPECIES}, got {species!r}")
    rows = mapping[mapping["species"] == species]
    lookup = dict(zip(rows["source_id"], rows["mouse_entrez"]))
    targets = df.columns.str.upper().map(lookup)
    kept = targets.notna()
    mapped = df.loc[:, kept].copy()
    mapped.columns = targets[kept]
    # Many-to-one: collapse the same way the reference package does (R/genes.R).
    return mapped.T.groupby(level=0).sum().T
```

If the Task 1 clone's `R/genes.R` collapses by mean or first-match instead of sum, mirror that and update `test_many_to_one_collapses_to_single_column` to assert the summed/averaged value explicitly.

- [ ] **Step 5: Run tests, then commit**

Run: `uv run pytest tests/preprocess/test_tage_mapping.py -v` — expected PASS.

```bash
git add clocks/build_tage_gene_mapping.py src/pyaging/preprocess/_tage.py tests/preprocess/test_tage_mapping.py
git commit -m "feat(preprocess): add tAge gene/ortholog mapping"
```

---

### Task 4: RLE, log, scale, and centering transforms

**Files:**
- Modify: `src/pyaging/preprocess/_tage.py`
- Test: `tests/preprocess/test_tage_transforms.py`

**Interfaces:**
- Consumes: Task 1 fixtures in `tests/data/tage/`.
- Produces (all samples×genes `pd.DataFrame` → same shape):
  `_rle_normalize(df)`, `_log_transform(df)`, `_scale_genes(df)`, `_center_against_reference(df, reference_index=None)` where `reference_index` is a list of sample labels (`None` → per-gene median over all samples).

- [ ] **Step 1: Write the failing fixture-parity tests**

```python
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyaging.preprocess._tage import (
    _center_against_reference,
    _log_transform,
    _rle_normalize,
    _scale_genes,
)

DATA = Path(__file__).parents[1] / "data" / "tage"
TOL = 1e-6


def _load(name):
    # Fixtures are R dumps; orient to samples x genes per tests/data/tage/README.md.
    frame = pd.read_csv(DATA / f"{name}.csv.gz", index_col=0)
    return frame.T if frame.shape[0] > frame.shape[1] else frame  # adjust per README


@pytest.fixture(scope="module")
def stages():
    return {
        name: _load(name)
        for name in ["input_expression", "after_rle", "after_log", "after_scale", "after_center_all", "after_center_refgroup"]
    }


def test_rle_matches_reference(stages):
    ours = _rle_normalize(stages["input_expression"])
    np.testing.assert_allclose(ours.values, stages["after_rle"].values, atol=TOL)


def test_log_matches_reference(stages):
    ours = _log_transform(stages["after_rle"])
    np.testing.assert_allclose(ours.values, stages["after_log"].values, atol=TOL)


def test_scale_matches_reference(stages):
    ours = _scale_genes(stages["after_log"])
    np.testing.assert_allclose(ours.values, stages["after_scale"].values, atol=TOL)


def test_center_all_matches_reference(stages):
    ours = _center_against_reference(stages["after_scale"])
    np.testing.assert_allclose(ours.values, stages["after_center_all"].values, atol=TOL)


def test_center_refgroup_matches_reference(stages):
    ref_ids = json.loads((DATA / "expected_predictions.json").read_text())["reference_group_sample_ids"]
    ours = _center_against_reference(stages["after_scale"], reference_index=ref_ids)
    np.testing.assert_allclose(ours.values, stages["after_center_refgroup"].values, atol=TOL)
```

If the example matrix arrives pre-mapped (`after_mapping.csv.gz` exists), chain from that instead of `input_expression`. Resolve the orientation heuristic against the README and hard-code it.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/preprocess/test_tage_transforms.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement, porting the R source exactly**

Read `$SCRATCH/tAge/R/preprocessing.R` (re-clone if the scratchpad was cleared) and port each function. Expected shapes (verify against the R source; parity tests are the arbiter):

```python
def _rle_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """DESeq-style median-of-ratios; per-sample size factors from genes finite in the log-mean."""
    log_means = np.log(df.replace(0, np.nan)).mean(axis=0)
    usable = np.isfinite(log_means)
    ratios = np.log(df.loc[:, usable].replace(0, np.nan)).sub(log_means[usable], axis=1)
    size_factors = np.exp(ratios.median(axis=1))
    return df.div(size_factors, axis=0)


def _log_transform(df: pd.DataFrame) -> pd.DataFrame:
    return np.log(df + 1.0)


def _scale_genes(df: pd.DataFrame) -> pd.DataFrame:
    std = df.std(axis=0, ddof=1).replace(0, np.nan)
    return df.sub(df.mean(axis=0), axis=1).div(std, axis=1).fillna(0.0)


def _center_against_reference(df: pd.DataFrame, reference_index=None) -> pd.DataFrame:
    reference = df if reference_index is None else df.loc[reference_index]
    if reference.empty:
        raise ValueError("reference_group selects no samples")
    return df.sub(reference.median(axis=0), axis=1)
```

Each function that fails parity gets corrected against the R source (base/`log2` vs `log1p`, ddof, NaN policy, zero-handling) until the fixture test passes at 1e-6. Do not adjust fixtures to fit the code.

- [ ] **Step 4: Run tests, then commit**

Run: `uv run pytest tests/preprocess/test_tage_transforms.py -v` — expected PASS.

```bash
git add src/pyaging/preprocess/_tage.py tests/preprocess/test_tage_transforms.py
git commit -m "feat(preprocess): port the tAge cohort transforms with reference parity"
```

---

### Task 5: `prepare_tage()` public function

**Files:**
- Modify: `src/pyaging/preprocess/_tage.py`
- Modify: `src/pyaging/preprocess/__init__.py` (export `prepare_tage`; add to `__all__`)
- Test: `tests/preprocess/test_prepare_tage.py`

**Interfaces:**
- Consumes: Tasks 3–4 helpers; `pyaging.utils._hf.download_hf_file(filename, repo_id=...)`.
- Produces:

```python
def prepare_tage(
    adata: anndata.AnnData,
    species: str,
    reference_group: "list[str] | np.ndarray | None" = None,
    dir: str = "pyaging_data",
) -> anndata.AnnData
```

Returns a new AnnData: `X` = centered scaleddiff matrix (float64), `var_names` = mouse Entrez strings, `obs` copied from the input, `uns["tage_prepared"] = True`, and `uns["tage_preparation"] = {"species", "n_input_genes", "n_mapped_genes", "n_reference_samples", "reference_group": [...] or "all_samples"}`. `reference_group` accepts obs names (list) or a boolean mask aligned to `adata.obs_names`. The mapping table is fetched via `download_hf_file("tage_gene_mapping.csv.gz", repo_id="pyaging/tage")` behind a small module-level loader that tests monkeypatch.

- [ ] **Step 1: Write the failing tests**

```python
import anndata
import numpy as np
import pandas as pd
import pytest

import pyaging as pya
from pyaging.preprocess import _tage

MAPPING = pd.DataFrame(
    {
        "species": ["mouse"] * 4,
        "source_id": ["G1", "G2", "G3", "G4"],
        "source_type": ["symbol"] * 4,
        "mouse_entrez": ["101", "102", "103", "104"],
    }
)


@pytest.fixture(autouse=True)
def _local_mapping(monkeypatch):
    monkeypatch.setattr(_tage, "_load_mapping", lambda dir: MAPPING)


def _adata(n_obs=4):
    rng = np.random.default_rng(0)
    X = rng.integers(1, 1000, size=(n_obs, 4)).astype(float)
    a = anndata.AnnData(X=X)
    a.var_names = ["G1", "G2", "G3", "G4"]
    a.obs_names = [f"s{i}" for i in range(n_obs)]
    return a


def test_prepare_stamps_uns_and_maps_names():
    out = pya.pp.prepare_tage(_adata(), species="mouse")
    assert out.uns["tage_prepared"] is True
    assert list(out.var_names) == ["101", "102", "103", "104"]
    assert out.uns["tage_preparation"]["reference_group"] == "all_samples"


def test_prepare_pipeline_matches_composed_helpers():
    a = _adata()
    out = pya.pp.prepare_tage(a, species="mouse")
    frame = pd.DataFrame(a.X, index=a.obs_names, columns=["101", "102", "103", "104"])
    expected = _tage._center_against_reference(
        _tage._scale_genes(_tage._log_transform(_tage._rle_normalize(frame)))
    )
    np.testing.assert_allclose(out.X, expected.values, atol=1e-12)


def test_reference_group_by_name_and_mask_agree():
    a = _adata()
    by_name = pya.pp.prepare_tage(a, species="mouse", reference_group=["s0", "s1"])
    mask = np.array([True, True, False, False])
    by_mask = pya.pp.prepare_tage(a, species="mouse", reference_group=mask)
    np.testing.assert_allclose(by_name.X, by_mask.X)
    assert by_name.uns["tage_preparation"]["n_reference_samples"] == 2


def test_single_sample_raises():
    with pytest.raises(ValueError, match="at least two samples"):
        pya.pp.prepare_tage(_adata(n_obs=1), species="mouse")


def test_empty_reference_group_raises():
    with pytest.raises(ValueError, match="reference_group"):
        pya.pp.prepare_tage(_adata(), species="mouse", reference_group=[])


def test_unknown_species_raises():
    with pytest.raises(ValueError, match="species"):
        pya.pp.prepare_tage(_adata(), species="ferret")


def test_public_export():
    assert "prepare_tage" in pya.pp.__all__
```

Also add a low-overlap warning test once the warning exists: input with 1 mappable + 5 unmappable genes triggers a logged warning (assert via the logger pattern used in `tests/preprocess`'s existing tests).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/preprocess/test_prepare_tage.py -v`
Expected: FAIL (no `prepare_tage`).

- [ ] **Step 3: Implement**

In `_tage.py`:

```python
import anndata

from ..utils._hf import download_hf_file


def _load_mapping(dir: str) -> pd.DataFrame:
    path = download_hf_file("tage_gene_mapping.csv.gz", dir, repo_id="pyaging/tage")
    return pd.read_csv(path, dtype=str)


def prepare_tage(adata, species, reference_group=None, dir="pyaging_data"):
    """Run the tAge cohort preprocessing (map -> RLE -> log -> scale -> center).

    The tAge clocks are cohort-relative: every step below uses statistics of
    the whole input, so predictions depend on which samples are processed
    together. A single sample cannot be prepared.
    """
    if adata.n_obs < 2:
        raise ValueError("prepare_tage needs at least two samples: the tAge clocks are cohort-relative")
    frame = pd.DataFrame(
        np.asarray(adata.X, dtype=np.float64), index=adata.obs_names, columns=adata.var_names
    )
    mapped = _map_to_mouse_entrez(frame, species, _load_mapping(dir))
    if mapped.shape[1] == 0:
        raise ValueError(f"no input genes could be mapped to mouse Entrez IDs for species {species!r}")
    reference_index = _resolve_reference_group(adata, reference_group)
    centered = _center_against_reference(
        _scale_genes(_log_transform(_rle_normalize(mapped))), reference_index
    )
    out = anndata.AnnData(X=centered.values, obs=adata.obs.copy())
    out.var_names = centered.columns
    out.obs_names = adata.obs_names
    out.uns["tage_prepared"] = True
    out.uns["tage_preparation"] = {
        "species": species,
        "n_input_genes": int(frame.shape[1]),
        "n_mapped_genes": int(mapped.shape[1]),
        "n_reference_samples": len(reference_index) if reference_index is not None else int(adata.n_obs),
        "reference_group": list(reference_index) if reference_index is not None else "all_samples",
    }
    return out


def _resolve_reference_group(adata, reference_group):
    if reference_group is None:
        return None
    reference = np.asarray(reference_group)
    if reference.dtype == bool:
        if reference.shape[0] != adata.n_obs:
            raise ValueError("boolean reference_group must have one entry per sample")
        names = adata.obs_names[reference]
    else:
        missing = set(map(str, reference)) - set(adata.obs_names)
        if missing:
            raise ValueError(f"reference_group names not in adata.obs_names: {sorted(missing)[:5]}")
        names = pd.Index(reference, dtype=object)
    if len(names) == 0:
        raise ValueError("reference_group selects no samples")
    return list(names)
```

Export from `preprocess/__init__.py` (`from ._tage import prepare_tage`, add to `__all__`). Add the low-overlap warning (mapped fraction below 0.5 of input genes → `logger.warning`) using the module's existing logging pattern from `_preprocess.py`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/preprocess -v` — new tests PASS, no regressions. Also `uv run pytest tests/test_public_api.py -v` (the export list test likely enumerates `pa.pp` — update its expectation).

- [ ] **Step 5: Commit**

```bash
git add src/pyaging/preprocess/_tage.py src/pyaging/preprocess/__init__.py tests/preprocess/test_prepare_tage.py tests/test_public_api.py
git commit -m "feat(preprocess): add prepare_tage cohort preprocessing"
```

---

### Task 6: Model classes and feature-range modality

**Files:**
- Modify: `src/pyaging/models/_models.py` (append `TAge`, `TAgeMortality`)
- Modify: `src/pyaging/data/feature_ranges.json` (`modality_defaults`)
- Modify: `clocks/metadata/controlled_vocabulary.json` (allowed `data_type` values)
- Test: `tests/utils/test_feature_ranges.py` (extend) and `tests/predict/test_required_uns_flag.py` (extend)

**Interfaces:**
- Produces: `pya.models.TAge` and `pya.models.TAgeMortality` — identity pre/postprocess, `required_uns_flag = "tage_prepared"`; used by the Task 7 notebooks.
- Produces: modality `"transcriptomics (relative)"` with `{"unit": null, "low": null, "high": null}` — centered expression is legitimately negative and unbounded, so the range checker stays silent.

- [ ] **Step 1: Write the failing tests**

```python
# tests/utils/test_feature_ranges.py — append:
def test_relative_transcriptomics_is_unbounded():
    units, low, high = resolve_feature_bounds(["12575"], "transcriptomics (relative)")
    assert units == [None]
    assert low[0] == -math.inf and high[0] == math.inf


# tests/predict/test_required_uns_flag.py — append:
def test_tage_models_declare_the_guard():
    import pyaging as pya

    for cls in (pya.models.TAge, pya.models.TAgeMortality):
        model = cls()
        assert model.required_uns_flag == "tage_prepared"
        x = torch.ones(1, 3, dtype=torch.float64)
        assert torch.equal(model.preprocess(x), x)
        assert torch.equal(model.postprocess(x), x)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/utils/test_feature_ranges.py tests/predict/test_required_uns_flag.py -v`
Expected: FAIL (missing modality, missing classes).

- [ ] **Step 3: Implement**

Append to `_models.py` (matching the file's class style):

```python
class TAge(pyagingModel):
    def __init__(self):
        super().__init__()
        # Cohort-relative clock: input must come from preprocess.prepare_tage.
        self.required_uns_flag = "tage_prepared"

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class TAgeMortality(pyagingModel):
    def __init__(self):
        super().__init__()
        self.required_uns_flag = "tage_prepared"

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x
```

Add to `feature_ranges.json` `modality_defaults`:

```json
"transcriptomics (relative)": {"unit": null, "low": null, "high": null}
```

Add `"transcriptomics (relative)"` to the `data_type` list in `clocks/metadata/controlled_vocabulary.json`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/utils tests/predict tests/test_clock_metadata.py -v`
Expected: PASS (the metadata test validates the vocabulary file's shape).

- [ ] **Step 5: Commit**

```bash
git add src/pyaging/models/_models.py src/pyaging/data/feature_ranges.json clocks/metadata/controlled_vocabulary.json tests/utils/test_feature_ranges.py tests/predict/test_required_uns_flag.py
git commit -m "feat(models): add TAge and TAgeMortality with a relative-transcriptomics modality"
```

---

### Task 7: Conversion notebooks

**Files:**
- Create: `clocks/notebooks/tage.ipynb`
- Create: `clocks/notebooks/tagemortality.ipynb`

**Interfaces:**
- Consumes: `pya.models.TAge` / `TAgeMortality`, Task 1 fixtures (for in-notebook parity), `clocks/build_tage_gene_mapping.py` output.
- Produces: `clocks/weights/tage.pt`, `clocks/weights/tagemortality.pt`, and `tage_gene_mapping.csv.gz` staged for the HF upload in Task 9.

Both notebooks follow the section skeleton of `clocks/notebooks/kdmage.ipynb` (Index → Instantiate → Metadata → Download dependencies → Features → Weights → Reference values → Pre/postprocess names → Check parameters → Feature ranges → Basic test → Parity → Save). Write `tage.ipynb` fully, then copy and adapt for `tagemortality.ipynb` (different pickle, metadata notes, and expected-prediction key) — the steps below apply to each in turn.

- [ ] **Step 1: Metadata cell**

```python
model.metadata["clock_name"] = "tage"  # / "tagemortality"
model.metadata["data_type"] = "transcriptomics (relative)"
model.metadata["species"] = "multispecies"  # mouse, rat, macaque, human via mouse orthologs
model.metadata["year"] = 2026
model.metadata["approved_by_author"] = "⌛"
model.metadata["citation"] = (
    "Tyshkovskiy, Alexander, et al. \"Universal transcriptomic hallmarks of "
    "mammalian ageing and mortality.\" Nature 654 (2026): 173-188."
)
model.metadata["doi"] = "https://doi.org/10.1038/s41586-026-10542-3"
model.metadata["research_only"] = True  # MGB Open Access License 1.0: non-commercial academic use only
model.metadata["notes"] = (
    "Elastic Net, Multispecies Multi-Tissue, scaleddiff variant. Cohort-relative: "
    "inputs must come from pyaging.preprocess.prepare_tage, and predictions are "
    "relative to the chosen reference group. <units note from Task 1 README>"
)
```

Check `model.metadata["species"]` against what `clocks/metadata/controlled_vocabulary.json` allows for multi-species clocks (other multispecies clocks in `clock_metadata.json` are the precedent — use their exact value).

- [ ] **Step 2: Download and unpack the pickle**

```python
FILENAME = "EN_Chronoage_Multispecies_Multitissue_scaleddiff.pkl"  # Mortality file in the other notebook
url = f"https://zenodo.org/records/18763485/files/{FILENAME}?download=1"
# urllib download to the notebook cwd, then:
est = joblib.load(FILENAME)
print(type(est))
```

If `est` is a `Pipeline`, every pre-estimator step must be reproduced here or refuted as identity — print each step and resolve before continuing (Task 1's README recorded this). Extract:

```python
features = [str(f) for f in est.feature_names_in_]
coef = est.coef_.ravel()
intercept = float(est.intercept_)
assert len(features) == len(coef)
```

- [ ] **Step 3: Build the linear model**

```python
model.features = features
model.base_model = pya.models.LinearModel(input_dim=len(features))
model.base_model.linear.weight.data = torch.tensor(coef, dtype=torch.float64).unsqueeze(0)
model.base_model.linear.bias.data = torch.tensor([intercept], dtype=torch.float64)
model.reference_values = None  # missing genes fill with 0 = the centered no-change value
model.preprocess_name = None
model.preprocess_dependencies = None
model.postprocess_name = None
model.postprocess_dependencies = None
model.feature_units = None  # unitless centered expression; modality default applies
```

`reference_values = None` is deliberate: `check_features_in_adata` fills absent genes with 0, which in centered space means "at the reference median" — the least-biased imputation for this clock. State that in a notebook markdown cell.

- [ ] **Step 4: Parity against the sklearn ground truth**

```python
import gzip, json
fixtures = Path("../../tests/data/tage")
expected = json.loads((fixtures / "expected_predictions.json").read_text())
matrix = pd.read_csv(fixtures / "after_center_all.csv.gz", index_col=0)
# orient per tests/data/tage/README.md, then:
aligned = matrix.reindex(columns=model.features, fill_value=0.0)
model.eval(); model.to(torch.float64)
with torch.inference_mode():
    ours = model(torch.tensor(aligned.values, dtype=torch.float64)).squeeze(-1)
diff = (ours - torch.tensor(expected["tage_center_all"], dtype=torch.float64)).abs().max().item()
print("max abs diff:", diff)
assert diff < 1e-6
```

Repeat for `center_refgroup`. In `tagemortality.ipynb` use the `tagemortality_*` keys.

- [ ] **Step 5: Feature ranges, basic test, save**

Follow the kdmage cells: `resolve_feature_ranges(model.features, model.metadata["data_type"])` (expect unbounded), a basic forward pass on zeros (the reference point — prediction should be a plausible baseline age/hazard; print it), then `torch.save(model, f"../weights/{model.metadata['clock_name']}.pt")`. Also run `python ../build_tage_gene_mapping.py "$SCRATCH/tAge" ../weights/tage_gene_mapping.csv.gz` once and add the mouse-Entrez self-mapping assertion from Task 3 Step 1 now that `model.features` exists.

- [ ] **Step 6: Execute both notebooks end-to-end, then commit**

```bash
git add clocks/notebooks/tage.ipynb clocks/notebooks/tagemortality.ipynb
git commit -m "feat(clocks): add tage and tagemortality conversion notebooks"
```

(Weights and the mapping asset follow the repo's existing weight-handling convention — Task 9 uploads them; check `.gitignore` before assuming `clocks/weights/*.pt` is committed.)

---

### Task 8: End-to-end integration test

**Files:**
- Test: `tests/integration/test_tage_end_to_end.py`

**Interfaces:**
- Consumes: everything above; local weight files (skip if absent, same pattern as existing integration tests — check `tests/integration/` for the skip convention and mirror it).

- [ ] **Step 1: Write the test**

```python
import json
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest

import pyaging as pya
from pyaging.preprocess import _tage

DATA = Path(__file__).parents[1] / "data" / "tage"
TOL = 1e-6


@pytest.fixture(scope="module")
def example_adata():
    frame = pd.read_csv(DATA / "input_expression.csv.gz", index_col=0)
    # orient samples x genes per tests/data/tage/README.md
    a = anndata.AnnData(X=frame.values.astype(np.float64))
    a.obs_names = frame.index
    a.var_names = frame.columns
    return a


def test_full_pipeline_matches_reference(example_adata, monkeypatch, local_weights):
    expected = json.loads((DATA / "expected_predictions.json").read_text())
    # example data is already mouse Entrez space (see fixtures README); if the
    # README says otherwise, pass the recorded species instead.
    prepared = pya.pp.prepare_tage(example_adata, species="mouse")
    pya.pred.predict_age(prepared, ["tage", "tagemortality"], verbose=False)
    np.testing.assert_allclose(
        prepared.obs["tage"].to_numpy(),
        np.array(expected["tage_center_all"]),
        atol=TOL,
    )
    np.testing.assert_allclose(
        prepared.obs["tagemortality"].to_numpy(),
        np.array(expected["tagemortality_center_all"]),
        atol=TOL,
    )


def test_predict_without_prepare_raises(example_adata, local_weights):
    with pytest.raises(ValueError, match="prepare_tage"):
        pya.pred.predict_age(example_adata.copy(), ["tage"], verbose=False)


def test_batch_size_does_not_change_predictions(example_adata, local_weights):
    prepared = pya.pp.prepare_tage(example_adata, species="mouse")
    small = prepared.copy()
    pya.pred.predict_age(prepared, ["tage"], verbose=False)
    pya.pred.predict_age(small, ["tage"], batch_size=2, verbose=False)
    np.testing.assert_allclose(prepared.obs["tage"], small.obs["tage"], atol=1e-12)
```

`local_weights` is whatever fixture/skip mechanism the existing integration tests use to point `load_clock` at `clocks/weights/` instead of HF — reuse it verbatim. `prepare_tage`'s mapping download must also be redirected to the local `clocks/weights/tage_gene_mapping.csv.gz` (monkeypatch `_tage._load_mapping`).

- [ ] **Step 2: Run**

Run: `uv run pytest tests/integration/test_tage_end_to_end.py -v`
Expected: PASS. A failure here is a real discrepancy — debug with the stage fixtures to find which half (preprocessing vs weights) drifted; do not widen TOL.

- [ ] **Step 3: Run the full suite, then commit**

Run: `uv run pytest` — expected: all green.

```bash
git add tests/integration/test_tage_end_to_end.py
git commit -m "test: verify tage end-to-end parity with the published pipeline"
```

---

### Task 9: Metadata, golds, docs, and release wiring

**Files:**
- Modify: `clocks/metadata/clock_metadata.json` (via the established `clocks/update_all_clocks.py` flow — do not hand-edit; check its --help/source for the per-clock add path)
- Modify: boundary golds via `clocks/generate_boundary_gold.py`
- Modify: docs only if the Clock Catalogue does not auto-generate from metadata (check `docs/` build; the catalogue reads clock metadata, so usually no manual edit)

- [ ] **Step 1: Register metadata and validate**

Run the repo's metadata update flow for the two new clocks, then:

```bash
uv run python clocks/validate_metadata.py
uv run pytest tests/test_clock_metadata.py -v
```

Expected: both clocks present with `research_only: true`, `data_type: "transcriptomics (relative)"`, valid vocabulary.

- [ ] **Step 2: Generate boundary golds**

```bash
uv run python clocks/generate_boundary_gold.py tage tagemortality   # match the script's real CLI
uv run pytest -k "boundary" -v
```

- [ ] **Step 3: Random-input gold tests**

Add the two clocks to whatever gold suite enumerates clocks (it may pick them up automatically from metadata — verify by running it and confirming `tage`/`tagemortality` appear in the output):

```bash
uv run pytest -k "gold" -v
```

- [ ] **Step 4: Docs touch-up**

Rebuild docs and confirm the two clocks appear in the Clock Catalogue with the research-only flag visible. Add a short "Cohort-relative transcriptomic clocks" note to the relevant tutorial or docs page pointing at `prepare_tage` (one paragraph + a 5-line usage snippet; place it wherever the blood-chemistry tutorial pattern lives).

- [ ] **Step 5: Full suite + commit**

```bash
uv run pytest
git add -A
git commit -m "feat(clocks): register tage and tagemortality metadata, golds, and docs"
```

- [ ] **Step 6: HF upload (USER-GATED — do not run unattended)**

Upload `tage.pt`, `tagemortality.pt`, `config.json` metadata, and `tage_gene_mapping.csv.gz` to `pyaging/tage` and `pyaging/tagemortality` via the repo's `clocks/hf_repo_sync.py` / `make upload-clocks-to-hf` flow. This step requires the user present (HF permission classifier has blocked unattended uploads before). After upload, re-run the integration test WITHOUT the local-weights fixture to confirm the live path works, and verify the gene-mapping download resolves.

---

## Self-review notes

- Spec coverage: guard (Task 2), mapping + shipped table (Tasks 3, 7, 9), transforms (Task 4), `prepare_tage` + errors + uns provenance (Task 5), model classes + license flag + units (Tasks 6–7), parity ≤1e-6 vs reference pipeline with and without reference group (Tasks 1, 4, 7, 8), standard suite/golds/metadata (Task 9), batch-size independence (Task 8). Single-sample error and low-overlap warning: Task 5.
- Known unknowns are quarantined into Task 1 (R function names, matrix orientation, pickle object shape, units) and recorded in `tests/data/tage/README.md`, which later tasks cite instead of re-deriving.
- Type consistency: `_map_to_mouse_entrez(df, species, mapping)`, `_rle_normalize/_log_transform/_scale_genes(df)`, `_center_against_reference(df, reference_index)`, `prepare_tage(adata, species, reference_group, dir)`, `required_uns_flag`, flag value `"tage_prepared"` — used identically across Tasks 2–8.
