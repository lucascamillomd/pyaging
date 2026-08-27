# DNAmFitAge Composite Clocks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace four sex-specific DNAmFitAge gait/grip clocks with two sex-gated clocks, make DNAmFitAge calculate original GrimAge internally, and publish the breaking change as pyaging 0.5.2.

**Architecture:** Two self-contained gated PyTorch models own the original female and male regressions and select them through the numeric `female` covariate. The revised DNAmFitAge artifact embeds gait, grip, VO2max, and original GrimAge, maps one public feature union into each component, and retains component-specific reference values. Hugging Face migration and the GitHub/PyPI release happen only after local equivalence and catalogue verification pass.

**Tech Stack:** Python 3.11+, PyTorch, pandas, AnnData, pytest, Jupyter/nbconvert, R for the authors' RDS conversion, Sphinx, Hugging Face Hub CLI/API, GitHub Actions, uv, PyPI Trusted Publishing.

**Spec:** `docs/superpowers/specs/2026-08-27-dnamfitage-composite-clocks-design.md`

## Global Constraints

- Target version is exactly `0.5.2`; the four removals are intentionally breaking in a patch release.
- Remove `dnamfitagegaitf`, `dnamfitagegaitm`, `dnamfitagegripf`, and `dnamfitagegripm` without aliases.
- Do not modify `CHANGELOG.md`.
- New public names are exactly `dnamfitagegait` and `dnamfitagegrip`.
- Revised `dnamfitage` accepts methylation features, `female`, and `age`; it does not accept a public `grimage` feature.
- Embed original GrimAge, not GrimAge2, inside `dnamfitage.pt`; inference must not download nested dependencies.
- Missing `female` and `age` follow GrimAge references `1.0` and `65.0`; supplied numeric sex values are not rejected.
- Preserve component-specific methylation reference values with NaN sentinels in composite public reference vectors.
- Permanently delete the four obsolete per-clock Hugging Face repositories, histories, and tags only after replacements pass fresh-download inference.
- Remove obsolete fallback weights from the aggregate repository's `main` branch without rewriting aggregate history or tags.
- Use the repository convention `import pyaging as pya`; `predict_age` mutates AnnData in place.

## File Structure

- `src/pyaging/models/_models.py`: gated component implementation and revised self-contained DNAmFitAge forward graph.
- `src/pyaging/models/__init__.py`: public model-class exports; old class exports are removed at retirement.
- `src/pyaging/predict/_pred_utils.py`: unavailable-clock error text with installed-version guidance.
- `tests/models/test_dnamfitage_composites.py`: small deterministic unit tests for gating, reference substitution, and embedded-component wiring.
- `tests/predict/test_dnamfitage_composites.py`: local-weight oracle equivalence tests for both sexes and the old two-stage workflow.
- `tests/predict/test_hf_loading.py`: unavailable-clock error contract.
- `tests/predict/test_gold_standard.py`: seeded catalogue golds; four old entries removed and two new entries added.
- `tests/predict/test_boundary_gold_standard.py`: boundary golds for the three changed public artifacts.
- `tests/test_clock_metadata.py`: expected catalogue size after transitional addition and final retirement.
- `clocks/notebooks/dnamfitagegait.ipynb`: reproducible construction of the gated gait artifact.
- `clocks/notebooks/dnamfitagegrip.ipynb`: reproducible construction of the gated grip artifact.
- `clocks/notebooks/dnamfitage.ipynb`: revised construction of the self-contained final artifact.
- `clocks/notebooks/dnamfitage{gaitf,gaitm,gripf,gripm}.ipynb`: deleted after replacement artifacts pass.
- `clocks/weights/*.pt`: ignored local build output; create three replacements and delete four retired weights.
- `clocks/metadata/clock_metadata.json`: curated entries and exact feature counts (`111`, `183`, and `1343`).
- `clocks/metadata/evidence_ledger.jsonl`: evidence records renamed/consolidated in alphabetical order.
- `clocks/metadata/all_clock_metadata.pt`: ignored regenerated aggregate used for local verification and Hub upload.
- `docs/source/clock_implementation.rst`: replacement notebook toctree entries.
- `docs/_static/clocks.json`, `docs/_static/clock_glossary.csv`: regenerated public catalogue artifacts.
- `tutorials/tutorial_utils.ipynb`: refreshed `show_all_clocks()` output after aggregate Hub metadata is live.
- `src/pyaging/__init__.py`: version bump from `0.5.1` to `0.5.2`.

---

### Task 1: Add the reusable sex-gated model boundary

**Files:**
- Modify: `src/pyaging/models/_models.py:2115-2190,2683-2750`
- Modify: `src/pyaging/models/__init__.py:30-45,195-215`
- Create: `tests/models/test_dnamfitage_composites.py`

**Interfaces:**
- Consumes: `pyagingModel`, `LinearModel`, PyTorch tensors, and the existing convention that `model.features` defines public column order.
- Produces: `DNAmFitAgeGait`, `DNAmFitAgeGrip`, and private `_DNAmFitAgeSexGated`; both public classes expose `female_model`, `male_model`, `female_feature_indices`, `male_feature_indices`, `female_reference_values`, `male_reference_values`, and `female_index`.

- [ ] **Step 1: Write failing unit tests for exact gating and per-branch references**

Create `tests/models/test_dnamfitage_composites.py` with helpers and assertions independent of generated weights:

```python
import pytest
import torch

from pyaging.models import DNAmFitAgeGait, DNAmFitAgeGrip, LinearModel


def _linear(weight, bias):
    model = LinearModel(len(weight)).to(torch.float64)
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([weight], dtype=torch.float64))
        model.linear.bias.copy_(torch.tensor([bias], dtype=torch.float64))
    return model


@pytest.fixture(params=[DNAmFitAgeGait, DNAmFitAgeGrip])
def gated_model(request):
    model = request.param().to(torch.float64)
    model.features = ["female_probe", "male_probe", "female"]
    model.female_model = _linear([2.0], 1.0)
    model.male_model = _linear([3.0], -1.0)
    model.female_feature_indices = torch.tensor([0])
    model.male_feature_indices = torch.tensor([1])
    model.female_reference_values = [2.0]
    model.male_reference_values = [4.0]
    model.female_index = 2
    model.reference_values = [float("nan"), float("nan"), 1.0]
    return model


def test_sex_gated_model_matches_both_branches_and_blends(gated_model):
    rows = torch.tensor(
        [[3.0, 4.0, 0.0], [3.0, 4.0, 1.0], [3.0, 4.0, 0.25]],
        dtype=torch.float64,
    )
    assert gated_model(rows).ravel().tolist() == pytest.approx([11.0, 7.0, 10.0])


def test_sex_gated_model_uses_branch_specific_references(gated_model):
    rows = torch.tensor(
        [[float("nan"), float("nan"), 0.0], [float("nan"), float("nan"), 1.0]],
        dtype=torch.float64,
    )
    assert gated_model(rows).ravel().tolist() == pytest.approx([11.0, 5.0])


def test_sex_gated_model_propagates_supplied_nan_female(gated_model):
    row = torch.tensor([[3.0, 4.0, float("nan")]], dtype=torch.float64)
    assert torch.isnan(gated_model(row)).all()
```

- [ ] **Step 2: Run the focused test and verify the new exports do not exist**

Run: `uv run pytest tests/models/test_dnamfitage_composites.py -q`

Expected: collection fails because `DNAmFitAgeGait` and `DNAmFitAgeGrip` are not exported.

- [ ] **Step 3: Implement the private gated base and public classes**

Add this model boundary near the current DNAmFitAge classes in `_models.py`:

```python
def _fill_dnamfitage_references(x, reference_values):
    reference = torch.as_tensor(reference_values, device=x.device, dtype=x.dtype)
    return torch.where(torch.isnan(x), reference, x)


class _DNAmFitAgeSexGated(pyagingModel):
    def __init__(self):
        super().__init__()
        self.female_model = None
        self.male_model = None
        self.female_feature_indices = None
        self.male_feature_indices = None
        self.female_reference_values = None
        self.male_reference_values = None
        self.female_index = None

    def forward(self, x):
        female = x[:, self.female_index].unsqueeze(1)
        female_x = _fill_dnamfitage_references(
            x[:, self.female_feature_indices], self.female_reference_values
        )
        male_x = _fill_dnamfitage_references(
            x[:, self.male_feature_indices], self.male_reference_values
        )
        female_prediction = self.female_model(female_x)
        male_prediction = self.male_model(male_x)
        return male_prediction + female * (female_prediction - male_prediction)

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class DNAmFitAgeGait(_DNAmFitAgeSexGated):
    pass


class DNAmFitAgeGrip(_DNAmFitAgeSexGated):
    pass
```

Export `DNAmFitAgeGait` and `DNAmFitAgeGrip` from `src/pyaging/models/__init__.py`. Keep the four old classes temporarily so their local weights remain loadable until oracle replacement tasks finish.

- [ ] **Step 4: Run focused and neighbouring model tests**

Run: `uv run pytest tests/models/test_dnamfitage_composites.py tests/predict/test_bioage_clocks.py -q`

Expected: all tests pass.

- [ ] **Step 5: Run lint and commit the isolated model boundary**

Run: `uv run ruff check src/pyaging/models tests/models/test_dnamfitage_composites.py`

Commit:

```bash
git add src/pyaging/models/_models.py src/pyaging/models/__init__.py tests/models/test_dnamfitage_composites.py
git commit -m "feat(models): add sex-gated DNAmFitAge components"
```

---

### Task 2: Improve unavailable-clock guidance without a PyPI network lookup

**Files:**
- Modify: `src/pyaging/predict/_pred_utils.py:110-140`
- Modify: `tests/predict/test_hf_loading.py:25-48`

**Interfaces:**
- Consumes: `_load_clock_impl(clock_name, device, dir, logger, indent_level)` and package `__version__` loaded lazily after package initialization.
- Produces: the same chained `NameError`, augmented with installed version and upgrade advice; non-resource download failures remain unchanged.

- [ ] **Step 1: Tighten the missing-resource test around exact guidance**

Extend `test_load_clock_translates_only_missing_resource_to_chained_name_error`:

```python
    message = str(error.value)
    from pyaging import __version__

    assert f"Clock horvath2013 is not available on pyaging {__version__}" in message
    assert "may require a newer pyaging release" in message
    assert "check PyPI" in message
    assert "pip install --upgrade pyaging" in message
```

Keep the assertions that the cause is the original `PyAgingResourceNotFoundError`, `logger.error` is called once, and `PyAgingDownloadError` propagates unchanged.

- [ ] **Step 2: Run the single test and verify the current message fails**

Run: `uv run pytest tests/predict/test_hf_loading.py::test_load_clock_translates_only_missing_resource_to_chained_name_error -q`

Expected: FAIL because the current message omits version and upgrade guidance.

- [ ] **Step 3: Add lazy, network-free version guidance**

Inside the `except PyAgingResourceNotFoundError` block, import the version lazily to avoid the package-initialization cycle and build this message:

```python
        from pyaging import __version__

        message = (
            f"Clock {clock_name} is not available on pyaging {__version__}. "
            "This clock may require a newer pyaging release; check PyPI and run "
            "`pip install --upgrade pyaging` if a newer version is available. "
            "Please refer to the clock names in the clock glossary table "
            "in the package documentation page: pyaging.readthedocs.io"
        )
```

Do not call PyPI or change the exception types caught.

- [ ] **Step 4: Run the complete Hub-loading unit file**

Run: `uv run pytest tests/predict/test_hf_loading.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the error-path change**

```bash
git add src/pyaging/predict/_pred_utils.py tests/predict/test_hf_loading.py
git commit -m "fix(predict): improve unavailable clock guidance"
```

---

### Task 3: Build and register merged gait and grip artifacts

**Files:**
- Create: `clocks/notebooks/dnamfitagegait.ipynb`
- Create: `clocks/notebooks/dnamfitagegrip.ipynb`
- Create locally, ignored: `clocks/weights/dnamfitagegait.pt`
- Create locally, ignored: `clocks/weights/dnamfitagegrip.pt`
- Modify: `clocks/metadata/clock_metadata.json:1086-1225`
- Modify: `clocks/metadata/evidence_ledger.jsonl` around the DNAmFitAge records
- Modify: `tests/test_clock_metadata.py:35-45`
- Create: `tests/predict/test_dnamfitage_composites.py`

**Interfaces:**
- Consumes: `DNAmFitAgeGait`, `DNAmFitAgeGrip`, the authors' `DNAmFitnessModelsandFitAge_Oct2022.rds`, and exact former-model oracle outputs captured before retirement.
- Produces: local self-contained `dnamfitagegait.pt` (111 features) and `dnamfitagegrip.pt` (183 features), plus curated/evidence records under the two new names.

- [ ] **Step 1: Add local-weight oracle tests that fail while the new artifacts are absent**

Create `tests/predict/test_dnamfitage_composites.py`:

```python
import hashlib
from pathlib import Path

import pandas as pd
import pytest
import torch

import pyaging as pya

pytestmark = pytest.mark.full_catalog
WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "clocks" / "weights"


def _beta(feature):
    digest = int(hashlib.sha256(feature.encode()).hexdigest()[:8], 16)
    return 0.2 + (digest % 6000) / 10000


def _load_local(name):
    path = WEIGHTS_DIR / f"{name}.pt"
    assert path.is_file(), f"execute clocks/notebooks/{name}.ipynb"
    model = torch.load(path, weights_only=False, map_location="cpu")
    model.to(torch.float64).eval()
    return model


@pytest.mark.parametrize(
    ("clock_name", "expected_count", "expected"),
    [
        ("dnamfitagegait", 111, [2.5945144207221453, 1.9238271172398522]),
        ("dnamfitagegrip", 183, [42.75495020679189, 30.54383665460349]),
    ],
)
def test_merged_fitness_clocks_match_retired_sex_specific_oracles(
    clock_name, expected_count, expected
):
    model = _load_local(clock_name)
    assert len(model.features) == expected_count
    frame = pd.DataFrame(
        [{feature: _beta(feature) for feature in model.features} for _ in range(2)],
        index=["male", "female"],
    )
    frame["female"] = [0.0, 1.0]
    values = torch.as_tensor(frame[model.features].to_numpy(), dtype=torch.float64)
    with torch.no_grad():
        predictions = model(values).ravel().tolist()
    assert predictions == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize(
    ("clock_name", "expected"),
    [
        ("dnamfitagegait", 1.9238271172398522),
        ("dnamfitagegrip", 30.54383665460349),
    ],
)
def test_missing_female_uses_grimage_reference(monkeypatch, clock_name, expected):
    model = _load_local(clock_name)
    frame = pd.DataFrame(
        [{feature: _beta(feature) for feature in model.features if feature != "female"}]
    )
    monkeypatch.setattr(
        "pyaging.predict._pred_utils.download_clock_weights",
        lambda *args, **kwargs: str(WEIGHTS_DIR / f"{clock_name}.pt"),
    )
    adata = pya.pp.df_to_adata(frame, imputer_strategy="constant", verbose=False)
    pya.pred.predict_age(adata, clock_name, verbose=False)
    assert float(adata.obs[clock_name].iloc[0]) == pytest.approx(expected, abs=1e-10)
    assert adata.uns[f"{clock_name}_missing_features"] == ["female"]
    assert adata.uns[f"{clock_name}_percent_na"] == pytest.approx(100 / len(model.features))
```

- [ ] **Step 2: Run the oracle tests and verify both missing artifact assertions fail**

Run: `uv run pytest tests/predict/test_dnamfitage_composites.py -q`

Expected: FAIL for absent `dnamfitagegait.pt` and `dnamfitagegrip.pt`.

- [ ] **Step 3: Create two reproducible conversion notebooks**

Use the current four notebooks as source provenance, but consolidate each pair. Each new notebook must:

1. Clone `https://github.com/kristenmcgreevy/DNAmFitAge.git` into a temporary notebook working directory.
2. Read `DNAmFitnessModelsandFitAge_Oct2022.rds` in R and export the relevant female/male coefficient tables plus `Female_Medians_All` and `Male_Medians_All`.
3. Preserve assay names exactly, including `ch.2.105901354F` and `ch.13.39564907R` in the male gait model.
4. Construct ordered unions with first occurrence retained:

```python
def ordered_union(*groups):
    return list(dict.fromkeys(feature for group in groups for feature in group))
```

5. Set the public order to `ordered_union(female_features, male_features) + ["female"]`.
6. Build `female_model` and `male_model` as `LinearModel` instances, create both index tensors, store each sex's reference list independently, set `female_index`, and set the public reference vector to methylation NaNs followed by `1.0`.
7. Assert the derived lengths are exactly 111 for gait and 183 for grip.
8. Save to `../weights/{clock_name}.pt`.

Use these curated metadata values in both notebook metadata cells and registry entries:

```python
# gait-specific values
model.metadata["clock_name"] = "dnamfitagegait"
model.metadata["notes"] = "Sex-gated blood DNAm gait-speed estimator using the published female and male regressions selected by the female input."
model.metadata["population"] = "adults"
model.metadata["n_features"] = 111

# grip-specific values
model.metadata["clock_name"] = "dnamfitagegrip"
model.metadata["notes"] = "Sex-gated blood DNAm maximum-handgrip-strength estimator using the published female and male regressions selected by the female input."
model.metadata["population"] = "adults"
model.metadata["n_features"] = 183
```

Copy all other curated fields from the former pair: data type, species, year, citation, DOI, research flag, tissue, prediction target, training target, unit, model type `LASSO regression`, platform, journal, last author, citations, and citation date. Keep the required same-line `# Paper:` evidence comments exactly synchronized with the new evidence-ledger source text.

- [ ] **Step 4: Add alphabetized registry and evidence records while retaining old records temporarily**

Insert `dnamfitagegait` before `dnamfitagegaitf` and `dnamfitagegrip` before `dnamfitagegripf` in both metadata files. Consolidate population evidence to adults and make the `n_features` evidence state that the packaged count is the ordered sex-specific feature union plus `female`. Change the transitional registry-size assertion from 179 to 181.

- [ ] **Step 5: Execute only the two new notebooks**

From `clocks/notebooks`, run:

```bash
uv run jupyter nbconvert --execute --inplace dnamfitagegait.ipynb
uv run jupyter nbconvert --execute --inplace dnamfitagegrip.ipynb
```

Expected: both exit 0 and create the two ignored weight files with the asserted counts.

- [ ] **Step 6: Run oracle, registry, and evidence tests**

Run:

```bash
uv run pytest tests/predict/test_dnamfitage_composites.py tests/test_clock_metadata.py::test_registry_has_every_implementation_notebook tests/test_clock_metadata.py::test_registry_uses_controlled_arrays tests/test_clock_metadata.py::test_evidence_is_complete_and_resolved -q
```

Expected: all pass; oracle tolerance is `1e-10` for both sexes.

- [ ] **Step 7: Commit tracked notebooks, metadata, and tests**

Do not force-add ignored weights.

```bash
git add clocks/notebooks/dnamfitagegait.ipynb clocks/notebooks/dnamfitagegrip.ipynb clocks/metadata/clock_metadata.json clocks/metadata/evidence_ledger.jsonl tests/test_clock_metadata.py tests/predict/test_dnamfitage_composites.py
git commit -m "feat(clocks): add merged DNAmFitAge gait and grip"
```

---

### Task 4: Embed original GrimAge inside DNAmFitAge

**Files:**
- Modify: `src/pyaging/models/_models.py:2115-2190`
- Modify: `clocks/notebooks/dnamfitage.ipynb`
- Replace locally, ignored: `clocks/weights/dnamfitage.pt`
- Modify: `clocks/metadata/clock_metadata.json` entry `dnamfitage`
- Modify: `clocks/metadata/evidence_ledger.jsonl` record `dnamfitage`
- Modify: `tests/models/test_dnamfitage_composites.py`
- Modify: `tests/predict/test_dnamfitage_composites.py`

**Interfaces:**
- Consumes: generated `dnamfitagegait.pt`, `dnamfitagegrip.pt`, retained `dnamfitagevo2max.pt`, retained original `grimage.pt`, and `female`/`age` public covariates.
- Produces: revised `DNAmFitAge` with `Gait`, `Grip`, `VO2Max`, `GrimAge`, component feature-index tensors, component-specific input filling, and no public `grimage` feature.

- [ ] **Step 1: Add a failing self-contained oracle test**

Append to `tests/predict/test_dnamfitage_composites.py`:

```python
def test_dnamfitage_embeds_grimage_and_matches_retired_two_stage_oracle():
    model = _load_local("dnamfitage")
    assert len(model.features) == 1343
    assert model.features[-2:] == ["female", "age"]
    assert "grimage" not in model.features
    assert model.GrimAge.metadata["clock_name"] == "grimage"

    frame = pd.DataFrame(
        [{feature: _beta(feature) for feature in model.features} for _ in range(2)],
        index=["male", "female"],
    )
    frame["female"] = [0.0, 1.0]
    frame["age"] = 57.0
    values = torch.as_tensor(frame[model.features].to_numpy(), dtype=torch.float64)
    with torch.no_grad():
        predictions = model(values).ravel().tolist()
    assert predictions == pytest.approx(
        [131.19968704579992, 133.55199017157764], abs=1e-10
    )


def test_dnamfitage_missing_age_and_female_use_grimage_references(monkeypatch):
    model = _load_local("dnamfitage")
    frame = pd.DataFrame(
        [
            {
                feature: _beta(feature)
                for feature in model.features
                if feature not in {"female", "age"}
            }
        ]
    )
    monkeypatch.setattr(
        "pyaging.predict._pred_utils.download_clock_weights",
        lambda *args, **kwargs: str(WEIGHTS_DIR / "dnamfitage.pt"),
    )
    adata = pya.pp.df_to_adata(frame, imputer_strategy="constant", verbose=False)
    pya.pred.predict_age(adata, "dnamfitage", verbose=False)
    assert float(adata.obs["dnamfitage"].iloc[0]) == pytest.approx(
        137.34972213448359, abs=1e-10
    )
    assert adata.uns["dnamfitage_missing_features"] == ["female", "age"]
```

- [ ] **Step 2: Run the new test against the old artifact and verify contract failure**

Run: `uv run pytest tests/predict/test_dnamfitage_composites.py::test_dnamfitage_embeds_grimage_and_matches_retired_two_stage_oracle -q`

Expected: FAIL because the old artifact has 630 features, includes `grimage`, and lacks `GrimAge`.

- [ ] **Step 3: Refactor `DNAmFitAge` into a self-contained composite**

Change constructor attributes to:

```python
self.Gait = None
self.Grip = None
self.VO2Max = None
self.GrimAge = None
self.features_Gait = None
self.features_Grip = None
self.features_VO2Max = None
self.features_GrimAge = None
self.female_index = None
self.age_index = None
```

Use one private method to align and fill each component:

```python
def _component_input(self, x, indices, component):
    values = x[:, indices]
    return _fill_dnamfitage_references(values, component.reference_values)
```

The forward method computes all rows through both final equations and blends them:

```python
female = x[:, self.female_index].unsqueeze(1)
gait = self.Gait(self._component_input(x, self.features_Gait, self.Gait))
grip = self.Grip(self._component_input(x, self.features_Grip, self.Grip))
vo2max = self.VO2Max(self._component_input(x, self.features_VO2Max, self.VO2Max))
grimage = self.GrimAge(self._component_input(x, self.features_GrimAge, self.GrimAge))

female_components = torch.concat(
    [
        (vo2max - 46.825091) / -0.13620215,
        (grip - 39.857718) / -0.22074456,
        (gait - 2.508547) / -0.01245682,
        (grimage - 7.978487) / 0.80928530,
    ],
    dim=1,
)
male_components = torch.concat(
    [
        (vo2max - 49.836389) / -0.141862925,
        (grip - 57.514016) / -0.253179827,
        (gait - 2.349080) / -0.009380061,
        (grimage - 9.549733) / 0.835120557,
    ],
    dim=1,
)
female_prediction = self.base_model_f(female_components)
male_prediction = self.base_model_m(male_components)
return male_prediction + female * (female_prediction - male_prediction)
```

Retain no-download semantics: `forward` calls only embedded modules.

- [ ] **Step 4: Add a small wiring test with fake embedded modules**

In `tests/models/test_dnamfitage_composites.py`, construct a `DNAmFitAge` with one-column constant-output components and identity-like final `LinearModel(4)` layers. Assert both binary sexes return their expected branch and `female=0.5` returns their midpoint. Also assert `_component_input` replaces a NaN using the embedded component's reference rather than the outer reference vector.

- [ ] **Step 5: Rebuild `dnamfitage.ipynb` around embedded artifacts**

The notebook must load local artifacts with `torch.load(..., weights_only=False, map_location="cpu")`, then set:

```python
model.Gait = torch.load("../weights/dnamfitagegait.pt", weights_only=False, map_location="cpu")
model.Grip = torch.load("../weights/dnamfitagegrip.pt", weights_only=False, map_location="cpu")
model.VO2Max = torch.load("../weights/dnamfitagevo2max.pt", weights_only=False, map_location="cpu")
model.GrimAge = torch.load("../weights/grimage.pt", weights_only=False, map_location="cpu")
```

Build the public feature list as the ordered union of each component's non-covariate assay features, followed by `female`, `age`. Create each component index tensor from exact feature names. Set `female_index`, `age_index`, and the outer reference list to NaN for all assay features followed by `1.0`, `65.0`. Assert:

```python
assert len(model.features) == 1343
assert model.features[-2:] == ["female", "age"]
assert "grimage" not in model.features
assert model.GrimAge.metadata["clock_name"] == "grimage"
```

Retain the published male/female final weights. Update notes to:

```text
Sex-specific Klemera–Doubal biological-age composite that calculates DNAm gait speed, grip strength, VO2max, and original DNAmGrimAge internally from methylation, age, and female inputs.
```

Set curated `n_features` to 1343 in the notebook, registry, and evidence ledger; explain that it is the deduplicated public union including `female` and `age`.

- [ ] **Step 6: Execute the revised notebook and rerun both oracle layers**

From `clocks/notebooks`, run: `uv run jupyter nbconvert --execute --inplace dnamfitage.ipynb`

Then run:

```bash
uv run pytest tests/models/test_dnamfitage_composites.py tests/predict/test_dnamfitage_composites.py -q
```

Expected: all tests pass; self-contained outputs equal `[131.19968704579992, 133.55199017157764]` within `1e-10`. These values were independently reproduced through the retained old two-stage workflow with original GrimAge outputs `[178.42414594960613, 176.52173460278965]`; the earlier planned values were not backward-equivalent.

- [ ] **Step 7: Commit source, notebook, metadata, evidence, and tests**

```bash
git add src/pyaging/models/_models.py clocks/notebooks/dnamfitage.ipynb clocks/metadata/clock_metadata.json clocks/metadata/evidence_ledger.jsonl tests/models/test_dnamfitage_composites.py tests/predict/test_dnamfitage_composites.py
git commit -m "feat(clocks): calculate GrimAge inside DNAmFitAge"
```

---

### Task 5: Retire the four public sex-specific clocks

**Files:**
- Modify: `src/pyaging/models/_models.py:2683-2750`
- Modify: `src/pyaging/models/__init__.py`
- Delete: `clocks/notebooks/dnamfitagegaitf.ipynb`
- Delete: `clocks/notebooks/dnamfitagegaitm.ipynb`
- Delete: `clocks/notebooks/dnamfitagegripf.ipynb`
- Delete: `clocks/notebooks/dnamfitagegripm.ipynb`
- Delete locally, ignored: four corresponding `clocks/weights/*.pt`
- Modify: `clocks/metadata/clock_metadata.json`
- Modify: `clocks/metadata/evidence_ledger.jsonl`
- Modify: `docs/source/clock_implementation.rst:42-48`
- Modify: `tests/test_clock_metadata.py:35-45`

**Interfaces:**
- Consumes: replacement classes, notebooks, local weights, registry entries, and evidence records from Tasks 1, 3, and 4.
- Produces: final 177-clock registry/notebook set with no import, notebook, local weight, registry, evidence, or docs index entry for the retired names.

- [ ] **Step 1: Change catalogue-shape expectations before removal**

Set `test_registry_has_every_implementation_notebook` to expect 177 entries. Add an explicit assertion:

```python
    retired = {
        "dnamfitagegaitf",
        "dnamfitagegaitm",
        "dnamfitagegripf",
        "dnamfitagegripm",
    }
    assert retired.isdisjoint(registry)
```

- [ ] **Step 2: Run the shape test and verify it fails against the transitional 181-clock set**

Run: `uv run pytest tests/test_clock_metadata.py::test_registry_has_every_implementation_notebook -q`

Expected: FAIL because old artifacts and registry entries still exist.

- [ ] **Step 3: Remove old classes, exports, notebooks, metadata, and evidence**

Use `apply_patch` for tracked files. Replace four toctree lines with:

```rst
   clock_notebooks/dnamfitagegait
   clock_notebooks/dnamfitagegrip
```

Delete only these ignored generated files after confirming exact paths:

```bash
rm clocks/weights/dnamfitagegaitf.pt
rm clocks/weights/dnamfitagegaitm.pt
rm clocks/weights/dnamfitagegripf.pt
rm clocks/weights/dnamfitagegripm.pt
```

Do not delete `dnamfitagevo2max.pt`, `grimage.pt`, replacement weights, or any Hub repository in this task.

- [ ] **Step 4: Prove retired names are absent locally**

Run:

```bash
rg -n "DNAmFitAge(GaitF|GaitM|GripF|GripM)|dnamfitage(gaitf|gaitm|gripf|gripm)" src clocks/metadata clocks/notebooks docs/source/clock_implementation.rst
```

Expected: no matches in runtime source or current artifact registries. Regression tests and the approved spec/plan intentionally retain the retired strings.

- [ ] **Step 5: Run registry, evidence, model, and oracle tests**

Run:

```bash
uv run pytest tests/test_clock_metadata.py::test_registry_has_every_implementation_notebook tests/test_clock_metadata.py::test_registry_uses_controlled_arrays tests/test_clock_metadata.py::test_evidence_is_complete_and_resolved tests/models/test_dnamfitage_composites.py tests/predict/test_dnamfitage_composites.py -q
```

Expected: all pass and registry count is 177.

- [ ] **Step 6: Commit the breaking local removal**

```bash
git add src/pyaging/models/_models.py src/pyaging/models/__init__.py clocks/notebooks clocks/metadata/clock_metadata.json clocks/metadata/evidence_ledger.jsonl docs/source/clock_implementation.rst tests/test_clock_metadata.py
git commit -m "refactor(clocks): remove sex-specific DNAmFitAge names"
```

---

### Task 6: Regenerate catalogue golds, metadata, and documentation for 0.5.2

**Files:**
- Modify: `src/pyaging/__init__.py:8`
- Modify: `tests/predict/test_gold_standard.py`
- Modify: `tests/predict/test_boundary_gold_standard.py`
- Regenerate locally, ignored: `clocks/metadata/all_clock_metadata.pt`
- Restamp locally, ignored: only `clocks/weights/{dnamfitage,dnamfitagegait,dnamfitagegrip}.pt`
- Modify: `docs/_static/clocks.json`
- Modify: `docs/_static/clock_glossary.csv`

**Interfaces:**
- Consumes: final 177-clock registry, notebooks, and local weight set.
- Produces: three `v0.5.2` changed weights, 174 byte-identical `v0.5.1` survivor weights, a truthful mixed-version aggregate, new seeded and boundary golds, and public catalogue assets with only the replacement names.

> **User-authorized focused scope (2026-08-27):** This is the binding scope for
> Tasks 6–10. Stamp only `dnamfitage`,
> `dnamfitagegait`, and `dnamfitagegrip` as `v0.5.2`; keep the 174 unaffected
> survivor weights byte-for-byte identical to the base checkout and verify that
> identity with an all-174 checksum comparison. Build the final 177-entry
> aggregate from that mixed local set, preserving each artifact's actual version
> (`174 × v0.5.1`, `3 × v0.5.2`). Calculate and test only the three changed
> seeded/boundary golds, remove the four retired entries, keep every unrelated
> gold literal unchanged, and run focused DNAmFitAge, registry/evidence, docs,
> catalogue, and representative artifact checks. The initial all-177 local
> restamp completed before this scope change was received; it must be reversed
> for the 174 unaffected weights and must not be carried into publication.

- [ ] **Step 1: Bump the package source version with `apply_patch`**

Change only:

```python
__version__ = "0.5.2"
```

Run: `uv run python -c 'import pyaging; assert pyaging.__version__ == "0.5.2"'`

- [ ] **Step 2: Restamp only the three changed weights and regenerate truthful aggregate metadata**

Do not run the exhaustive form of `clocks/update_all_clocks.py`. After the three
changed notebooks have generated their weights, run this exact narrow command
from the repository root:

```bash
uv run python clocks/update_all_clocks.py v0.5.2 \
  --clock dnamfitage \
  --clock dnamfitagegait \
  --clock dnamfitagegrip
```

The selected mode uses the same staged serialization and registry merge as the
full updater, but resaves only those three weights and rebuilds aggregate
metadata from every artifact's embedded version. Restore the other 174 weights
byte-for-byte from the base checkout and require an all-174 checksum comparison
to produce no differences.

Expected: the weight filenames exactly match all 177 registry keys; the three
changed weights carry `v0.5.2`; the 174 unaffected weights remain byte-identical
`v0.5.1`; and `all_clock_metadata.pt` has exact version counts
`174 × v0.5.1` and `3 × v0.5.2`.

- [ ] **Step 3: Make the seeded catalogue suite consume local release weights**

The replacement names are deliberately not live on Hugging Face yet. Refactor `tests/predict/test_gold_standard.py` to load `clocks/weights/{clock_name}.pt` with `torch.load(..., weights_only=False, map_location="cpu")`, matching the local-artifact boundary already used by the boundary suite. If a weight is absent, skip with the notebook-generation instruction. Replace the second remote `predict_age` load with:

```python
check_features_in_adata(random_adata, clock, logger, indent_level=indent_level)
predictions = predict_ages_with_model(
    random_adata, clock, device, 1024, logger, indent_level=indent_level
)
pred = float(np.asarray(predictions).ravel()[0])
```

Import `check_features_in_adata` and `predict_ages_with_model` from `pyaging.predict._pred_utils`. Remove download-cache cleanup from this local-only test and release model/frame objects normally. Update the module description to state that golds pin the current local release artifacts, not live Hub state.

- [ ] **Step 4: Generate the three changed seeded catalogue predictions**

Run this read-only calculation from the repository root and copy the printed `repr` values into `gold_standard_dict` for exactly `dnamfitage`, `dnamfitagegait`, and `dnamfitagegrip`:

```python
import numpy as np
import pandas as pd
import torch

import pyaging as pya
from pyaging.predict._pred_utils import (
    check_features_in_adata,
    predict_ages_with_model,
)

for clock_name in ("dnamfitage", "dnamfitagegait", "dnamfitagegrip"):
    clock = torch.load(f"clocks/weights/{clock_name}.pt", weights_only=False)
    clock.to(torch.float64).to("cpu").eval()
    partial = clock.features[: max(1, len(clock.features) * 2 // 3)]
    np.random.seed(42)
    frame = pd.DataFrame(
        np.abs(np.random.normal(loc=0.5, scale=1, size=(1, len(partial)))),
        columns=partial,
    )
    adata = pya.pp.df_to_adata(frame, imputer_strategy="constant", verbose=False)
    logger = pya.logger.Logger("task6_seeded_gold")
    pya.logger.silence_logger("task6_seeded_gold")
    check_features_in_adata(adata, clock, logger, indent_level=1)
    predictions = predict_ages_with_model(
        adata,
        clock,
        "cpu",
        1024,
        logger,
        indent_level=1,
    )
    print(clock_name, repr(float(np.asarray(predictions).ravel()[0])))
```

Remove the four retired entries. Review that no unrelated dictionary value changes.

- [ ] **Step 5: Calculate and review the three changed boundary golds**

Import `predict_at_boundaries` from
`tests/predict/test_boundary_gold_standard.py` and print predictions only for
`dnamfitage`, `dnamfitagegait`, and `dnamfitagegrip`. Do not run the exhaustive
boundary generator.

Replace only those three entries, remove the four retired entries, and use an
AST comparison against the parent revision to prove every unrelated dictionary
literal is unchanged.

- [ ] **Step 6: Run the exact focused full-catalog gold cases**

Run:

```bash
uv run pytest -m full_catalog \
  'tests/predict/test_gold_standard.py::test_all_clocks[dnamfitage]' \
  'tests/predict/test_gold_standard.py::test_all_clocks[dnamfitagegait]' \
  'tests/predict/test_gold_standard.py::test_all_clocks[dnamfitagegrip]' \
  'tests/predict/test_boundary_gold_standard.py::test_boundary_predictions_match_gold[dnamfitage]' \
  'tests/predict/test_boundary_gold_standard.py::test_boundary_predictions_match_gold[dnamfitagegait]' \
  'tests/predict/test_boundary_gold_standard.py::test_boundary_predictions_match_gold[dnamfitagegrip]' \
  tests/predict/test_dnamfitage_composites.py -q
```

Expected: exactly 12 focused tests pass; no selected test references a retired
name. The explicit `-m full_catalog` is required because project defaults
otherwise deselect both gold modules.

- [ ] **Step 7: Validate focused weight and aggregate consistency**

Load only the three changed weights and assert their feature counts are
1,343/111/183, their `version` and `metadata["version"]` are `v0.5.2`, and
their stored `feature_units` equal `resolve_feature_ranges(...)`. Separately
assert the aggregate key set exactly equals the 177-key registry, excludes the
four retired names, and has versions `174 × v0.5.1` plus `3 × v0.5.2`.

- [ ] **Step 8: Regenerate committed Clock Explorer assets from local aggregate metadata**

Run:

```bash
uv run python docs/source/make_clock_data.py --metadata-path clocks/metadata/all_clock_metadata.pt
```

Assert with `rg` that `docs/_static/clocks.json` and `docs/_static/clock_glossary.csv` contain the two new names and revised `dnamfitage`, contain none of the four retired names, and report feature counts 111, 183, and 1343.

- [ ] **Step 9: Build documentation and run catalogue-focused tests**

Run:

```bash
uv run make -C docs html
uv run pytest \
  tests/test_clock_metadata.py::test_registry_has_every_implementation_notebook \
  tests/test_clock_metadata.py::test_registry_uses_controlled_arrays \
  tests/test_clock_metadata.py::test_evidence_is_complete_and_resolved -q
```

Expected: Sphinx exits 0, the three metadata tests pass, and generated docs
contain the three current DNAmFitAge pages.

- [ ] **Step 10: Commit version, golds, and generated public catalogue assets**

```bash
git add src/pyaging/__init__.py tests/predict/test_gold_standard.py tests/predict/test_boundary_gold_standard.py docs/_static/clocks.json docs/_static/clock_glossary.csv
git commit -m "chore: prepare DNAmFitAge clocks for v0.5.2"
```

---

### Task 7: Complete local release verification before external mutation

**Files:**
- Verify: all tracked changes and ignored local release artifacts
- Build locally, ignored: `dist/*`

**Interfaces:**
- Consumes: completed source, notebooks, local weights, metadata aggregate, docs, and version 0.5.2.
- Produces: a clean, tested release candidate ready for reversible Hub upload; no external state changes occur in this task.

> **Focused verification amendment (2026-08-27):** Do not run all-177
> inference, all local full-catalog tests, or all notebook tests. Verification is
> limited to the three changed DNAmFitAge artifacts/notebooks, registry and
> evidence consistency, generated static catalogue assertions, documentation,
> Ruff/format/diff checks, configured hooks, and representative unchanged
> artifacts. Distribution/release checks remain local and non-mutating when
> requested separately.

- [ ] **Step 1: Execute the three changed conversion notebooks from a clean temporary working directory context**

Run the gait, grip, and final DNAmFitAge notebooks in dependency order with a 600-second cell timeout. Record the SHA256 of each generated weight, execute the same notebook a second time, and require the weight SHA256 to be identical before proceeding. Execution counts or captured command output may change notebook JSON, but weight-byte variance is a failure.

- [ ] **Step 2: Narrowly restamp the three notebook outputs and rebuild aggregate/static metadata**

Notebook construction deliberately leaves release versions unset. Immediately
after the reproducibility run, and before any artifact verification or upload,
run from the repository root:

```bash
uv run python clocks/update_all_clocks.py v0.5.2 \
  --clock dnamfitage \
  --clock dnamfitagegait \
  --clock dnamfitagegrip
uv run python docs/source/make_clock_data.py \
  --metadata-path clocks/metadata/all_clock_metadata.pt
```

Require both `model.version` and `model.metadata["version"]` to equal
`v0.5.2` for exactly those three weights. Require the aggregate to contain 177
keys with versions `174 × v0.5.1` / `3 × v0.5.2`, and require an all-174
checksum comparison against the base checkout to remain empty. This selected
mode must not stage or resave an unaffected weight.

- [ ] **Step 3: Run formatting and lint checks without unrelated rewrites**

Run:

```bash
uv run ruff check src/pyaging tests
uv run ruff format --check src/pyaging tests
git diff --check
```

Expected: all exit 0.

- [ ] **Step 4: Run the focused non-catalogue regression subset**

Run:

```bash
uv run pytest \
  tests/models/test_dnamfitage_composites.py \
  tests/predict/test_hf_loading.py \
  tests/test_clock_metadata.py::test_registry_has_every_implementation_notebook \
  tests/test_clock_metadata.py::test_registry_uses_controlled_arrays \
  tests/test_clock_metadata.py::test_evidence_is_complete_and_resolved -q
```

Expected: exactly 14 focused tests pass.

- [ ] **Step 5: Run focused full-catalog and notebook tests**

Run:

```bash
uv run pytest -m full_catalog \
  'tests/predict/test_gold_standard.py::test_all_clocks[dnamfitage]' \
  'tests/predict/test_gold_standard.py::test_all_clocks[dnamfitagegait]' \
  'tests/predict/test_gold_standard.py::test_all_clocks[dnamfitagegrip]' \
  'tests/predict/test_boundary_gold_standard.py::test_boundary_predictions_match_gold[dnamfitage]' \
  'tests/predict/test_boundary_gold_standard.py::test_boundary_predictions_match_gold[dnamfitagegait]' \
  'tests/predict/test_boundary_gold_standard.py::test_boundary_predictions_match_gold[dnamfitagegrip]' \
  tests/predict/test_dnamfitage_composites.py -q
uv run pytest --nbmake \
  clocks/notebooks/dnamfitagegait.ipynb \
  clocks/notebooks/dnamfitagegrip.ipynb \
  clocks/notebooks/dnamfitage.ipynb
```

Expected: exactly 12 focused catalogue/composite tests pass and all three
changed notebooks pass.

- [ ] **Step 6: Build and inspect distributions**

Run:

```bash
uv build
uvx twine check dist/*
uv run pytest tests/test_release_configuration.py -q
```

Inspect wheel and sdist filenames and metadata; both must say 0.5.2 and must not contain ignored clock weights.

- [ ] **Step 7: Verify exact repository state before Hub work**

Run:

```bash
git status --short
git log --oneline --decorate -10
git diff main@{upstream}..HEAD --stat
```

Expected: only intentional tracked changes/commits are present; ignored weights and aggregate exist locally; `CHANGELOG.md` is unchanged.

---

### Task 8: Upload and verify replacement Hugging Face artifacts

**Files:**
- External update: only `pyaging/dnamfitage`, `pyaging/dnamfitagegait`, and `pyaging/dnamfitagegrip`
- External update: `lucascamillomd/pyaging-data` aggregate metadata
- Modify after upload: `tutorials/tutorial_utils.ipynb` output only

**Interfaces:**
- Consumes: authenticated Hugging Face user `lucascamillomd`, final 177 local weight files, curated registry, and aggregate metadata.
- Produces: live updates to the three replacement repositories and aggregate metadata on `main`, verified through token-free fresh downloads; no deletion occurs until every check passes.

> **Focused upload amendment (2026-08-27):** Upload only
> `dnamfitage.pt`, `dnamfitagegait.pt`, `dnamfitagegrip.pt`, and
> `all_clock_metadata.pt`. Do not upload or retag the other 174 per-clock
> repositories during this release. The aggregate truthfully retains `v0.5.1`
> for those unchanged artifacts and uses `v0.5.2` only for the three changed
> clocks.

- [ ] **Step 1: Verify authentication, ownership, public aggregate repository, and exact local candidates**

Run:

```bash
uv run hf auth whoami --format json
make verify-hf-data-repo-public
```

Require user `lucascamillomd`. Confirm the local files `dnamfitage.pt`, `dnamfitagegait.pt`, and `dnamfitagegrip.pt` exist and the four retired files do not.

- [ ] **Step 2: Sync the three changed repositories and aggregate metadata**

Upload the exact local files for `dnamfitage`, `dnamfitagegait`, and
`dnamfitagegrip` to their corresponding repositories, then upload
`clocks/metadata/all_clock_metadata.pt` to `lucascamillomd/pyaging-data`.
Do not use the all-clock upload target and do not mutate the other 174
repositories.

- [ ] **Step 3: Verify replacement artifacts through fresh anonymous downloads**

Create the cache with `tempfile.mkdtemp(prefix="pyaging-hf-verify-")` and pass it as `cache_dir` to `hf_hub_download(..., token=False, force_download=True)` for:

- `pyaging/dnamfitage/dnamfitage.pt`
- `pyaging/dnamfitagegait/dnamfitagegait.pt`
- `pyaging/dnamfitagegrip/dnamfitagegrip.pt`

Load each file with current source; assert class names, feature counts
1343/111/183, and both `model.version` and `model.metadata["version"]` equal
`v0.5.2`; then run the deterministic oracle inputs from
`tests/predict/test_dnamfitage_composites.py`. Require the same expected outputs
before continuing. Audit the upload commits and require that only these three
per-clock repositories plus the aggregate repository changed; the other 174
per-clock repositories must be untouched.

- [ ] **Step 4: Verify aggregate metadata on `main`**

Download `all_clock_metadata.pt` anonymously with a fresh cache. Assert it has
177 keys, includes the three current names/counts, excludes all four retired
names, and has exact version counts `174 × v0.5.1` and `3 × v0.5.2`. Require
the three `v0.5.2` keys to be exactly `dnamfitage`, `dnamfitagegait`, and
`dnamfitagegrip`.

- [ ] **Step 5: Refresh the utility tutorial against live aggregate metadata**

Execute `tutorials/tutorial_utils.ipynb` with a newly created temporary Hugging Face cache so it cannot reuse pre-upload aggregate metadata. Confirm `show_all_clocks()` output contains `dnamfitagegait` and `dnamfitagegrip` and no retired name. Rebuild docs and run the notebook test:

```bash
uv run pytest --nbmake tutorials/tutorial_utils.ipynb
uv run make -C docs html
```

- [ ] **Step 6: Commit the refreshed tutorial output**

```bash
git add tutorials/tutorial_utils.ipynb
git commit -m "docs: refresh clock catalogue tutorial"
```

---

### Task 9: Cross the irreversible Hugging Face retirement boundary and tag data v0.5.2

**Files:**
- Permanently delete external repositories: `pyaging/dnamfitagegaitf`, `pyaging/dnamfitagegaitm`, `pyaging/dnamfitagegripf`, `pyaging/dnamfitagegripm`
- Delete four files from external aggregate repository `main`
- Create/update external tag: `v0.5.2` only on the three changed per-clock repositories and aggregate repository

**Interfaces:**
- Consumes: successful fresh-download verification from Task 8 and the user's explicit authorization to delete all four repositories including history and tags.
- Produces: retired names unavailable from live Hub state and the three changed names plus aggregate metadata reproducibly available at revision `v0.5.2`.

- [ ] **Step 1: Re-run the destructive-operation preflight immediately before deletion**

Confirm authentication still resolves to `lucascamillomd`. Run `hf models info --format json` separately for `pyaging/dnamfitagegaitf`, `pyaging/dnamfitagegaitm`, `pyaging/dnamfitagegripf`, and `pyaging/dnamfitagegripm`; verify each returned ID exactly equals the requested repository. Stop if any identity differs or a replacement fresh-download check no longer passes.

- [ ] **Step 2: Remove obsolete aggregate fallback files from `main`**

Run one explicit deletion commit:

```bash
uv run hf repos delete-files lucascamillomd/pyaging-data dnamfitagegaitf.pt dnamfitagegaitm.pt dnamfitagegripf.pt dnamfitagegripm.pt --type model --revision main --commit-message "Remove retired DNAmFitAge sex-specific weights"
```

Use `HfApi.file_exists(..., revision="main")` to assert all four are absent and `all_clock_metadata.pt` remains present.

- [ ] **Step 3: Permanently delete each exact obsolete per-clock repository**

Run these four commands individually; do not use globs or variables:

```bash
uv run hf repos delete pyaging/dnamfitagegaitf --type model --no-missing-ok --yes
uv run hf repos delete pyaging/dnamfitagegaitm --type model --no-missing-ok --yes
uv run hf repos delete pyaging/dnamfitagegripf --type model --no-missing-ok --yes
uv run hf repos delete pyaging/dnamfitagegripm --type model --no-missing-ok --yes
```

After each command, confirm `HfApi.repo_exists(repo_id, repo_type="model")` is false. Report that repository history and tags are no longer recoverable from Hugging Face.

- [ ] **Step 4: Tag the three changed repositories and aggregate data**

Create or repoint `v0.5.2` only for `pyaging/dnamfitage`,
`pyaging/dnamfitagegait`, `pyaging/dnamfitagegrip`, and
`lucascamillomd/pyaging-data`. Do not use the all-clock tag target and do not
retag the other 174 per-clock repositories.

- [ ] **Step 5: Verify pinned replacement resolution and removed-name error behavior**

With `PYAGING_DATA_REVISION=v0.5.2` and a fresh cache:

1. Load and run `dnamfitage`, `dnamfitagegait`, and `dnamfitagegrip` through `pya.pred.load_clock`/`predict_age`.
2. Assert feature counts and deterministic oracle outputs.
3. Attempt each retired name and assert `NameError` contains installed version 0.5.2, `check PyPI`, and the upgrade command.
4. Assert the aggregate tagged metadata excludes all four retired keys and has
   exact version counts `174 × v0.5.1` and `3 × v0.5.2`, with only the three
   changed keys at `v0.5.2`.
5. Audit tags/commits and require that only the three changed per-clock
   repositories plus aggregate metadata received `v0.5.2`; the other 174
   repositories remain untouched.

Do not proceed to GitHub publication unless every pinned check passes.

---

### Task 10: Push main, publish GitHub/PyPI 0.5.2, and verify a clean install

**Files:**
- External update: GitHub `main`
- External create: annotated Git tag `v0.5.2`
- External create: GitHub release `v0.5.2`
- External create: immutable PyPI distribution `pyaging==0.5.2`

**Interfaces:**
- Consumes: clean local commits, verified Hub tag `v0.5.2`, passing local suite, and GitHub push credentials.
- Produces: public GitHub and PyPI release 0.5.2 with working replacement clocks.

- [ ] **Step 1: Verify final commit and remote preconditions**

Run:

```bash
git status --short
git fetch origin main --no-tags
git log --oneline origin/main..HEAD
git tag --list v0.5.2
git push --dry-run origin main
```

Require a clean tracked worktree, expected implementation commits only, no pre-existing local tag `v0.5.2`, and a successful authenticated dry run. If `gh auth status` is invalid, authenticate before relying on `gh run`/`gh release` inspection; Git push credentials are checked separately by the dry run.

- [ ] **Step 2: Push all implementation commits to main**

Run: `git push origin main`

Verify the remote main SHA equals local `HEAD` before tagging.

- [ ] **Step 3: Create and push the annotated release tag**

Run:

```bash
git tag -a v0.5.2 -m "Release v0.5.2"
git push origin v0.5.2
```

This triggers `.github/workflows/release.yaml`.

- [ ] **Step 4: Wait for the release workflow rather than assuming dispatch is completion**

Use `gh run list --workflow release.yaml --branch v0.5.2 --limit 1 --json databaseId,headSha,status,conclusion` to obtain the numeric `databaseId`, verify `headSha` equals the tagged commit, and pass that exact integer to `gh run watch` with `--exit-status`. Verify, in order:

1. tagged commit is an ancestor of remote main;
2. tag equals package version;
3. non-full/non-online tests pass;
4. wheel and sdist build and pass Twine checks;
5. PyPI Trusted Publishing succeeds;
6. GitHub release creation succeeds.

If a job fails before PyPI publication, fix the cause without moving or reusing a published version. If PyPI 0.5.2 has published, any code correction requires a later version.

- [ ] **Step 5: Verify GitHub release and PyPI metadata**

Run:

```bash
gh release view v0.5.2 --json tagName,isDraft,isPrerelease,url,assets
```

Require tag `v0.5.2`, non-draft, non-prerelease, and both wheel and sdist assets. Verify PyPI reports 0.5.2 and hashes corresponding to the workflow artifacts.

- [ ] **Step 6: Verify a clean PyPI installation and pinned Hub access**

Create a new virtual environment under a new `/tmp/pyaging-0.5.2-*` directory, install only `pyaging==0.5.2` from PyPI, and run:

```python
import pyaging as pya

assert pya.__version__ == "0.5.2"
for name, count in {
    "dnamfitage": 1343,
    "dnamfitagegait": 111,
    "dnamfitagegrip": 183,
}.items():
    model = pya.pred.load_clock(name, verbose=False)
    assert len(model.features) == count
    assert model.version == "v0.5.2"
    assert model.metadata["version"] == "v0.5.2"

for retired in (
    "dnamfitagegaitf",
    "dnamfitagegaitm",
    "dnamfitagegripf",
    "dnamfitagegripm",
):
    try:
        pya.pred.load_clock(retired, verbose=False)
    except NameError as error:
        assert "pyaging 0.5.2" in str(error)
        assert "check PyPI" in str(error)
    else:
        raise AssertionError(f"retired clock unexpectedly loaded: {retired}")
```

Repeat with `PYAGING_DATA_REVISION=v0.5.2`. Run deterministic inference for the three current clocks using the oracle helper and require the pinned expected outputs.

For both live `main` and pinned `v0.5.2`, anonymously download aggregate
metadata and assert 177 keys, no retired keys, exact versions
`174 × v0.5.1` / `3 × v0.5.2`, and that the `v0.5.2` keys are exactly the
three changed clocks. Confirm the release audit shows mutations only in those
three per-clock repositories plus `lucascamillomd/pyaging-data`; the other 174
per-clock repositories must be unchanged.

- [ ] **Step 7: Report final immutable state**

Report the GitHub release link, PyPI 0.5.2 link, final Git SHA, the four exact
Hub tags (three changed repositories plus aggregate), confirmation that the
other 174 repositories were untouched, aggregate version counts
`174 × v0.5.1` / `3 × v0.5.2`, three verified replacement repositories, four
permanently deleted repository IDs, test/build results, and the clean-install
verification. Confirm `CHANGELOG.md` was not changed.
