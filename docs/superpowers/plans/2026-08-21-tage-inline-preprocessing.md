# tAge Inline Preprocessing Plan (follow-up)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox syntax.

**Goal:** Remove the separate `pa.pp.prepare_tage` user step: `predict_age` runs the tAge cohort pipeline automatically for the tage/tagemortality clocks, with species given as mammalian-clock-style 0/1 indicator columns and reference group defaulting to all samples; update the transcriptomics tutorial to cover the new clocks.

**Spec:** docs/superpowers/specs/2026-08-21-tage-clocks-design.md (amended by this plan: Approach A's public helper is replaced by an in-predict cohort hook at the user's direction).

## Global Constraints

- No behavior change to any other clock; raw `adata.X` is never mutated by the tAge transform (other clocks in the same call see original data).
- Cohort math still runs exactly once per predict_age call (shared between tage and tagemortality when both are requested), before batching; batch-size independence preserved.
- All existing parity numbers must hold unchanged (fixtures are untouched); tests re-target the new entry path.
- Species indicators: optional var columns named `mouse`, `human`, `rat`, `macaque` valued 0/1. Exactly one species column set to 1 (uniform across samples) selects the species; all absent/all-zero → default `mouse` with a logged warning; mixed values across samples or multiple species set → ValueError. Indicator columns are excluded from the gene pipeline.
- Reference group: optional boolean column `adata.obs["tage_reference_group"]`; absent → all samples (default None semantics).
- `prepare_tage` becomes private `_prepare_tage` (removed from `pyaging.preprocess.__all__` and docs); the guard attribute `required_uns_flag` infrastructure stays but TAge/TAgeMortality no longer set it (preprocessing is automatic; the failure mode it guarded no longer exists).
- Notebooks re-executed so shipped .pt carries `cohort_transform = "tage"`, no `required_uns_flag`, and metadata notes describing the species-column and reference-column idioms; registry/evidence/docs stay in sync (the same 23-field parity as before).
- Full suite green, ruff clean, `uv run pytest`.

### Task A: In-predict cohort transform + API change

**Files:** src/pyaging/preprocess/_tage.py, src/pyaging/preprocess/__init__.py, src/pyaging/models/_models.py, src/pyaging/models/_base_models.py (only if a new declared attribute needs a default), src/pyaging/predict/_pred.py and/or _pred_utils.py, tests/preprocess/test_prepare_tage.py (rework), tests/integration/test_tage_end_to_end.py (rework), tests/predict/test_required_uns_flag.py (trim tAge-specific parts, keep generic guard tests), tests/test_public_api.py, docs/source/pyaging.preprocess.rst, docs/superpowers/specs/2026-08-21-tage-clocks-design.md (amend).

Steps: TDD the species-indicator resolver (one-hot detection, default-mouse warning, mixed/multiple errors, indicator exclusion from gene pipeline); TDD the predict-path hook (model attribute `cohort_transform: str | None = None`; when "tage", predict_age computes the transformed cohort frame once per call — cache shared across the clock loop — and this clock's `X_{clock}` matrix is built from the transformed frame with missing model features taking reference_values); rework e2e tests to call predict_age directly on raw counts (genes×samples fixture transposed, species column added) asserting the SAME expected_predictions values; keep batch-size independence and obs-metadata tests; privatize _prepare_tage; amend spec doc section "Architecture" to describe the in-predict flow.

### Task B: Notebooks, registry, tutorial, docs

**Files:** clocks/notebooks/{tage,tagemortality}.ipynb (attribute + metadata notes; re-execute; parity cells must pass), registry via the established flow (clock_metadata.json notes fields, evidence untouched unless notes are audited fields — follow validate_metadata), docs/source/pyaging.preprocess.rst (rewrite the cohort-relative section: species columns + reference column + automatic preprocessing), the transcriptomics tutorial notebook under docs/tutorials or tutorials/ (add a tage+tagemortality section on the bundled example data: default run, reference-group run, research-only note), docs rebuild + catalogue check.
