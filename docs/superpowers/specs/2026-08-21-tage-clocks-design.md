# tAge Clocks (Tyshkovskiy et al., Nature 2026) — Design

**Date:** 2026-08-21
**Status:** Approved design, pending implementation plan
**Source paper:** Tyshkovskiy A, ..., Gladyshev VN. "Universal transcriptomic hallmarks of mammalian ageing and mortality." *Nature* 654:173–188 (2026). DOI: 10.1038/s41586-026-10542-3
**Upstream assets:** Zenodo record 18763485 (models, expression data), GitHub `Gladyshev-Lab/tAge` (reference pipeline)

## Goal

Add the paper's two flagship transcriptomic clocks to pyaging:

- **`tage`** — Multispecies Multi-Tissue ChronoAge, Elastic Net, Scaled (`_scaleddiff`) variant.
- **`tagemortality`** — Multispecies Multi-Tissue Mortality (log10 hazard ratio), Elastic Net, Scaled variant.

Bayesian Ridge variants (0.9–2.4 GB each) are out of scope. Other outcomes
(NormalizedAge), scopes (mouse-only, rodent tissue-specific), the YuGene
variants, and the module-specific clocks are deferred; this design's
preprocessing function must not preclude adding them later.

## Key constraints

1. **Cohort-relative clocks.** The published pipeline is cohort-level in three
   places: RLE size factors, per-gene standard scaling, and per-gene centering
   against a reference group (default: the median across all input samples).
   A single sample cannot be meaningfully predicted in isolation. pyaging's
   predict loop batches at 1024 samples, so cohort math must happen once,
   before batching — never inside `model.preprocess`.
2. **Mouse Entrez gene space.** The models are trained on mouse Entrez IDs;
   the reference pipeline maps rat/macaque/human input via orthologs, and
   accepts Symbol/Ensembl/Entrez ID types.
3. **License.** Models are under the MGB Open Access License 1.0 —
   non-commercial academic use only. The clocks are flagged research-only in
   metadata; pyaging itself remains MIT.

## Architecture (Approach A — approved; superseded, amended at user direction post-review)

> **Amendment (2026-08-21).** Approach A shipped as described below and was
> then replaced: the public `pa.pp.prepare_tage()` step is gone, and
> `predict_age` performs the cohort preprocessing itself. A model declares
> `cohort_transform = "tage"`; `predict_age` resolves that name against a
> registry of transforms, runs the transform over the whole input **once per
> call** (both tAge clocks share one cached frame), and builds each clock's
> `obsm["X_{clock}"]` from the transformed frame instead of from `adata.X` —
> same reference-value substitution and missing-feature bookkeeping as the
> ordinary path, which is why both share one implementation. The raw matrix is
> never mutated, so other clocks in the same call still see the original data,
> and all cohort math still happens before batching.
>
> The two arguments the public helper took are now read off the input, since
> there is no user call left to pass them to: the species from a 0/1 indicator
> column among `var_names` (`mouse`/`rat`/`macaque`/`human`, the mammalian-clock
> covariate idiom, dropped before the gene filter; absent or all-zero means
> mouse with a warning, two set or one varying between samples is an error), and
> the reference group from a boolean `obs["tage_reference_group"]` (absent means
> centre on every sample). `prepare_tage` survives as the private
> `_prepare_tage`, returning the transformed frame and stamping provenance into
> `uns["tage_preparation"]`.
>
> The `required_uns_flag` guard in "Error handling" below is therefore obsolete
> for these clocks: the failure mode it existed to catch — raw counts reaching a
> centred clock — can no longer happen. The attribute and its mechanism stay for
> any future clock whose contract cannot be satisfied automatically, and a
> declared `cohort_transform` supersedes it so that `.pt` files built before the
> change keep working. Parity against the reference fixtures is unchanged at
> ~1e-12.

### Component 1: `pa.pp.prepare_tage()`

New public function in `src/pyaging/preprocess/`:

```python
prepare_tage(adata, species, reference_group=None) -> AnnData
```

- `species`: one of `"mouse"`, `"rat"`, `"macaque"`, `"human"`.
- `reference_group`: optional boolean mask or list of `obs` names selecting
  control samples; when `None`, centering uses the per-gene median across all
  samples (the reference pipeline's default).

Steps, in order, matching `tAge_preprocessing()` in the reference repo:

1. Map input `var_names` (Symbol / Ensembl / Entrez for the given species) to
   mouse Entrez IDs via a shipped mapping table (see Component 3). Drop
   unmapped genes; collapse many-to-one mappings the same way the reference
   pipeline does.
2. RLE (relative log expression) normalization across the cohort.
3. Log transform.
4. Per-gene standard scaling (the `_scaleddiff` feature space).
5. Per-gene centering against the reference group (or overall median).

Returns a transformed AnnData with mouse Entrez `var_names` and stamps
`adata.uns["tage_prepared"] = True` (plus provenance details: species,
reference-group size, genes mapped/dropped). All cohort math happens here,
exactly once, so downstream batch size never affects predictions.

### Component 2: Clock models and notebooks

- Two notebooks in `clocks/notebooks/` (`tage.ipynb`, `tagemortality.ipynb`)
  following the v0.5.0 pattern: download the EN pickle from Zenodo, extract
  elastic-net coefficients and intercept, build pyaging's linear PyTorch
  model, write weights + metadata.
- Two model classes in `src/pyaging/models/_models.py` with identity
  `preprocess`/`postprocess`, except for a guard (see Error handling).
- Weights hosted on pyaging's HF repos like the other clocks, with the
  license field set to `MGB Open Access License 1.0 (non-commercial research
  only)`.
- `tage` prediction units: as reported by the reference pipeline (expected
  months in rodent space — confirm during conversion and record in metadata
  and boundary golds). `tagemortality` predicts log10 hazard ratio;
  `feature_units` recorded for expression inputs.

### Component 3: Ortholog / ID mapping table

The Symbol/Ensembl→Entrez and cross-species ortholog tables used by the
reference pipeline are extracted once in the conversion notebooks and shipped
alongside the clock weights on HF (downloaded lazily like other clock
assets). `prepare_tage` loads them at call time.

## Data flow

Superseded by the amendment above; the flow is now:

```
user AnnData (raw counts; optional species indicator column and
              obs["tage_reference_group"])
  → pa.pred.predict_age(adata, ["tage", "tagemortality"])
      → cohort transform, once per call → frame in mouse Entrez space
      → per clock: align features → predict
```

## Error handling

- `predict_age` on `tage`/`tagemortality` raises a clear error when
  `adata.uns["tage_prepared"]` is absent: raw counts must never silently
  reach a centered clock. Message points to `pa.pp.prepare_tage`.
- `prepare_tage` errors on: unknown `species`; a single-sample input
  (cohort centering is degenerate — documented limitation); an empty or
  fully-out-of-cohort `reference_group`.
- `prepare_tage` warns when mapped-gene overlap with the clock feature set
  falls below the thresholds the existing feature-matching warnings use.

## Testing

1. **Numerical parity (load-bearing):** run the reference Python pipeline and
   `prepare_tage` + pyaging predict on a subset of the published rodent
   expression data from Zenodo; require agreement to ≤1e-6 (tighter if the
   pickled floats allow) for both the preprocessed matrix and the final
   predictions of both clocks, with and without an explicit reference group.
2. **Standard suite:** random-input gold tests against live HF weights,
   boundary golds, metadata validation (including the research-only license
   field).
3. **Unit tests for `prepare_tage`:** each species mapping path, Symbol vs
   Ensembl vs Entrez inputs, reference-group vs all-sample centering,
   single-sample error, unknown-species error, low-overlap warning, uns
   marker stamping.
4. **Guard test:** `predict_age` without `prepare_tage` raises the intended
   error.

## Out of scope

- Bayesian Ridge model variants.
- NormalizedAge, YuGene, single-tissue, mouse-only, and module-specific
  clocks (the `prepare_tage` design deliberately supports them later).
- TACO web-tool integration.
- Any commercial-use licensing arrangement with MGB.
