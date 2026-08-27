# DNAmFitAge Composite Clocks Design

**Date:** 2026-08-27
**Target release:** pyaging 0.5.2

## Purpose

Simplify the public DNAmFitAge clock family while preserving the published
sex-specific calculations. Replace the separate female and male gait and grip
clocks with two sex-gated clocks, and make DNAmFitAge calculate original
DNAmGrimAge internally from methylation values, chronological age, and sex.

This release intentionally makes a breaking interface change in a patch
version. The four former clock names are removed without compatibility aliases.

## Public interface

The release adds these clocks:

- `dnamfitagegait`: predicts gait speed in metres per second from the union of
  the female and male model CpGs plus `female`.
- `dnamfitagegrip`: predicts maximum handgrip strength in kilograms from the
  union of the female and male model CpGs plus `female`.

The release changes `dnamfitage` so that its public inputs are the union of the
CpGs needed by gait, grip, VO2max, and original GrimAge, plus `age` and
`female`. A caller no longer calculates or supplies a `grimage` column.

The release removes these public clock names, classes, weight files, conversion
notebooks, metadata entries, generated documentation entries, and Hugging Face
repositories:

- `dnamfitagegaitf`
- `dnamfitagegaitm`
- `dnamfitagegripf`
- `dnamfitagegripm`

No aliases or deprecation period will be provided, and `CHANGELOG.md` will not
be modified for this change.

## Model architecture

### Gait and grip

Add `DNAmFitAgeGait` and `DNAmFitAgeGrip` classes. Each serialized model owns
both original sex-specific linear regressions, the ordered union of their CpG
features, the index mapping for each regression, and model-specific reference
values.

For a sample with a binary `female` value, the merged model must reproduce the
corresponding former model:

- `female = 0` returns the former male-model prediction.
- `female = 1` returns the former female-model prediction.

The models use numeric blending between the two branch results rather than
boolean masks that can leave an output unassigned:

```text
prediction = male_prediction + female * (female_prediction - male_prediction)
```

This is exact at 0 and 1. Consistent with the current GrimAge input style,
supplied numeric values outside `{0, 1}` are not rejected and therefore
interpolate or extrapolate. A supplied NaN propagates unless preprocessing
replaces it. If the `female` feature is absent, normal feature alignment emits
the existing missing-feature warning and supplies the stored reference value
`1.0`.

Each branch applies its own reference values after selecting its feature
subset. This prevents a shared CpG from being imputed with the wrong value if
the female and male source models store different references.

To preserve that distinction through the existing feature-alignment layer,
the composite model's public reference vector uses NaN sentinels for CpGs and
concrete references only for public covariates such as `female` and `age`.
Alignment therefore leaves an absent CpG as NaN, and the selected embedded
component replaces it with that component's own reference immediately before
evaluation.

### DNAmFitAge

The revised `DNAmFitAge` serialized model is self-contained. It owns:

- the new gated gait component;
- the new gated grip component;
- the existing DNAmFitAge VO2max component;
- the original GrimAge model, not GrimAge2;
- the existing female and male Klemera-Doubal final regressions;
- feature-index mappings and component-specific reference values.

Its forward pass calculates gait, grip, VO2max, and original GrimAge from the
same input matrix. It then applies the existing sex-specific standardization
constants and Klemera-Doubal equations. The final female and male results use
the same numeric blending rule as the merged fitness clocks.

No nested download or call to `load_clock` occurs during inference. Embedding
GrimAge duplicates its parameters in `dnamfitage.pt`, but makes one Hub download
sufficient and keeps inference deterministic and offline after download.

The outer feature list ends with `female` and `age`, with indices stored
explicitly rather than inferred solely from positional conventions. Component
inputs are filled with their component-specific reference values before the
component runs. Missing `female` follows the GrimAge convention (`1.0`), and
missing `age` follows the GrimAge convention (`65.0`), with the existing
missing-feature warning and bookkeeping in `adata.uns`.

## Conversion notebooks and weights

Replace the four former gait and grip notebooks with:

- `clocks/notebooks/dnamfitagegait.ipynb`
- `clocks/notebooks/dnamfitagegrip.ipynb`

The notebooks continue to obtain the published coefficients and sex-specific
medians from the authors' `kristenmcgreevy/DNAmFitAge` repository. Each notebook
constructs both branch regressions, their feature union and mappings, reference
data, metadata, and the final serialized model.

Update `clocks/notebooks/dnamfitage.ipynb` to construct the self-contained
model from the published fitness equations and the retained local
`clocks/weights/grimage.pt` artifact. The notebook records enough assertions to
detect missing or misordered component features during conversion.

The resulting current weight set contains:

- `clocks/weights/dnamfitagegait.pt`
- `clocks/weights/dnamfitagegrip.pt`
- an updated `clocks/weights/dnamfitage.pt`

The four former local weight files are deleted.

## Metadata and documentation

Replace the four former entries in the curated clock metadata registry with
entries for `dnamfitagegait` and `dnamfitagegrip`. Update the `dnamfitage`
metadata to describe internal original-GrimAge calculation and the new public
feature contract. Generated feature counts must come from the serialized
models, not hand-maintained estimates.

Regenerate aggregate metadata, the clock glossary CSV and JSON, HTML clock
notebooks, and the Sphinx clock implementation index. Remove all generated
references to the four retired names. Update user-facing examples or prose only
where searches show the retired names or the external `grimage` input contract.

## Unavailable-clock error

Keep model lookup network-free beyond the existing Hugging Face download
attempt. When a clock cannot be found, the raised error must:

- retain the requested clock name and the existing clock-catalogue guidance;
- report the installed pyaging version;
- state that the clock may require a newer release;
- advise the user to check PyPI and upgrade if a newer version is available.

The error must not query PyPI, add startup latency, or fail differently while
offline. Tests patch the download boundary and assert the actionable message.

## Verification

Before deleting the former local weights, generate deterministic oracle cases
for both sexes. Preserve expected numeric outputs in tests, not the obsolete
models themselves.

Tests must establish:

1. `dnamfitagegait` at `female=0` equals the former male gait model and at
   `female=1` equals the former female gait model.
2. `dnamfitagegrip` similarly equals the two former grip models.
3. The revised self-contained `dnamfitage` equals the former workflow in which
   original GrimAge is calculated first and supplied to DNAmFitAge.
4. Equivalence holds for deterministic ordinary inputs and boundary/reference
   inputs within the project's established floating-point tolerances.
5. Missing `female` and `age` use reference values and retain existing warning
   and `adata.uns` bookkeeping behavior.
6. Non-binary numeric `female` values follow the documented blending equation,
   and supplied NaN behavior is pinned.
7. Removed or unknown clock names produce the revised unavailable-clock error.
8. New weights serialize and load on CPU, expose correct metadata and feature
   order, and run through `predict_age`.
9. The standard non-online suite, full-catalog gold tests, conversion notebooks,
   tutorial tests where applicable, documentation build, package build, and
   distribution metadata checks pass.

## Hugging Face migration

Upload and verify these repositories first:

- `pyaging/dnamfitagegait`
- `pyaging/dnamfitagegrip`
- updated `pyaging/dnamfitage`

Verification uses fresh Hub downloads rather than only local files. After all
three replacements pass inference, permanently delete these four Hugging Face
repositories, including their commit history and tags:

- `pyaging/dnamfitagegaitf`
- `pyaging/dnamfitagegaitm`
- `pyaging/dnamfitagegripf`
- `pyaging/dnamfitagegripm`

Before each deletion, verify the authenticated Hugging Face account and exact
repository ID. The deletion is intentionally irreversible and has already been
authorized by the project owner.

Remove the same four weight files from the `main` branch of the legacy
`lucascamillomd/pyaging-data` repository so fallback loading cannot restore the
retired names. Do not rewrite or delete that aggregate repository's history or
old tags because they contain every other clock.

Tag all surviving per-clock repositories and the aggregate repository at
`v0.5.2`. Verify that the three new artifacts load when
`PYAGING_DATA_REVISION=v0.5.2` and that retired names fail through the intended
error path.

## GitHub and PyPI release

The target package version is `0.5.2`. Publication proceeds in this order:

1. Complete local implementation, artifact generation, and all verification.
2. Upload and verify replacement Hugging Face artifacts.
3. Delete obsolete Hub repositories and current aggregate fallback files.
4. Tag the surviving Hub repositories and aggregate repository `v0.5.2` and
   verify pinned resolution.
5. Commit the complete package change and push `main` to GitHub.
6. Create and push annotated Git tag `v0.5.2`.
7. Let `.github/workflows/release.yaml` verify the tag, build distributions,
   publish through PyPI Trusted Publishing, and create the GitHub release.
8. Wait for the workflow and verify the GitHub release, PyPI version, and a
   clean installation's access to the replacement clocks.

The Hub deletion step and PyPI publication are irreversible boundaries. A
failure before either boundary is fixed before advancing. A failure after PyPI
publishes requires a subsequent release because PyPI distributions are
immutable.
