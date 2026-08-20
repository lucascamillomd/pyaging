# Changelog

Notable changes for each release. Versions follow [semantic versioning](https://semver.org/).

## 0.5.0

### Breaking change

- **`phenoage` now takes `c_reactive_protein` in mg/dL instead of the pre-logged `log_crp`.**
  Levine's natural log has moved inside the clock, so pass CRP exactly as the assay reports
  it and delete any `np.log(...)` you were applying to it. This lets a single raw CRP column
  feed clocks that transform it differently — the BioAge-derived clocks below use `log1p`,
  not `ln`. If you keep passing a pre-logged value you will see an out-of-range warning,
  because a logged CRP is usually negative while the expected raw range starts at
  0.01 mg/dL.

  The lowercase `age` and `female` feature names are *not* a change in this release: Hugging
  Face has served those spellings all along, and no migration is needed for them.

### New clocks

The catalogue grows from 173 to 177 clocks, all four clinical:

- `kdmage` — Klemera-Doubal biological age, fit sex-specifically on NHANES III via the R
  package [BioAge](https://doi.org/10.1007/s11357-021-00480-5) (Kwon & Belsky,
  *GeroScience* 2021). Output in years.
- `homeostaticdysregulation` — Mahalanobis distance from a young, healthy NHANES III
  reference cohort, fit sex-specifically, from the same source. **The output is a log
  dysregulation score, not an age in years**; larger values mean greater dysregulation.
- `phenoagesaopaulo` — PhenoAge refit on NHANES III without creatinine, albumin, and
  alkaline phosphatase, with its own refit Gompertz mortality-to-age constants. Pooled
  across sexes, so it takes no `female` column.
- `linage2` — 59-feature principal-component clinical clock (Fong et al., *npj Aging* 2025),
  drawing on 85 laboratory and questionnaire inputs. Reproduces the published example
  subjects to within 0.005 years.

### Out-of-range input warnings

`predict_age` now checks each clock's input against a package-wide registry of per-feature
units and broad plausibility bounds
([`src/pyaging/data/feature_ranges.json`](src/pyaging/data/feature_ranges.json)). Values
outside those bounds almost always mean wrong units or swapped columns, so the check names
the feature, its expected range and unit, the share of values affected, and the observed
minimum and maximum.

The check is **warn-only**. It never blocks a prediction, never modifies your data, and
makes no claim about clinical abnormality — the bounds are deliberately far wider than any
reference interval. Missing values are ignored; those are already reported separately.

Two public helpers expose the same registry:

- `pya.utils.get_feature_ranges(clock_name)` returns a clock's features with their units and
  bounds as a DataFrame.
- `pya.utils.resolve_feature_ranges(features, data_type, feature_units=None)` resolves an
  arbitrary feature list.

Every clock now carries a `feature_units` attribute, and every clock notebook documents the
ranges its features are expected to fall in.

### Fixes

- Locally built wheels were missing the `pyaging.data` subpackage: an unanchored `data/`
  pattern in `.gitignore` matched it anywhere in the tree.
- `clocks/update_all_clocks.py` read a `clock.version` attribute that no clock defines,
  which had made the aggregate clock metadata impossible to regenerate.
