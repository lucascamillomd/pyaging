# Design: Add missing "easy" clocks from methylCIPHER, biolearn, and OmniAge

**Date:** 2026-07-04
**Author:** Lucas Camillo (with Claude)
**Status:** Approved — proceeding to implementation plan

## Goal

Add the aging clocks present in three external packages but missing from
pyaging, restricted to those that are "easy" to implement — i.e. pure linear
models (`LinearModel`), PC-linear models (`PCLinearModel`), or clocks that
reuse existing pyaging infrastructure (mean-beta / percentile / TNSC / anti-log
/ sigmoid transforms). Package priority order (most → least important):

1. https://github.com/HigginsChenLab/methylCIPHER  (R)
2. https://github.com/bio-learn/biolearn          (Python, PyPI)
3. https://github.com/Duzhaozhen/OmniAge          (R + Python monorepo)

Naming convention: **all lowercase, no spaces, no hyphens** (matches existing
pyaging clocks).

## Scope: 37 clocks

Confirmed decisions:
- Implement all 34 "easy" + 3 "medium" (new-transform) clocks = **37 total**.
- **Cross-validate every clock against its source package** (accuracy gate).
- **Include everything**: single-CpG markers and disease/trait predictors count
  (pyaging already ships trait predictors such as McCartney BMI/smoking).
- **Checkpoint after the first 3 clocks** (full end-to-end incl. S3 upload +
  gold standard), then continue autonomously through the rest.
- After all clocks pass: bump package to **v0.2.0**, commit + push, cut a
  GitHub **release** listing the new clocks.

### Conflict-resolution rule
When a clock exists in multiple packages with differing coefficients or output
units (e.g. Weidner, or gestational days-vs-weeks), implement against and
cross-validate against the **highest-priority package** that contains it
(methylCIPHER > biolearn > OmniAge). Record the chosen source in the clock's
metadata `notes`.

## The per-clock workflow (repeatable recipe)

For each clock, in order:

1. **Model class** — add a small class to `pyaging/models/_models.py` following
   existing style (79-char headers, PascalCase). Most are the standard pattern:
   `preprocess` imputes missing features from `reference_values`, base model is
   linear, `postprocess` is identity. Reuse existing classes/transforms where
   noted in the table below.
2. **Notebook** — create `clocks/notebooks/<name>.ipynb` from `template.ipynb`:
   download coefficients from the source package (curl / git clone / Rscript,
   mirroring existing notebooks), build the base model, set metadata (citation,
   DOI, year, species, notes), `features`, weights, intercept, `reference_values`,
   `preprocess_name`/`postprocess_name`, then `torch.save` to
   `../weights/<name>.pt`.
3. **Source cross-validation** — in an isolated scratchpad environment, run the
   original package on the SAME random full-feature beta matrix and assert the
   pyaging output matches within tolerance (default 1e-3 relative, looser if the
   source uses float32 truncation). This is the correctness gate.
   - methylCIPHER: `Rscript` calling `calc*` functions.
   - biolearn: `pip install biolearn`; use `model_gallery`.
   - OmniAge: `pip install` OmniAgePy from source (`OmniAgePy/pyproject.toml`).
4. **Execute notebook** — `jupyter nbconvert --execute --inplace` to regenerate
   the `.pt`.
5. **Gold standard** — run pyaging's test harness (seed 42, 1/3 feature dropout,
   constant imputation) to get the deterministic prediction, add it to
   `gold_standard_dict` in `tests/predict/test_gold_standard.py`.
6. **Merge metadata** — regenerate `clocks/metadata/all_clock_metadata.pt` via
   `clocks/update_all_clocks.py` (run for the new version; it stamps version and
   merges metadata for ALL weights — safe, does not re-execute notebooks).
7. **S3 upload** — `aws s3 cp` the new `<name>.pt` to
   `s3://pyaging/clocks/weights0.1.0/` and sync the updated
   `all_clock_metadata.pt` to `s3://pyaging/clocks/metadata0.1.0/`. (The `0.1.0`
   in the S3 path is a data-schema version, independent of the package version.)

Two independent validations exist and must not be conflated:
- **Correctness** (step 3): pyaging vs. source package, full feature set.
- **Reproducibility** (step 5): pyaging's own regression gold standard.

## Model classes / transforms required

| Pattern | Clocks | Reuse |
|---|---|---|
| Standard linear + reference imputation, identity postprocess | mccartneyalcohol, weidner, vidalbralo, garagnani, bocklandt, dnamfili, dnamstress, cellpopage, replitalinorm, senchronoage, sencultureage, senmortalityage, dunedinpoam38, depressionbarbu, reedbmi, downsyndrome, prostatecancerkirby, hepatoxu, mccartneytotalhdlratio, mccartneywhr, compil6, neusin, gliasin, hep, ctsliver, ensembleagehumanmouse | new class each, `DNAmPhenoAge` pattern |
| Linear + anti-log-linear postprocess (adult_age=20) | corticalclock | `SkinAndBlood` postprocess |
| Linear + gestational unit postprocess | bohlin, epicga, mayne | identity or ÷7, set by source |
| 95th-percentile score | stemtocvitro | `stemTOC` logic |
| Mean-beta score | epicmithyper (mean), epicmithypo (1−mean) | `epiTOC1` logic |
| TNSC mitotic (delta/beta0) | epitoc3 | `epiTOC2` logic |
| Linear + sigmoid postprocess | cvdwesterman (sigmoid), adbahadosingh (sigmoid + offset) | `torch.sigmoid` (McCartney) |
| Linear + anti-log postprocess (adult_age=48, ÷12) | wu | new postprocess variant |

`depressionbarbu` already has a class in `_models.py` (currently unused) — only
the notebook + weights are missing.

## Ranked clock list (implementation order)

### Tier A — methylCIPHER (16)
| # | name | model | postprocess | ~#feat | citation / DOI |
|---|---|---|---|---|---|
| 1 | mccartneyalcohol | linear | identity | 450 | McCartney 2018, 10.1186/s13059-018-1514-1 |
| 2 | weidner | linear | identity | 3 | Weidner 2014, 10.1186/gb-2014-15-2-r24 |
| 3 | vidalbralo | linear | identity | 8 | Vidal-Bralo 2016, 10.3389/fgene.2016.00126 |
| 4 | garagnani | linear (1 CpG ELOVL2) | identity | 1 | Garagnani 2012, 10.1111/acel.12005 |
| 5 | bocklandt | linear (1 CpG) | identity | 1 | Bocklandt 2011, 10.1371/journal.pone.0014821 |
| 6 | dnamfili | linear (frailty) | identity | 20 | Li 2022, 10.1038/s41467-022-32893-x |
| 7 | dnamstress | linear | identity | 211 | Jung 2023, 10.1016/j.biopsych.2022.06.036 |
| 8 | cellpopage | linear | identity | 42 | Lujan 2024, PMID 38956711 |
| 9 | replitalinorm | linear | identity | 218 | Endicott 2022, PMID 36347867 |
| 10 | senchronoage | linear | identity | 187 | Kasamoto 2026, PMID 41746138 |
| 11 | sencultureage | linear | identity | 142 | Kasamoto 2026, PMID 41746138 |
| 12 | senmortalityage | linear | identity | 91 | Kasamoto 2026, PMID 41746138 |
| 13 | corticalclock | linear | anti-log(20) | 347 | Shireby 2020, 10.1093/brain/awaa334 |
| 14 | bohlin | linear | gestational | 251 | Bohlin 2016, 10.1186/s13059-016-1063-4 |
| 15 | mayne | linear | gestational | 62 | Mayne 2017, 10.2217/epi-2016-0103 |
| 16 | dunedinpoam38 | linear | identity | 46 | Belsky 2020, 10.7554/eLife.54870 |

### Tier B — biolearn (5)
| # | name | model | ~#feat | citation / DOI |
|---|---|---|---|---|
| 17 | depressionbarbu | linear (class exists) | 195 | Barbu 2020, 10.1038/s41380-020-0808-3 |
| 18 | reedbmi | linear | 134 | Reed 2020, 10.1186/s13148-020-00841-5 |
| 19 | downsyndrome | linear classifier | 652 | 10.1038/s41467-021-21064-z |
| 20 | prostatecancerkirby | linear classifier | 3 | Kirby 2017, 10.1186/s12885-017-3252-2 |
| 21 | hepatoxu | linear (cfDNA HCC) | 9-10 | Xu 2017, 10.1038/nmat4997 |

### Tier C — OmniAge (13)
| # | name | model | postprocess | ~#feat | citation / DOI |
|---|---|---|---|---|---|
| 22 | mccartneytotalhdlratio | linear | identity | 413 | McCartney 2018, 10.1186/s13059-018-1514-1 |
| 23 | mccartneywhr | linear | identity | 227 | McCartney 2018, 10.1186/s13059-018-1514-1 |
| 24 | compil6 | linear | identity | 36 | 10.1093/gerona/glab046 |
| 25 | epicga | linear | gestational | 176 | Haftorn 2021, 10.1186/s13148-021-01055-z |
| 26 | neusin | linear (raw beta, neuron) | identity | 673 | Tong 2024, 10.18632/aging.206184 |
| 27 | gliasin | linear (raw beta, glia) | identity | 221 | Tong 2024, 10.18632/aging.206184 |
| 28 | hep | linear (raw beta, hepatocyte) | identity | 71 | Tong 2024, 10.18632/aging.206184 |
| 29 | ctsliver | linear (raw beta, liver) | identity | 91 | Tong 2024, 10.18632/aging.206184 |
| 30 | ensembleagehumanmouse | linear | identity | 1 subclock | Haghani 2025, 10.1007/s11357-025-01808-1 |
| 31 | stemtocvitro | 95th percentile | identity | 629 | Zhu 2024, 10.1038/s41467-024-48649-8 |
| 32 | epicmithyper | mean beta | identity | subset 1348 | Duran-Ferrer 2020, 10.1038/s43018-020-00131-2 |
| 33 | epicmithypo | 1 − mean beta | identity | subset 1348 | Duran-Ferrer 2020, 10.1038/s43018-020-00131-2 |
| 34 | epitoc3 | TNSC mitotic | identity | 170 | Teschendorff 2020, 10.1186/s13073-020-00752-3 |

### Tier D — medium / new transform (3)
| # | name | model | postprocess | ~#feat | citation / DOI |
|---|---|---|---|---|---|
| 35 | wu | linear | anti-log(48) ÷12 | 111 | Wu 2019, 10.18632/aging.102399 |
| 36 | cvdwesterman | linear | sigmoid | 235 | Westerman 2020, 10.1161/JAHA.119.015299 |
| 37 | adbahadosingh | linear | sigmoid + offset | 4 | Bahado-Singh 2021, 10.1371/journal.pone.0248375 |

## Reference (default beta) values

Where the source package ships gold-standard / training-mean beta values for
missing-CpG imputation (e.g. corticalclock's `ref_mean`, cellpopage's 2543-CpG
reference, dnamstress reference), store them in `model.reference_values` so
pyaging can impute missing features. Where the source provides none, set
`reference_values = None` (matching e.g. `hannum`).

## Verification harness

Isolated scratchpad env (never touches pyaging's deps):
- Python venv with `biolearn` + OmniAgePy for Python cross-validation.
- R 4.3.1 (already installed) with methylCIPHER for R cross-validation.
- Shared random beta matrix per clock (fixed seed) fed to both source and
  pyaging; assert element-wise match within tolerance.

## Release steps (after all 37 pass)

1. `make version VERSION=v0.2.0` (updates `pyproject.toml` + `pyaging/__init__.py`).
2. Regenerate `all_clock_metadata.pt` at v0.2.0; final S3 sync.
3. `make lint format` clean; full `test` gold-standard suite green.
4. Commit + push to `main`.
5. `gh release create v0.2.0` with notes listing the newly added clocks.

## Out of scope (excluded)

Need major new infrastructure or population-level standardization incompatible
with pyaging's per-sample forward model:
- **Nonlinear / new modality**: MiAge (per-sample optimizer), GP-age ×6 (GPy),
  OrganAge ×4+ (Olink/SomaScan proteomic), Bernabeu cAge (hybrid quadratic),
  Peters transcriptomic (quantile-norm pipeline), single-cell transcriptomic
  clocks (scImmuAging, BrainCTClock, ScAgePolyakClock, BuckleyMouseSVZ),
  DNAmCTFClock (PMML/Java), HurdleInflammAge (proprietary remote API).
- **Cross-sample normalized** (population-dependent): CompCRP, CompCHIP,
  CompSmokeIndex, EpiScores, CTS-intrinsic (Neu-In/Glia-In/Brain),
  EnsembleAgeDynamic (50 sub-clocks), PASTA CT46.

These can be revisited later as separate, larger efforts.
