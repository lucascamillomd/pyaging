# Changelog

## 0.6.3

This package release uses existing model weights and Hugging Face files. It does
not create a `v0.6.3` data tag.

Changes that can alter predictions:

- epiTOC2 and EpiTOC3 average over the CpGs a sample has, as the authors' R
  code does. A missing CpG was scored as beta = -1, which reversed the sign of
  the estimate when about 10% of CpGs were absent.
- Clocks that read `age` or `female` (GrimAge, GrimAge2 and its proxies,
  PCGrimAge, DNAmFitAge, CpGPTGrimAge3, clinical PhenoAge and others) raise a
  `ValueError` when those columns are not features. They were replaced by a
  reference value or zero, so every sample was scored at the same age and sex.
- A NaN inside a supplied column is filled like an absent feature (a finite
  reference value, else a packaged CpG median) or left as NaN. Rank-based
  clocks such as DunedinPACE read it as the highest value. The share of such
  cells is stored in `adata.uns["<clock>_percent_nan_values"]` and reported
  as a `UserWarning`.
- A sample with no observed values is no longer imputed into a cohort-average
  sample. It stays NaN through `df_to_adata` and receives NaN predictions.
- DunedinPACE, Stubbs and `quantile_normalize_with_gold_standard` resolve
  tied values as `preprocessCore::normalize.quantiles.use.target` does.
  Results on rounded beta values no longer depend on sort order or device.
- Mammalian2/3 and their blood and skin variants raise a `ValueError` when a
  sample has no species column set, instead of using the first species.
- `bigwig_to_df` reads bigWigs with Ensembl chromosome names (`1`, `X`), which
  returned all zeros. A file with none of the gene chromosomes raises a
  `ValueError`; genes on a chromosome the file lacks are NaN with a warning.

Fixes:

- Preserve metadata columns in `epicv2_probe_aggregation`.
- `predict_age` accepts AnnData opened with `backed="r"`, ignores whitespace
  and repeated clock names, and leaves no `X_<clock>` matrix behind when a
  clock fails.
- `find_clock_by_doi` matches bare DOIs, `doi:` prefixes and DOI URLs in any
  letter case. `get_clock_metadata` reports an unknown clock without raising
  `KeyError`.
- An empty `PYAGING_DATA_REVISION` means the default revision. Offline, clock
  weights cached only in the shared data repository are found. Malformed
  clock names report "clock not available".
- `df_to_adata` accepts a single column name for `metadata_cols`.
- `pyaging.utils.download` takes a `timeout` and no longer hangs on a stalled
  server. The clock metadata file loads with `torch.load(weights_only=True)`.
- BiTAge and `binarize` no longer overwrite their input. `Logger` no longer
  reconfigures loggers it did not create.

Performance:

- `imputer_strategy="knn"` reproduces scikit-learn's `KNNImputer` about 60
  times faster with a third of the peak memory. Exactly tied neighbours are
  ordered by sample position.
- Clock weights and metadata are checked against the Hub once per process and
  revision, and several clocks are resolved concurrently. A long-running
  process sees a new push to `main` after a restart or `ClockCache.clear()`.
- DunedinPACE, Stubbs, Pasta, Reg, stemTOC and BiTAge preprocess whole batches
  at once. SystemsAge no longer copies its 4 GB rotation on every batch.
- `import pyaging` no longer loads scikit-learn or statsmodels, and
  `epicv2_probe_aggregation` uses about a tenth of the peak memory.

Release tooling:

- `make release` pushes the commit before publishing weights and pushes the
  tag last, after a preflight check of the branch, version and tag. The
  release workflow runs the tutorials before publishing to PyPI.

## 0.6.2

- Use the [Kuo et al. (2021) equation](https://doi.org/10.1111/acel.13376) for
  clinical PhenoAge. The leading 1 in the 2019 erratum compressed the age
  scale in 0.6.1. The biomarker coefficients follow Kuo et al. (2021).

## 0.6.1

- Apply the [2019 correction](https://doi.org/10.1371/journal.pmed.1002760)
  to the clinical PhenoAge coefficients and mortality-to-age equation.
  The equation as printed has a lower limit of 141.50 years.

## 0.6.0

- Fill absent CpGs for eligible human DNA methylation clocks with beta
  references from Hannum 450K (GSE40279) and BoA EPIC v2 (GSE246337).
  Shared CpGs use the mean of the two cohort medians. A clock's own reference
  vector takes precedence, and uncovered features fall back to zero.

## 0.5.5

- Add Kuo et al.'s PAC from original-author code and coefficients, with 128
  Olink NPX proteins plus chronological age and mortality-equivalent output
  in years. Required inputs are checked before generic missing-feature filling.
- Remove the 44 reduced-panel Olink Explore 1536 OrganAge entries from the
  current catalogue. Retain 46 full-panel models and shorten their names by
  dropping `olink3000`; coefficients and raw output scales are unchanged.
- Specify plasma training material, assay normalization, protein identifiers,
  missing-data behavior and model-specific feature units for proteomic clocks.
  Add a [proteomic input guide](tutorials/tutorial_proteomics.ipynb).
- Audit all six clocks in the 2026 Insilico trial. ProtAge lacks public trained
  weights/reference preprocessing; ipfP3GPT is restricted to UK Biobank RAP;
  PAOPAC provides only opaque Windows inference assets. These three are not
  added as executable clocks; see the [source audit](docs/development/proteomic-source-audit-0.5.5.md).
- Preserve committed catalogue data during documentation builds, avoiding stale
  remote metadata overwriting newly released entries.

## 0.5.4

- Add CellDRIFT, MiAge, PCBrainAge, and the original 370-CpG `lambda.1se`
  IntrinClock model. The existing 380-CpG `intrinclock` is unchanged.
- Add the original 55-CpG PRC and 458-CpG non-PRC PhenoAge contributions,
  without an age intercept or independent recalibration.
- Expose all eight PCGrimAge smoking/protein proxies as standalone predictors.
- Add 90 proteomic OrganAge outputs from the authors’ recommended fold-one
  chronological/mortality models for Olink 3000 and 1500 panels.
- Preserve author preprocessing, output units, source provenance, and licenses.
  MiAge uses the authors’ five-start bounded optimization on the CPU.
- Validate the additions against native R author-code fixtures, including missing
  features and batching; publish the new artifacts and data snapshot `v0.5.4`.

See [release validation](docs/development/release-0.5.4.md) for source and
comparison details.

## 0.5.3

This package release uses existing model weights and Hugging Face files. It does
not create a `v0.5.3` data tag.

- Add an optional bounded `pya.pred.ClockCache` for repeated prediction across
  datasets, with separate entries for device and data revision.
- Speed up feature matching, EPIC v2 probe aggregation, quantile normalization,
  and bigWig signal assembly.
- Support sparse prediction input and keep aligned matrices on the CPU before
  transferring each inference batch. Protect retained inputs from model transforms.
- Preserve metadata for numeric, datetime, and hierarchical sample identifiers;
  return writable arrays with independent original and imputed snapshots.
- Correct EPIC v2 singleton probe names and repeated-label aggregation; validate
  ambiguous feature names, invalid batch sizes, imputation strategies, and tAge inputs.
- Publish local downloads atomically so interrupted files cannot become cache hits.
  Preserve HTTP error diagnostics and fall back to the same data revision when a
  per-clock repository lacks the requested tag.
- Keep logging and Hugging Face progress settings isolated from other libraries.
- Make Pasta/Reg missing-value filling independent of batch size and vectorize
  their tied ranks. Apply PCGrimAge's existing reference imputation in its forward
  pass. Preserve dtype/device for all-missing mitotic-clock inputs.
- Evaluate the PhenoAge mortality-to-age links in log space to avoid artificial
  infinities at extreme inputs. Coefficients and mathematical formulas are unchanged.
- Stop clock notebook builds on failure and make version updates portable across
  macOS and Linux. Derive the default release version from the package.

See [the audit report](docs/development/audit-0.5.3.md) for findings, validation,
benchmarks, and follow-up recommendations.
