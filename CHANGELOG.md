# Changelog

## 0.5.5

- Add Kuo et al.'s PAC from original-author code and coefficients, with 128
  Olink NPX proteins plus chronological age and mortality-equivalent output
  in years. Required inputs are checked before generic missing-feature filling.
- Remove the 44 reduced-panel Olink Explore 1536 OrganAge entries from the
  current catalogue. Retain 46 full-panel models and shorten their names by
  dropping `olink3000`; coefficients and raw output scales are unchanged.
- Specify plasma training material, assay normalization, protein identifiers,
  missing-data behavior and model-specific feature units for proteomic clocks.
  Add a [proteomic input guide](docs/source/proteomic_clocks.md).
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
