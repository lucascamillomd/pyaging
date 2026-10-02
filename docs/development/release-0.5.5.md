# 0.5.5 validation and source record

The executable catalogue contains 238 entries: 191 unchanged non-OrganAge
entries, 46 renamed full-panel OrganAge entries and one new PAC entry.

## Sources and scientific scope

PAC uses the original Kuo lab R source at
`e15edb66800f876a928cec9c8856500048c46fea`. Fitted coefficients and full-precision
Gompertz parameters are extracted by `clocks/build_proteoclock.py`, with pinned
SHA-256 checks. Insilico's rounded parameters are not used. Input is 128
assay-normalized Olink NPX proteins plus chronological age; output is
mortality-equivalent years. The original source imposes no scoring-time scaling
or imputation, and requires every predictor column. The public pyaging path
validates this before its generic missing-feature filling step.

OrganAge still uses the original authors' fold-one coefficients at
`5147b0301ec7f4abdb10ef650d04f47454ddc8fd`. The 44 feature-reduced Explore 1536
entries are removed from the live catalogue. The 46 full Explore 3072 models
drop `olink3000` from their names. Their ordered features, double-precision
coefficients and intercepts are unchanged. The original 90-model oracle files
remain intact; current tests select and rename the full-panel outputs.

The [proteomics tutorial](../../tutorials/tutorial_proteomics.ipynb) distinguishes plasma training
material from serum trial measurements, NPX from concentrations and standardized
values, and raw OrganAge mortality scores from mortality-equivalent years. It
also distinguishes the original models from additional preprocessing in the
Insilico trial. The [source audit](proteomic-source-audit-0.5.5.md) records why
ProtAge, ipfP3GPT and PAOPAC are not executable entries in this release.

## Validation

- Native R 4.5.3 executed the original PAC function on eight synthetic input
  profiles across complete, reordered and explicit-NA cases. Maximum finite
  prediction discrepancy was 3.55e-14 years.
- PAC tests verify required age/protein columns, feature names, batch sizes,
  sample order, serialization, provenance and caller-selected prior imputation.
- All 46 retained OrganAge artifacts preserve the original coefficient and
  intercept values exactly and pass the original R prediction fixtures.
- All 47 published artifacts passed canonical metadata, notebook, field-level
  evidence, feature-count and aggregate-runtime consistency validation.
- The 191 pre-existing non-OrganAge canonical metadata records are unchanged.
- Default local suite: 624 passed, 82 skipped. Skips cover optional older local
  artifacts and unavailable CUDA; 361 full-catalogue/online cases are deselected.
- Ruff and catalogue JavaScript tests pass. Distribution and documentation
  builds are checked before release.

The public artifacts are uploaded before aggregate metadata. Old Hugging Face
files and immutable v0.5.4 tags are retained for reproducibility; presence of an
archived file is not current catalogue membership. The v0.5.5 data snapshot
includes the new artifacts and the updated aggregate catalogue.
