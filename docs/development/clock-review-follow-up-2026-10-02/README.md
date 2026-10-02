# Clock metadata follow-up and tutorials

Six GPT-6 Luna workers reviewed disjoint batches of 40 catalogue entries, covering all **240 clocks**. They checked the current metadata, evidence ledger and DOI-linked citation snapshot and used primary papers, publisher records and author code to investigate proposed changes. This is a bounded follow-up to the previous source audit, not a fresh full-text or numerical reproduction of every clock. Each batch records its source-access limits and retained versus unresolved findings.

GPT-6 Astra replaced the standalone proteomic guide with the [proteomics tutorial](../../../tutorials/tutorial_proteomics.ipynb) and refreshed the mammalian tAge/tAgeMortality section of the [RNA tutorial](../../../tutorials/tutorial_rnaseq.ipynb). Both notebooks execute, retain their outputs and distinguish synthetic demonstrations from biological validation. The old page is removed, its links point to the tutorial, and necessary platform/preparation instructions remain in prose. No new required-covariate, normalization, missingness or cohort-dependence fields were added. Required covariates remain ordinary model features.

## Corrections applied

- Removed stale age-conversion labels from nine GrimAge2 proxies. Their classes already return the score unchanged; `postprocess_name` now agrees with the existing notebook `None` convention, and the metadata key is omitted. The aggregate and catalogue match. PAI1 was already correct.
- Corrected both Ocampo ATAC-clock citations from pages 635–650 to **1789–1806**, following the [publisher record](https://link.springer.com/article/10.1007/s11357-023-00986-0). The first-online year remains 2023; the issue citation remains 2024. Updated registry, evidence, notebook citation cells/rendered citation, serialized metadata, configs and model cards together.
- Removed obsolete Bohlin ledger warnings claiming no official minimum-lambda variant was available. The author package exposes that variant. This does not claim a new end-to-end coefficient reproduction.
- Fixed three skipped heading levels in tAge model notebooks, a dead PhenoAge contents link, and accidental autodocumentation of imported Rich `Columns` so strict Sphinx succeeds.

Citation counts remain the verified **2026-10-02 OpenAlex snapshot** across 82 DOIs. Changing citation page numbers does not warrant substituting publisher citation counts, and preprint/journal counts are not added together.

## Validation and publication

- Offline suite: **692 passed, 72 skipped, 361 deselected**. Skips include unavailable CUDA and optional local assets.
- Metadata/catalogue checks after the citation edits: **97 passed, 1 deselected**.
- Tutorial execution: **2 passed**. Strict Sphinx, catalogue JavaScript, Ruff and diff checks pass.
- All **11 updated model objects** preserve every tensor-storage byte and fixed-profile predictions. See [proxy validation](proxy-label-validation.json) and [citation validation](atac-citation-validation.json).
- Corrected per-clock objects/configs, the shared fallback objects and aggregate were published to Hugging Face. Existing licenses and release tags were retained. [Publication commits](hf-publication.json) and [fresh anonymous verification](anonymous-verification.json) record the results. The public prediction API was checked for all 11 corrected objects.

## Remaining scientific work

No clock's scoring equation changed in this batch. The six McCartney sigmoid transformations need a separate numerical correction with complete original output-scale and reference-example validation. DepressionBarbu's intercept provenance, CVDWesterman and ZhangMortality's packaged-formula provenance, the 240-site cell-deconvolution variant, PAOPAC trial/public-version equivalence and several exact training-cohort details remain qualified in the reviews. Weidner's array-site approximation remains explicitly documented. Unknown training details were not invented or imported from another package.

Source-backed training facts are collected in the batch reports. This batch does not introduce a new training schema; that separate plan item remains open. New feature-count breakdown remains deferred, and compatibility, citation-role and demographic-distribution fields remain excluded.

## Review records

- [Luna batch 1](batch-1-review.md), [structured findings](batch-1-review.json)
- [Luna batch 2](batch-2-review.md), [structured findings](batch-2-review.json)
- [Luna batch 3](batch-3-review.md), [structured findings](batch-3-review.json)
- [Luna batch 4](batch-4-review.md), [structured findings](batch-4-review.json)
- [Luna batch 5](batch-5-review.md), [structured findings](batch-5-review.json)
- [Luna batch 6](batch-6-review.md), [structured findings](batch-6-review.json)

The [machine-readable summary](summary.json) records exact coverage and outstanding evidence. The [revised plan](../../superpowers/plans/2026-10-02-clock-metadata-reconciliation.md) separates completed metadata/tutorial work from remaining scientific investigations.
