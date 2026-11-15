# Clock metadata and citation audit — 2 October 2026

This audit follows the clinical PhenoAge correction released in **pyaging 0.5.7**. That numerical correction is documented separately in [the PhenoAge investigation](phenoage-gompertz-correction.md). The catalogue audit changes metadata; it does not change model predictions.

## Review coverage

The [per-clock review](../../clocks/metadata/methods_audit_2026-10-02.json) covers all **240 clocks**. One GPT-6 Luna worker and the root agent reviewed the catalogue against papers, supplements, author code and the published objects. Additional requested workers could not start because of the app's agent limit.

**235 records have no unresolved catalogue fields in this review; five remain partial.** For `cpgptgrimage3` and `cpgptpcgrimage3`, the author tutorial and packaged models confirm the input/output structure, but the training cohort fields retain the previous author clarification because the paper methods could not be independently rechecked. For `yingadaptage`, `yingcausage` and `yingdamage`, the exact assay and age breakdown of the final 2,664-person fitting cohort remains unresolved. The evidence ledger no longer conflates that cohort with the separate Generation Scotland age-association analysis. Existing labels are retained rather than replaced by guesses.

“Complete” here describes the metadata review, not reproduction of a model's training or validation. Author-approval flags are not inferred from papers, and a research-only label does not establish redistribution permission.

## Citation counts

All **240 clocks** have OpenAlex citation counts observed on **2026-10-02**, covering **82 distinct publication DOIs**. Each count was retrieved by exact DOI and checked against the paper title. Counts for a preprint and its journal article are not added together. Multiple clocks from the same paper share its citation count.

The [citation snapshot](../../clocks/metadata/citation_audit_2026-10-02.json) records the publication title, DOI, OpenAlex work identifier, API URL, response SHA-256, count, observation date and associated clock names. It preserves the evidence for this snapshot even when live counts change.

PASTA, PASTAMouse and REG now cite their 2026 *Advanced Science* journal article. MammalianFemale now cites the 2024 *Science Advances* article already used for the companion lifespan predictor. HypoClock's count is tied to the 2020 paper defining its 678-site implementation, rather than the 2018 biological precursor.

## Confirmed interpretation changes

| Clock | Correction | Primary evidence |
|---|---|---|
| `tagemortality` | Specify **log10 hazard ratio**, replacing the ambiguous “log hazard” unit. | [Author output documentation](https://gladyshev-lab.github.io/tAge/index.html#interpreting-the-output) |
| `pcdnamtl` | Specify **kilobases**, replacing “base pairs.” The author scoring function returns the original DNAmTL score without multiplying by 1,000; figure-level base-pair deviations are separate. | [Author scoring function, line 85](https://github.com/MorganLevineLab/PC-Clocks/blob/main/run_calcPCClocks.R#L85), [original DNAmTL paper](https://doi.org/10.18632/aging.102173) |
| `pasta`, `pastamouse` | Clarify that the classifier produces relative age on a human-year scale, including when transferred to mouse orthologues. | [Journal article, Methods 4.3](https://doi.org/10.1002/advs.76740) |
| `mammalianfemale` | Document SeSAMe-normalized mammalian-array inputs, 320K-to-40K probe mapping, female probability output and exclusion of marmosets from published validation. | [Author prediction function](https://github.com/caeseriousli/MammalianMethylationPredictors/blob/main/R/predictSex.R), [paper, Table 1](https://doi.org/10.1126/sciadv.adm7273) |
| `hypoclock` | Remove stale wording implying that the registry still cites the 2018 paper; distinguish that precursor from the named 2020 implementation. | [2020 methods](https://doi.org/10.1186/s13073-020-00752-3) |
| `cabec`, `xchrom`, `ychrom` | Separate the fitting assay from cross-platform compatibility and validation. cABEC was fitted on EPIC; X/Y chromosome scores were fitted on 450K. | [cABEC methods](https://doi.org/10.1186/s12864-020-07168-8), [X/Y methods and Table 3](https://doi.org/10.1186/s12864-021-07675-2) |
| `gliasin`, `neusin` | Include EPIC in model development because the final penalty was selected on an EPIC cohort after fitting coefficients on 450K. | [Clock construction methods](https://doi.org/10.18632/aging.206184) |
| `dnamphenoage`, `dnamphenoageprc`, `dnamphenoagenonprc` | Label the parent fitting platform as 450K; describe 27K and EPIC overlap as compatibility. | [Original supplementary methods, pp. 2–3](https://cdn.aging-us.com/article/101414/supplementary/SD1/0/aging-v10i4-101414-supplementary-material-SD1.pdf) |
| `corticalclock`, `lin`, `weidner` | Include the broad age ranges used for fitting or feature selection, rather than labelling them adult-only or age-unspecified. | [Cortex methods](https://doi.org/10.1093/brain/awaa334), [Lin methods](https://doi.org/10.18632/aging.100908), [Weidner results](https://doi.org/10.1186/gb-2014-15-2-r24) |
| `depressionbarbu` | Specify LASSO, adjusted M-value inputs and a unitless MDD risk score. Annotate its feature units without imposing the generic beta-value range. | [Training and DNA methylation methods](https://doi.org/10.1038/s41380-020-0808-3) |
| `mccartneyalcohol`, `mccartneysmoking` | Replace physical exposure units with the adjusted natural-log `value + 1` scales used in training. | [Original additional file 4](https://static-content.springer.com/esm/art%3A10.1186%2Fs13059-018-1514-1/MediaObjects/13059_2018_1514_MOESM4_ESM.pdf) |
| Six McCartney sigmoid scores | Explain the package's bounded output rather than implying an original phenotype residual or probability. | [Packaged postprocessing](https://github.com/lucascamillomd/pyaging/blob/v0.5.7/src/pyaging/models/_models.py), [paper methods](https://doi.org/10.1186/s13059-018-1514-1) |
| `weidner` | Explain that the published PDE4C pyrosequencing site is upstream of the package's `cg17861230` feature label. Direct array-probe substitution is an approximation. | [Pyrosequencing methods and Figure 2c](https://doi.org/10.1186/gb-2014-15-2-r24) |
| CpGPT GrimAge3 pair | Update the author citation and explicitly require derived CpGPT/GrimAge2 proxies and age. | [Author tutorial](https://github.com/lcamillo/CpGPT/blob/5bfe6ac5e32c8ef7effc56349fe17770de09b7cb/tutorials/predict_mortality.ipynb) |

## Numerical questions exposed by the review

These findings need a separate numerical investigation; they are now visible in the clock notes:

- The six McCartney BMI/body-fat/education/cholesterol implementations apply a sigmoid that the original continuous-outcome LASSO methods do not specify. This audit records their actual output scale and preserves their coefficients and postprocessing.
- DepressionBarbu does not perform the paper's M-value normalization and covariate adjustment. Its input annotation is corrected, but the provenance of the existing calibration intercept and end-to-end equivalence still require investigation.
- Weidner's array-labelled PDE4C input does not measure the exact published pyrosequencing site. Relabelling a probe cannot make the assays equivalent.

The previously documented 240-site implementation of the 12-cell deconvolution family also remains distinct from the paper's optimized 1,200-site reference. Its notes explicitly identify the variant.

## Consistency and numerical preservation

The audit inspected the current public **per-clock** Hugging Face objects for all 240 clocks. Those objects were the baseline; older shared-repository copies were not used as the source for rebuilding models.

All packaged effective feature counts agree with the registry, including the existing nonzero-coefficient policies for CellPopAge and EnsembleAgeHumanMouse. The registry, notebook source metadata, serialized model metadata and aggregate metadata passed the complete 240-clock consistency validator.

Metadata repacking preserves the original tensor-storage bytes, checked with SHA-256 for every storage entry. Legacy artifact-version labels were normalized to 0.5.7 where metadata was rebuilt. Notebook metadata and its rendered display were refreshed while retaining numerical outputs, except for two DepressionBarbu input-guidance cells whose stale beta-value outputs were cleared. Those revised demonstration cells were not executed. This was not a rerun of every training notebook; the PhenoAge notebook was executed separately for its formula correction.

Hugging Face updates cover model metadata embedded in `.pt` objects, per-clock `config.json` files and model cards, and the shared aggregate. Existing author licenses and companion assets are retained. Historical release tags are not moved. The catalogue JSON/CSV and expandable citation dates use the same registry.

Validation includes the local suite (682 passed, 72 skipped, 361 deselected), the catalogue JavaScript check, lint, the Sphinx build, full artifact consistency and anonymous retrieval of all 240 per-clock configs. The skipped tests include optional CUDA and locally unbuilt assets; feature-count and metadata inspection separately covered the published objects for all 240 clocks. This audit is not a numerical revalidation of every clock against every author dataset.
