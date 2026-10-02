# Batch 3 primary-source review

Reviewed all 40 records in `batch-3-input.json` against the exact-DOI primary publication, original author code/supplement where needed, and the existing audit context. Citation-count snapshots retain their observed 2026-10-02 values when the DOI/title mapping matches the OpenAlex work. No shared registry, runtime implementation, or citation audit file was edited.

## Supported corrections

- `ocampoatac1` and `ocampoatac2`: the DOI, title, journal, last author, tissue, platform and target are correct. The citation’s pagination is wrong: GeroScience 46(2) is pages **1789–1806**, not 635–650. The model registry year 2023 is retained because the article first appeared online on 2023-11-04; the journal issue citation correctly uses 2024. Primary source: [publisher article](https://link.springer.com/article/10.1007/s11357-023-00986-0).
- Six McCartney outcomes (`mccartneybmi`, `mccartneybodyfat`, `mccartneyeducation`, `mccartneyhdlcholesterol`, `mccartneyldlcholesterol`, `mccartneytotalcholesterol`) have sigmoid postprocessing that is absent from the original authors’ linear LASSO score construction. Proposed metadata corrections are `postprocess: sigmoid → identity` and output units describing the fitted adjusted residual scale at a general level; the exact source transformations remain unverified (see unresolved items). The score does not recover the absolute phenotype because age, sex and 10 genetic PCs were regressed out before LASSO. The four other McCartney models (alcohol, smoking, total:HDL ratio and WHR) already return linear scores.

The McCartney fixture uses the original publisher coefficient workbook. With beta=1 at the listed CpG and zero at all other model sites, the direct author-weight outputs are BMI 0.4465519683, body fat 17.0092599823, education 3.1253169882, HDL 1.2764034274, LDL 2.3404997225, total cholesterol 1.8702743694. Current sigmoid outputs are respectively 0.6098191215, 0.9999999590, 0.9579250500, 0.7818369383, 0.9121761266, and 0.8664900211. This is an algebraic synthetic fixture, not performance validation. The primary paper documents phenotype residualization and coefficient scoring ([paper](https://link.springer.com/article/10.1186/s13059-018-1514-1)); original weights are in [Additional file 1](https://static-content.springer.com/esm/art%3A10.1186%2Fs13059-018-1514-1/MediaObjects/13059_2018_1514_MOESM1_ESM.xlsx). Runtime code was not changed.

## Training facts

The JSON records source-backed training facts for each clock where the primary source exposed them; bounded evidence gaps are listed explicitly. Examples of important distinctions retained: Lin was discovered across birth-to-78 age-associated samples but its multivariate 450K fitting stage used the adult Hannum cohort; LinAge2 was trained on NHANES IV 1999–2000 mortality and tested on 2001–2002; the Lee placental clocks have distinct control, robust and uncomplicated-term training cohorts; and OrganAge coefficients are trained on UK Biobank plasma proteomics, while the article reports a broader 50,000-plus-participant study.

The review retained the existing model/tissue/platform/output descriptions wherever the primary paper or original author artifact supported them. Feature-count disputes remain outside this review except where a coefficient fixture was needed to verify the McCartney score equation. ## Unresolved source checks

- McCartney D10: the publisher paper and Additional file 1 independently support linear scoring and the sigmoid fixture. Exact transformed phenotype scales still need direct extraction from Additional file 4, and full author-reference parity was not tested; numerical runtime changes are deferred.
- Knight: six training cohorts are confirmed, but a combined sample count was not independently reconciled from the primary cohort table/supplement.
- Mayne: the pooled healthy-placenta training description is supported; pooled N and gestational-age range remain unverified from a readable primary-source passage.
- OrganAge: the broad UK Biobank study facts are supported, but per-variant cohort N and age spans were not independently mapped.

These are evidence limits, not proposed metadata corrections. The original sources are recorded in the JSON.
