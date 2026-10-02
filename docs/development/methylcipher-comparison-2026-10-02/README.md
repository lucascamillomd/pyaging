# methylCIPHERv2 comparison and remediation scope

Checked **2026-10-02** against the current pyaging catalogue and [methylCIPHERv2 commit f9e5c724](https://github.com/HigginsChenLab/methylCIPHERv2/tree/f9e5c724b2d64199b696a882161fe245a39fcf6c). This is an audit and plan, not an update to canonical metadata, predictions or published artifacts. One GPT-6 Luna reviewer independently reviewed field definitions, coverage, identities and the final findings; additional workers were blocked by the app thread limit.

## Coverage and what the counts mean

- pyaging: **240** catalogue entries.
- methylCIPHERv2: **147 internal records**, comprising **131 publicly requestable clocks** and **16 hidden routed branches**. The native `routed_members()` accessor confirms this distinction.
- **104 comparable public clocks** map to 104 pyaging entries. Eight additional hidden branches belong to four matched sex-routed predictors.
- The remaining public entries comprise 25 unmatched clocks and two related outputs: the SystemsAge chronological component and a separate Retroelement 450K variant. These are not evidence that the corresponding final pyaging model is missing metadata.
- The upstream codebook covers 140 internal records; its seven sex-routing aliases do not have their own descriptor rows. Do not treat a blank alias row as missing scientific evidence for its underlying components.

The [crosswalk](crosswalk.json) accounts for every upstream record. `PhenoAge` maps to **`dnamphenoage`**, not clinical `phenoage`; `IntrinClock370` and `IntrinClock380` remain separate models. `cAge` (Bernabeu 2023) has no current pyaging counterpart and must not be mapped to `han` (Han 2020). Shared names, shared papers and shared feature counts do not establish numerical equivalence.

The [full comparison](field_comparison.csv) contains **1,560 field comparisons**. The [flagged-field table](flagged_fields.csv) contains **114 literal differences**: 19 feature-count, 18 platform, 34 algorithm, 19 tissue, 12 DOI and 12 year entries. **This is not a count of 114 errors.** Broad versus specific labels, different citation roles and different count definitions explain many flags. Free-text descriptions, missing upstream values and runtime contracts remain separately labelled rather than being counted as disagreements.

## Findings and decisions

| ID | Finding | Resolution or next action | Priority |
|---|---|---|---|
| D01 | Nine `grimage2*` proxy records say `postprocess=cox_to_years`; their actual methods return `x`. | Confirmed stale metadata. Correct the nine labels and notebook annotations to absent/none; preserve numerical behavior. Names: `grimage2adm`, `grimage2b2m`, `grimage2cystatinc`, `grimage2gdf15`, `grimage2leptin`, `grimage2loga1c`, `grimage2logcrp`, `grimage2packyrs`, `grimage2timp1`. | P1 |
| D02 | Nineteen `n_features` versus `n_cpgs` differences. | All explained by age/sex covariates or normalization inputs. Preserve existing counts and explain relevant differences in notes. New count-breakdown fields are deferred after user review. | P1 |
| D03 | Eight original GrimAge proxies cite 2019 upstream but 2022 in pyaging; PRC/non-PRC PhenoAge cite the 2018 parent upstream and 2022 decomposition in pyaging. | Resolve the displayed citation/DOI and explain reuse or derivation in existing notes and the evidence ledger after checking original sources. New citation-role fields are excluded; do not overwrite citations from a string comparison. | P1 |
| D04 | Retroelement V1/V2 use a 2023 preprint upstream and a 2024 journal article in pyaging. | Keep pyaging's journal reference. Record the relationship; this is not a pyaging correction. | Resolved |
| D05 | Eighteen platform differences, including Dunedin clocks, MiAge, HypoClock, PC clocks and Zhang2019. | Adjudicate fitting, feature selection, validation and scoring compatibility separately. The existing pyaging `platform` field means fitting or feature selection. Upstream EPIC compatibility must not become a training-platform claim. | P1 |
| D06 | Thirty-four algorithm differences. | Resolve simple synonym/subtype differences without changing scores. For SystemsAge's 12 entries, distinguish mortality/Cox fitting from the packaged PCA/linear calculation. Do the same for PRC subscores and staged models. | P1 |
| D07 | Nineteen tissue differences. | Preserve broad and specific descriptors separately where both are correct. Check skin/blood model development, feature-selection tissues, and sorted-monocyte versus generic-blood labels against original methods. | P1 |
| D08 | Normalization, covariates, missingness and cohort dependence are not directly comparable from labels alone. | No new structured input fields or catalogue-wide runtime-contract audit after user review. Preserve necessary existing notes/tutorial guidance and correct confirmed errors only. | P1 |
| D09 | Upstream HypoClock exposes mean methylation; pyaging exposes `1 - mean`. | Unresolved score orientation/variant, not a confirmed numerical bug. Establish the named output from the original paper/code and either document the deliberate complement or correct/version the implementation. | P0 investigation |
| D10 | Six McCartney implementations apply sigmoid; upstream and the published continuous-outcome methods use linear scores. | Carry forward the numerical investigation from the prior audit; verify original coefficients, transformed targets and independent reference predictions before removing sigmoid and correcting output units. | P0 |
| D11 | DepressionBarbu lacks the original M-value adjustment pipeline, and the existing intercept's provenance remains unresolved. | Trace the original supplement/code and calibration; specify prepared-input requirements, fix only proven numerical departures. This clock is outside the mutual subset but remains in scope from the previous audit. | P0 |
| D12 | Weidner uses an array-labelled PDE4C input whereas the published pyrosequencing site is upstream of that probe. | Establish assay-level equivalence or explicitly expose an approximation. Cross-package agreement does not resolve this biological measurement difference. | P0 investigation |
| D13 | The 12-cell deconvolution family uses the already disclosed 240-site reference, distinct from the optimized 1,200-site publication reference. | Preserve the variant/reference provenance in existing notes and the evidence ledger; do not silently swap reference matrices. | P1 |
| D14 | Five records remain partial from the earlier source audit: the CpGPT GrimAge3 pair and three Ying clocks. | Keep evidence gaps visible. Upstream's LOLIPOP/2,664-person annotation is a lead for Ying, not sufficient primary evidence by itself. Separate methylation-fitting samples from causal-CpG discovery and validation samples. | P1 |
| D15 | Proteomic guidance sits on a standalone page; the RNA tutorial already includes both tAge outputs. | Replace the proteomic page with a tutorial and per-clock notes. Refresh/discover the existing RNA section rather than creating duplicate examples. | P1 |

Every raw flagged field is assigned to D02–D07 in `flagged_fields.csv`; D01 and D08–D15 cover runtime discoveries and the previously identified work. The [implementation plan](../../superpowers/plans/2026-10-02-clock-metadata-reconciliation.md) defines acceptance checks and propagation for each item.

## Feature and identity checks

The [feature audit](feature_comparison.json) inspected and SHA-256-checked the current published object for each of the 104 matched pyaging clocks. It compares assay identifiers including `cg`, `ch.` and `rs` probes; non-CG assay probes are not mistaken for clinical covariates.

Of 104 upstream scoring panels, **68 matched pyaging's assay-input set exactly**, **32 were unavailable in the local catalog** because they belong to external packs or routed aliases, and four pyaging models expose a larger input set:

- CellPopAge: 2,543 candidate inputs versus 42 selected predictors.
- X/Y chromosome predictors: 453,152 inputs supporting autosomal normalization versus 4,047/284 scoring probes.
- DunedinPACE: 20,000 normalization inputs versus 173 scoring probes.

These are known representation differences. External packs and routed dependency unions should be checked where necessary to resolve substantive input questions in the next contract audit; they are not inferred to match from counts alone.

For eight original GrimAge protein/smoking proxies, [coefficient comparisons](grimage_proxy_identity.json) found matching feature/covariate sets and coefficients/intercepts consistent with the existing float32 serialization. Six have an extra age input in pyaging; Leptin and PAI1 do not. The largest relative coefficient difference was below `6.9e-8`. The upstream GrimAge2 recipe reuses these original surrogates. This supports the mapping and dual publication provenance, but is not an end-to-end imputation/normalization parity test.

## Metadata additions after user review

The independent [Luna review](luna-review.md) and [schema recommendations](luna-recommendations.json) preserve the original proposal. The revised implementation scope below supersedes those recommendations, keeping existing display fields compatible.

| Original rank | Addition | Decision |
|---|---|---|
| 1 | Input requirements: assay/value scale, required covariates and dependencies, normalization, missing-column/NaN rules, cohort dependence | Exclude after further user review. Covariates remain ordinary features; keep essential guidance in existing notes and tutorials. |
| 2 | Feature-count breakdown and implementation variant | Defer. Keep existing counts and explain important distinctions in notes. |
| 3 | Training design: cohort/stage, samples versus participants, assay/outcome ages, tissue/platform and health status | Include, with original-source evidence and explicit unknowns. |
| 4 | Separate validation/platform-compatibility fields | Exclude. Still correct errors in existing platform metadata. |
| 5 | New citation-role and source/asset-provenance fields | Exclude. Still correct existing citations and preserve the current evidence ledger. |
| 6 | Sex and ancestry distributions | Exclude. Required sex inputs remain ordinary model features. |

These ranks refer to proposed metadata additions, not numbered scientific implementation tasks. Numerical investigations and the requested tutorial migration remain in scope. Astra handles tutorials; multiple Luna workers review all 240 clock/citation records. No new structured covariate, normalization, missingness or cohort-dependence fields will be added.

Among the **140 codebook rows**, upstream has training cohort for 132, sample count for 127, health status for 128, age bounds for 121, sex distribution for 80 and ancestry for 66. These are availability counts, not verified accuracy counts; sex/ancestry distributions will not be imported under the revised scope. Its descriptors sometimes combine heterogeneous cohorts and stages; copy neither a pooled count nor an age span without checking what it describes. Dunedin outcome-trajectory ages, for example, are not the ages at which methylation was fitted.

Do not add redundant copies of `tissue`, `platform`, `model_type`, `notes` or citation counts. Do not import the upstream `external` flag (all pyaging weights are remote), subjective generation/tags, or mixed license labels as author approval. Preserve `null/unknown`, `not applicable`, and verified empty/none distinctly; do not manufacture sex or ancestry estimates from geography.

## Evidence limitations and reproduction

`summary.json` records package commits, hashes, counts and scope. The public upstream `R/sysdata.rda` is the source of the index, codebook, citation and runtime-contract exports. Its embedded provenance points to metadata-source commit `19c9039079b27543c2c625ed5ece5a5ef4b8e0e2`, but that source repository currently returns HTTP 404. The [sync script](https://github.com/HigginsChenLab/methylCIPHERv2/blob/f9e5c724b2d64199b696a882161fe245a39fcf6c/data-raw/sync.R#L297-L367) joins descriptive fields without field-level source evidence and collapses blank/NA/NR values. The [public listing](https://github.com/HigginsChenLab/methylCIPHERv2/blob/f9e5c724b2d64199b696a882161fe245a39fcf6c/R/list_clocks.R#L31-L47) documents their meanings.

Reproduction inputs are the pinned package's serialized metadata, pyaging's pinned registry and the current per-clock objects whose SHA-256 values are recorded here. Native R 4.5.3 loaded the package data and verified the callable/hidden split; Python `rdata` exported the tables. The comparison preserves original values beside classifications, and tensor checks include their tolerance. No model was rebuilt, no numerical source was copied into pyaging and no production metadata was changed during this comparison.

The clinical PhenoAge gamma correction is already released in 0.5.7 and is not reopened. ProtAge/ipfP3GPT access limitations and unmatched additional clocks remain separate asset/feature requests, not metadata corrections.
