# Clock Metadata Reconciliation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the identified scientific and runtime discrepancies, add useful metadata with traceable evidence, and move modality guidance into executable tutorials.

**Architecture:** Keep the canonical registry and field-level evidence ledger as the source of truth. Add optional structured metadata without redefining existing fields, validate it against model artifacts and actual runtime behavior, and propagate accepted changes to notebooks, HF and the catalogue. Numerical corrections require original-author reference calculations; methylCIPHERv2 is a comparison implementation.

**Tech Stack:** Python, PyTorch, AnnData, JSON/JSONL, original-author R reference functions, pytest, Jupyter/nbmake, Sphinx and Hugging Face Hub.

**Spec:** [Comparison report and discrepancy register](../../development/methylcipher-comparison-2026-10-02/README.md), including `flagged_fields.csv`, `feature_comparison.json` and the independent Luna review.

## Scope after user review

The only new metadata under consideration is **training-cohort detail**: cohorts, sample/participant counts, age ranges and health status. Do not add separate required-covariate fields: covariates already appear in model features. Do not add normalization, missing-data-handling or cohort-dependence fields. The proposed input-contract block is removed.

Feature-count breakdown remains deferred. New validation/platform compatibility, citation-role/source-asset provenance, and sex/ancestry fields remain excluded. Correct mistakes in existing fields and citations using original sources and the existing evidence ledger. Keep essential preparation guidance in existing clock notes and tutorials.

Use **GPT-6 Astra** for the proteomics tutorial replacement and RNA tutorial refresh. Use multiple **GPT-6 Luna** workers to review all 240 clocks/citations in disjoint batches and investigate small errors. Integrate only source-backed corrections, preserve uncertain findings explicitly, and avoid simultaneous edits to shared registries.

## Global Constraints

- Use original papers, supplements and author code for decisions. Use methylCIPHERv2/Biolearn only to cross-check; do not transplant their implementations.
- Preserve current clock names and legacy metadata meanings. `platform` remains fitting/feature-selection platform; `n_features` remains the current effective implementation count.
- Separate metadata-only changes from changes to scores. Preserve tensor-storage bytes in metadata-only repacks and never move historical HF/Git release tags.
- Unknown, unreported, not applicable and known-none are distinct. A second package's populated cell does not establish primary-source confirmation.
- Preserve assay identifiers, normalization scope, missingness behavior and explicit input/output units. Protein-name overlap never establishes platform equivalence.
- Use Luna for independent source/metadata reviews, as requested. If concurrency limits prevent multiple workers, queue their reviews rather than silently substituting another model.
- No author correspondence is authorized. Document unavailable evidence/assets without fabricating a result or a nonfunctional clock.
- The PhenoAge gamma correction is complete in 0.5.7. Do not redo it or change DNAmPhenoAge on that basis.

## Review focus

1. Distinguish actual errors from differences in naming, citation choice and count definitions. Keep current feature-count semantics.
2. Check publication identity and citation-count provenance for all 240 catalogue entries. Do not sum journal/preprint counts or infer author approval.
3. Resolve questionable numerical definitions against original-author sources before changing scores. Metadata review alone does not establish numerical parity.
4. Keep tutorials scientifically accurate and execute their examples. Do not create structured input fields or a new catalogue-wide preprocessing audit.

## Order and completion policy

Start with Task 1, then Tasks 2–5 as independent scientific investigations. Tasks 6–8 cover training details, metadata/citation review and integration; Task 9 proceeds independently with Astra. Task 10 publishes only validated changes. Each task ends in its own reviewable commit. A source-access gap may remain explicitly unresolved; it must not be converted to “confirmed” to make a completeness test pass.

The audit already accounts for all 131 public upstream entries and 16 hidden branches, maps 104 shared public clocks, and assigns all 114 literal differences to D02–D07. Do not repeat that inventory. Close or carry forward each discrepancy ID D01–D15 in the report as implementation proceeds.

### Task 1: Correct the nine stale GrimAge2 proxy labels — D01

**Files:** Modify `clocks/notebooks/grimage2{adm,b2m,cystatinc,gdf15,leptin,loga1c,logcrp,packyrs,timp1}.ipynb`, `clocks/metadata/clock_metadata.json`, and metadata construction/validation as needed; test `tests/test_clock_metadata.py`.

**Interfaces:** Keep each class's existing identity `postprocess`. Its runtime metadata must omit the postprocess label, matching the existing `None` convention. PAI1 is already identity/no misleading label and needs no forced numerical change.

- [ ] Add a regression check loading these nine objects: recorded postprocess is absent/none, and `model.postprocess(torch.tensor([-1., 0., 2.]))` returns the same values. Confirm the metadata assertion fails on the current records.
- [ ] Correct the notebook annotations and regenerated registry/runtime metadata. Inspect other explicit runtime labels for the same copy/paste issue; add a discrepancy record if one is found.
- [ ] Repack metadata only; compare every tensor-storage SHA-256 and fixed predictions before/after. Run `uv run pytest tests/test_clock_metadata.py -q`.
- [ ] Commit the correction; queue HF publication for Task 10. Do not change proxy coefficients or units to years.

### Task 2: Reproduce and correct the six McCartney outputs — D10

**Files:** Modify the six corresponding classes in `src/pyaging/models/_models.py` and notebooks `mccartney{bmi,bodyfat,education,hdlcholesterol,ldlcholesterol,totalcholesterol}.ipynb`; create `tests/models/test_mccartney_author_parity.py` and `tests/data/mccartney/`; update their registry/ledger entries.

**Interfaces:** Public names remain unchanged. Expected outputs come from the original published coefficients applied to the correctly adjusted/transformed targets, not from methylCIPHER's results.

- [ ] Retrieve and hash the original coefficient supplement and outcome-preparation methods. Determine the exact target scale separately for all six outcomes, including the logged BMI outcome; preserve the existing alcohol/smoking corrections unless contrary primary evidence is found.
- [ ] Generate independent R/NumPy fixtures for profiles producing negative, zero and positive linear scores. Add tests that distinguish the linear author output from the current bounded sigmoid. Run the new file and confirm the current implementation fails the author fixtures.
- [ ] Remove sigmoid only where the original definition and fixtures support it. Update output units, notes and notebook examples together; do not label adjusted residuals as physical measurements or probabilities.
- [ ] Run the new tests plus existing model/prediction tests. Record prediction differences for fixed profiles and the package-version requirement, then commit. Do not regenerate golden expectations from the corrected Python class.

### Task 3: Resolve DepressionBarbu's input pipeline and intercept — D11

**Files:** `clocks/notebooks/depressionbarbu.ipynb`, its metadata/ledger records and `src/pyaging/models/_models.py`; create `tests/models/test_depressionbarbu_author_parity.py` and `tests/data/depressionbarbu/` if executable original reference material is available.

**Interfaces:** Prepared adjusted M-values are the current declared input. Do not silently convert a beta matrix or estimate missing covariate-adjustment coefficients from the prediction cohort.

- [ ] Trace all 196 coefficients and the current intercept `12.2169841` to original-author files; distinguish training-outcome residualization, methylation adjustment and any later calibration. Record an unavailable source explicitly.
- [ ] If supported, construct fixed prepared-M-value reference profiles and independent expected scores, testing intercept, coefficient order and output scale. Confirm any proposed numerical fix fails these tests before implementation.
- [ ] Implement only proven differences. If preprocessing requires unavailable fitted references/covariates, retain the prepared-input interface and clearly mark the unresolved parity limitation; do not claim a complete beta-to-score implementation.
- [ ] Execute the revised notebook and reference tests when the reference is available; otherwise commit only source-backed metadata/provenance changes and retain D11 as unresolved with its exact missing asset.

### Task 4: Resolve Weidner's assay identity — D12

**Files:** `clocks/notebooks/weidner.ipynb`, its metadata/ledger records and the Weidner implementation; create `tests/models/test_weidner_author_parity.py` and `tests/data/weidner/` only for a supported corrected scoring path.

**Interfaces:** Preserve the distinction between the published PDE4C pyrosequencing site and `cg17861230`. Do not equate them because both packages use the same array label.

- [ ] Identify the original site's genomic coordinate/build, assay sequence and beta-value scale, then compare the three assay inputs and published equation against the notebook.
- [ ] If a correct input contract can be specified, create author-equation fixtures with explicit assay labels and verify that the current substitution is distinguishable. Never infer equivalence from nearby coordinates alone.
- [ ] Either expose the exact assay contract with a migration note, or retain an explicitly named/documented approximation. A label-only correction must preserve scores; an input/numerical change needs reference tests and a release note.
- [ ] Run targeted tests and commit the resolved contract, or retain the exact unresolved assay question without changing the numerical model.

### Task 5: Resolve HypoClock orientation — D09

**Files:** `clocks/notebooks/hypoclock.ipynb`, the `HypoClock` class in `_models.py`, its registry/ledger records; create `tests/models/test_hypoclock_author_contract.py`.

**Interfaces:** Current pyaging returns `1 - mean`; upstream exposes a raw mean before any downstream acceleration analysis. Neither package is automatically the authority.

- [ ] Pin the original named-score definition and distinguish raw mean methylation, complement/hypomethylation burden and age-adjusted HypoScore.
- [ ] Build independent fixtures for complete beta values 0, 0.5 and 1, plus omitted sites, supplied NaNs and the existing `-1` sentinel. Assertions must specify the intended output orientation and missingness denominator.
- [ ] If the complement is intentional, preserve it and identify the variant/output direction explicitly. If it is wrong for the published name, correct/version the score and document the transformation from previous results.
- [ ] Run the targeted tests; commit a supported resolution. Do not change orientation solely to match methylCIPHERv2.

### Task 6: Add source-backed training detail — D14

**Files:** Registry/ledger and clock notebooks; adjust validation/serialization only as needed for accepted training fields.

**Interfaces:** Keep existing fields readable. A compact optional `training_design` block may contain cohort name/accession, fitting stage, sample/participant counts, age-at-assay/outcome ranges and health status. Retain source evidence and explicitly unknown values. Do not introduce input-contract, detailed count, compatibility, citation-role or demographic-distribution blocks.

- [ ] Collect training facts during the Luna reviews, separating fitting cohorts from feature selection, calibration and validation. Do not sum overlapping cohorts or substitute outcome ages for assay ages.
- [ ] Add only source-backed detail, with null/unreported distinguished from verified zero. Reuse the current ledger for provenance.
- [ ] If structured fields are introduced, validate nonnegative counts, ordered age ranges with units, and metadata serialization. Preserve legacy records and existing `n_features`.
- [ ] Synchronize accepted training details through notebooks, artifacts and catalogue during Task 10.

### Task 7: Adjudicate existing metadata and populate training details — D03, D04, D05, D06, D07, D13, D14

**Files:** The canonical registry/ledger, affected `clocks/notebooks/*.ipynb`, `controlled_vocabulary.json`, the comparison discrepancy register and a new dated methods-review snapshot. Preserve the 2026-10-02 audit as a historical snapshot.

**Interfaces:** Accepted training details plus the existing 16 audited scientific fields. All 114 flagged rows must end in a recorded disposition: corrected, retained with reason, different variant/definition, or unresolved with missing evidence.

- [ ] Review the exact clock lists in `flagged_fields.csv` against original sources. Prioritize SystemsAge algorithm/cohort stages, PC fitting versus original-parent cohorts, the 18 platform flags, and the five previous evidence gaps. Treat upstream LOLIPOP details as leads, not confirmed values.
- [ ] Resolve original-versus-reuse citations for the eight GrimAge proxies and parent-versus-derivative citations for the PRC pair using existing citation/DOI fields, notes and the evidence ledger; do not add citation-role fields. Keep Retroelement's current journal citation. Preserve single-publication citation counts; never sum preprint and journal counts.
- [ ] Populate structured training fields first for the 104 shared clocks, then the remaining 136 from their existing primary evidence. Mark genuinely absent/unreviewed values explicitly. Limit new training fields to cohort/stage, sample/participant counts, ages and health status; omit sex/ancestry distributions.
- [ ] Preserve the documented 240-site deconvolution variant and reference provenance in existing notes and the evidence ledger; do not add a variant field. Do not replace its matrix with the paper's 1,200-site reference in this task.
- [ ] Run registry/evidence validation and the complete artifact consistency validator after repacking. Numerical tensor hashes must remain unchanged. Review the final field diffs with Luna and commit.

### Task 8: Integrate the Luna reviews across the full catalogue

**Files:** Dated per-batch review artifacts, registry, evidence ledger and affected notebooks.

**Interfaces:** Six disjoint batches of 40 clocks cover all 240 entries. Workers write separate proposed corrections and evidence; the coordinator alone updates shared metadata.

- [ ] Record a disposition for every clock and proposed correction. Existing same-day citation snapshots may be retained after checking DOI/title associations.
- [ ] Check each accepted change against its original source and current implementation. Reject unsupported changes and keep source-access gaps visible.
- [ ] Correct confirmed small metadata errors, including the nine stale proxy postprocess labels. Preserve scores and tensor bytes for metadata-only edits.
- [ ] Run focused validation and the full metadata/artifact consistency checks. Keep numerical changes separate and require original-author reference results.

### Task 9: Replace the proteomic page and refresh RNA guidance — D15

**Files:** Create `tutorials/tutorial_proteomics.ipynb`; delete `docs/source/proteomic_clocks.md`; modify `docs/source/index.rst`, `docs/source/tutorials/index.rst`, `tutorials/tutorial_rnaseq.ipynb` and relevant clock notes/notebooks. Retain developer source audits as provenance records.

**Interfaces:** The standard Sphinx config copies `tutorials/*.ipynb`; register the new notebook in the tutorial toctree. Update all current links to the removed page. Do not add a second standalone proteomic overview under a different name.

- [ ] Move the page's essential input guidance into a concise proteomic tutorial: construct sample-aligned AnnData without unintended imputation; score PAC/HPS, one chronological and one mortality OrganAge output, and PAOPAC with an explicit full-cohort example; inspect feature coverage and output units.
- [ ] Use a redistributable public example if available, otherwise a deterministic, explicitly synthetic dataset demonstrating API behavior, not biological performance. Preserve assay-specific case, `nppb` versus `ntprobnp`, age alignment, NPX scale, plasma/serum distinctions and the absence of automatic cross-platform harmonization.
- [ ] Keep clock-specific rules in notes: PAC/HPS no additional scaling and required columns; OrganAge full Explore 3072 fold/identifier/missingness rules; PAOPAC's exponentiation, zero fill, full-cohort scaling/LOWESS, optional matrix TDI and singleton behavior. Keep trial-version equivalence and unavailable-model discussion in developer provenance, with a short tutorial link where useful.
- [ ] Refresh the existing tAge section in `tutorial_rnaseq.ipynb`: link it from the tutorial introduction, lead with an explicit species indicator, preserve raw-count/reference-group requirements and the distinction between mouse-month relative age and log10 mortality hazard. Do not use the earlier C. elegans example as mammalian tAge input.
- [ ] Execute both notebooks using `uv run pytest --nbmake tutorials/tutorial_proteomics.ipynb tutorials/tutorial_rnaseq.ipynb`. Execute examples after needed HF assets are published; inspect outputs and warnings rather than only parsing notebooks.
- [ ] Build Sphinx, check navigation and links, and confirm the old page is absent from the current source/toctree. Commit the documentation migration. No additional unit tests are needed merely to assert headings or file deletion.

### Task 10: Propagate, validate and release the accepted changes

**Files:** HF model objects/configs/cards/sidecars and aggregate; `docs/_static/clocks.json`, `docs/_static/clock_glossary.csv`, `docs/_static/clock_explorer.js`, model notebooks, version files and release notes as appropriate.

**Interfaces:** Existing HF publication tools and data-revision behavior. Display accepted training detail compactly; keep legacy catalogue filters and unavailable fields usable.

- [ ] Synchronize registry, notebooks, `.pt` metadata, per-clock configs/cards, aggregate and catalogue. For metadata-only changes, verify every tensor-storage hash; for numerical changes, verify independent author-reference fixtures and clearly record affected outputs/version requirements.
- [ ] Run `uv run pytest -m 'not full_catalog and not online'`, Ruff, catalogue JavaScript checks, targeted notebook execution and the 240-clock consistency validator. Run affected real-asset prediction tests with explicit recorded skip counts; do not claim unavailable CUDA/author data were tested.
- [ ] Publish model/sidecar changes before aggregate metadata; preserve original licenses and existing release tags. Verify anonymous fresh downloads, changed metadata and affected public-API predictions, including shared-repository fallback where files exist.
- [ ] Commit/push reviewable changes and merge after CI. If numerical/runtime code changes are accepted, release the next patch after the latest published version at execution time; metadata-only work alone need not trigger PyPI publication. Follow `docs/development/README.md` and avoid release targets that silently upload/tag the entire catalogue.
- [ ] Verify the published catalogue/tutorials, removed proteomic navigation entry, accepted training metadata displays, and a clean installation's affected predictions. Close D01–D15 individually; the release report must name any remaining source/asset gaps.

## Self-review

The plan covers every flagged field group, all prior unresolved implementation questions, the five partial reviews, the accepted training metadata additions and both documentation changes. New input-contract fields are excluded. Feature-count fields are deferred; new compatibility, citation-role/provenance and demographic-distribution fields are excluded. It preserves the distinction between discovery, primary-source confirmation and executable numerical validation. The source review recommends adding evidence-backed detail rather than replacing correct pyaging metadata with a larger but less traceable upstream table.

## Execution update, 2 October 2026

Completed the requested Astra tutorial replacement/RNA refresh and six Luna review batches covering all 240 clocks/citations. Implemented nine proxy-label corrections and two citation corrections, with unchanged scores; synchronized Hugging Face and the catalogue. Existing documentation build issues were fixed. See the [execution report](../../development/clock-review-follow-up-2026-10-02/README.md) for validation and source gaps.

The numerical investigations and a separate structured training-data migration remain open. The reviewer-generated McCartney numerical/unit proposals are explicitly deferred rather than silently applied as metadata changes. No excluded input-contract fields were introduced.
