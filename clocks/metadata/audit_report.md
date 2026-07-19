# Clock Metadata Source Audit

## Scope

173 clocks across 71 originally assigned DOI families; 72 canonical DOI families
after correcting MammalianLifeSpan to the version-of-record DOI.

## Controlled-vocabulary decisions

- Tissue, platform, prediction target, training target, and unit use multi-valued controlled terms.
- Source-specific detail remains in the evidence ledger.

## Evidence status counts

- author-confirmed: 6
- code-confirmed: 313
- paper-confirmed: 2390
- supplement-confirmed: 59
- unresolved: 0

## Access issues

- altumage: First-pass source limitation (retained for provenance; final metadata evidence resolved): EPIC compatibility is explicit, but the training datasets themselves used only 27K and 450K arrays; platform therefore excludes EPIC under the audit's training/selection rule.
- bocklandt: First-pass source limitation (retained for provenance; final metadata evidence resolved): The paper's validated predictor uses two CpGs, whereas the packaged Pyaging Bocklandt model exposes only cg09809672. The metadata proposal follows the paper and flags the implementation mismatch.
- bohlin: First-pass source limitation (retained for provenance; final metadata evidence resolved): The official paper and supplement report a 96-CpG primary ultrasound model, but the third-party coefficient file used by pyaging contains 251 CpGs.
- bohlin: First-pass source limitation (retained for provenance; final metadata evidence resolved): No author-hosted 251-CpG coefficient model was located; the implemented coefficient set cannot be mapped to one of the paper’s six reported LASSO variants.
- cpgptgrimage3: First-pass source limitation (retained for provenance; final metadata evidence resolved): The 2024 manuscript anonymizes mortality-cohort identities and does not report their age ranges or exact array mix.
- cpgptgrimage3: First-pass source limitation (retained for provenance; final metadata evidence resolved): The names CpGPTGrimAge3 and CpGPTPCGrimAge3 and their compact feature formulas are documented in author-maintained implementation code/tutorials but not described by name in the assigned manuscript.
- cpgptpcgrimage3: First-pass source limitation (retained for provenance; final metadata evidence resolved): The 2024 manuscript anonymizes mortality-cohort identities and does not report their age ranges or exact array mix.
- cpgptpcgrimage3: First-pass source limitation (retained for provenance; final metadata evidence resolved): The names CpGPTGrimAge3 and CpGPTPCGrimAge3 and their compact feature formulas are documented in author-maintained implementation code/tutorials but not described by name in the assigned manuscript.
- cvdwesterman: First-pass source limitation (retained for provenance; final metadata evidence resolved): Paper final cross-study learner uses a 1,305-CpG union, whereas the packaged pyaging score has 235 coefficients and a sigmoid; the mapping provenance is not documented by the paper.
- deconvolutebloodepicbcell: First-pass source limitation (retained for provenance; final metadata evidence resolved): Publisher version-of-record methods required subscription; the complete open bioRxiv manuscript and author repository were inspected instead.
- deconvolutebloodepicbcell: First-pass source limitation (retained for provenance; final metadata evidence resolved): The reporting-summary supplement contains no model coefficients or reference-donor age details.
- deconvolutebloodepiccd4tcell: First-pass source limitation (retained for provenance; final metadata evidence resolved): Publisher version-of-record methods required subscription; the complete open bioRxiv manuscript and author repository were inspected instead.
- deconvolutebloodepiccd4tcell: First-pass source limitation (retained for provenance; final metadata evidence resolved): The reporting-summary supplement contains no model coefficients or reference-donor age details.
- deconvolutebloodepiccd8tcell: First-pass source limitation (retained for provenance; final metadata evidence resolved): Publisher version-of-record methods required subscription; the complete open bioRxiv manuscript and author repository were inspected instead.
- deconvolutebloodepiccd8tcell: First-pass source limitation (retained for provenance; final metadata evidence resolved): The reporting-summary supplement contains no model coefficients or reference-donor age details.
- deconvolutebloodepicmonocyte: First-pass source limitation (retained for provenance; final metadata evidence resolved): Publisher version-of-record methods required subscription; the complete open bioRxiv manuscript and author repository were inspected instead.
- deconvolutebloodepicmonocyte: First-pass source limitation (retained for provenance; final metadata evidence resolved): The reporting-summary supplement contains no model coefficients or reference-donor age details.
- deconvolutebloodepicneutrophil: First-pass source limitation (retained for provenance; final metadata evidence resolved): Publisher version-of-record methods required subscription; the complete open bioRxiv manuscript and author repository were inspected instead.
- deconvolutebloodepicneutrophil: First-pass source limitation (retained for provenance; final metadata evidence resolved): The reporting-summary supplement contains no model coefficients or reference-donor age details.
- deconvolutebloodepicnkcell: First-pass source limitation (retained for provenance; final metadata evidence resolved): Publisher version-of-record methods required subscription; the complete open bioRxiv manuscript and author repository were inspected instead.
- deconvolutebloodepicnkcell: First-pass source limitation (retained for provenance; final metadata evidence resolved): The reporting-summary supplement contains no model coefficients or reference-donor age details.
- downsyndrome: First-pass source limitation (retained for provenance; final metadata evidence resolved): The paper reports an EWAS, not a trained prediction clock; pyaging derives a zero-intercept weighted sum from Supplementary Data 2 beta_overall estimates.
- dunedinpoam38: First-pass source limitation (retained for provenance; final metadata evidence resolved): The paper's scientific unit is a rate (years of biological aging per calendar year), correcting the current generic “years” label.
- dunedinpoam38: First-pass source limitation (retained for provenance; final metadata evidence resolved): EPIC validation/compatibility was not treated as a training platform.
- epitoc3: First-pass source limitation (retained for provenance; final metadata evidence resolved): The assigned 2020 Genome Medicine paper defines epiTOC2 but does not name or define epiTOC3; the 170-CpG epiTOC3 model is only confirmed in the official EpiMitClocks code.
- garagnani: First-pass source limitation (retained for provenance; final metadata evidence resolved): The source paper did not publish a one-CpG prediction equation and no official author prediction code was located; packaged output semantics were therefore established from the downstream coefficient artifact.
- hannum: First-pass source limitation (retained for provenance; final metadata evidence resolved): Original Molecular Cell Table S3 supplement was not directly accessible; NCBI OA API reports PMC3780611 as not open access. Paper text and packaged model code were accessible.
- hepatoxu: First-pass source limitation (retained for provenance; final metadata evidence resolved): The publisher main-text PDF required subscription; bibliographic facts came from the publisher page and model details from the publicly accessible original supplement.
- hypoclock: First-pass source limitation (retained for provenance; final metadata evidence resolved): The 2018 Nature Genetics/PMC supplementary attachment bundle was not retrievable through the Europe PMC bundle endpoint (HTTP 404), and direct legacy attachment paths returned HTML rather than the files. The structured full text, supplementary captions, authors’ official probe annotations, the 2020 HypoClock supplement/code, and the packaged model were available and used.
- lin: First-pass source limitation (retained for provenance; final metadata evidence resolved): The Lin supplementary PDF downloaded successfully, but two independent local PDF engines hung during text extraction/rendering; coefficient count and executable form were therefore cross-checked against the packaged pyaging model, while the paper HTML supplied the training and outcome evidence.
- reedbmi: First-pass source limitation (retained for provenance; final metadata evidence resolved): No official author clock implementation was identified. A parsed review of the downstream coefficient file used by pyaging confirms 135 CpG rows plus a zero intercept, matching the paper's 135-CpG methylation score; the earlier 134 count came from file line-ending behavior.
- replitalinorm: First-pass source limitation (retained for provenance; final metadata evidence resolved): The paper does not name the 218-CpG upstream normalizer “RepliTaliNorm” as a standalone clock; Pyaging exposes it separately. Its metadata therefore describes its actual construction role, not the final RepliTali target.
- senchronoage: First-pass source limitation (retained for provenance; final metadata evidence resolved): The paper/supplement and methylCIPHER README report 188 CpGs, but the official author-lab coefficient object and packaged Pyaging model both contain 187 methylation inputs. n_features records the executable-model count.
- sencultureage: First-pass source limitation (retained for provenance; final metadata evidence resolved): The paper/supplement and methylCIPHER README report 141 CpGs, but the official author-lab coefficient object and packaged Pyaging model both contain 142 methylation inputs. n_features records the executable-model count.
- sencultureage: First-pass source limitation (retained for provenance; final metadata evidence resolved): The paper does not state a user-facing unit; inspection of the no-postprocess Pyaging linear model makes the returned scale a log-odds score.
- senmortalityage: First-pass source limitation (retained for provenance; final metadata evidence resolved): The paper/supplement and methylCIPHER README report 89 CpGs, but the official author-lab coefficient object and packaged Pyaging model both contain 91 methylation inputs. n_features records the executable-model count.
- senmortalityage: First-pass source limitation (retained for provenance; final metadata evidence resolved): The paper does not label a user-facing unit; the no-postprocess Pyaging model returns the Cox linear predictor (log-hazard score), not a hazard ratio.
- yingadaptage: First-pass source limitation (retained for provenance; final metadata evidence resolved): The paper reports AdaptAge as 1,000 sites, whereas the packaged coefficient model has 999 CpG inputs. The reason for the discrepancy is unresolved; n_features records actual packaged methylation inputs.
- yingcausage: First-pass source limitation (retained for provenance; final metadata evidence resolved): The paper reports CausAge as 586 sites, whereas the packaged coefficient model has 585 CpG inputs. The reason for the discrepancy is unresolved; n_features records actual packaged methylation inputs.
- yingdamage: First-pass source limitation (retained for provenance; final metadata evidence resolved): The paper reports DamAge as 1,090 sites, whereas the packaged coefficient model has 1,089 CpG inputs. The reason for the discrepancy is unresolved; n_features records actual packaged methylation inputs.
- zhangmortality: First-pass source limitation (retained for provenance; final metadata evidence resolved): The paper’s score is an integer count of ten CpGs crossing cohort-specific quartile thresholds; pyaging instead implements a continuous weighted sum of raw beta values.
- zhangmortality: First-pass source limitation (retained for provenance; final metadata evidence resolved): No author-provided code supporting the pyaging continuous formula was located; the paper states that SAS analysis code is available only on request.

## Source contradictions and adjudications

- cpgptgrimage3: cpgptgrimage3 and cpgptpcgrimage3 were trained in blood with the 450k array to predict mortality. it was trained in the FHS cohort with methylation data. (Direct author clarification in Codex task, 2026-07-18). The paper anonymizes the mortality cohorts; direct author clarification identifies the Framingham Heart Study (FHS) adult whole-blood methylation cohort measured on the Illumina 450K array.
- cpgptpcgrimage3: cpgptgrimage3 and cpgptpcgrimage3 were trained in blood with the 450k array to predict mortality. it was trained in the FHS cohort with methylation data. (Direct author clarification in Codex task, 2026-07-18). The paper anonymizes the mortality cohorts; direct author clarification identifies the Framingham Heart Study (FHS) adult whole-blood methylation cohort measured on the Illumina 450K array.

## Changed-value summary

The reviewed dry-run proposed changes for all 173 clocks exactly once. It
rendered 3,633 curated notebook assignments, including 1,557 controlled-field
assignments with same-line `# Paper:` evidence comments. The five multi-valued
fields changed structurally in every notebook because their prior scalar values
become Python lists; the evidence audit's semantic change counts remain listed
below.

Across serialized model metadata, the proposal contains 2,250 key additions,
740 value changes, and no removals. Across the aggregate metadata object, it
contains 173 key additions, 1,668 value changes, and no removals. The aggregate
runtime fields are unchanged. In serialized model metadata, `version` changes
from the prior null metadata value to the registry/runtime value rather than
being removed.

Notable scientific corrections include limiting CpGPTGrimAge3 and
CpGPTPCGrimAge3 training to FHS whole blood on Illumina 450K with mortality as
the training target; expressing DunedinPoAm38 in biological years per
chronological year; expressing SenMortalityAge as a log-hazard score; and
recording ZhangMortality as the paper's unitless weighted score while retaining
the documented implementation discrepancy. The final runtime-consistency review
also corrected OcampoATAC2 from the 228 predictors used by OcampoATAC1 to the
380 non-zero OCR coefficients in the authors' separate all-samples model, and
corrected ReedBMI from 134 to the 135 CpG coefficient rows confirmed by both the
paper and parsed coefficient table.

The final evidence-sample review also updated MammalianLifeSpan from the 2023
preprint to the 2024 *Science Advances* version of record and identified its
training assay as Horvath MammalMethylChip40; corrected the Neu-Sin citation to
pages 13452–13504 and confirmed its 672 non-intercept CpGs from the official
coefficient object; and documented that IntrinClock's paper reports 381 CpGs
while both the official `lambda.min` model and pyaging use an identical
380-probe set. For all 12 TwelveCell outputs, the training target is now the
paper-supported known artificial-mixture cell-type proportions. The separate
10-positive/10-negative contrast pattern is explicitly limited to Biolearn's
undocumented 240-row replacement artifact, rather than being presented as the
published 1,200-CpG IDOL optimization target. The open 2022 primary paper and
official coefficient/code artifacts supplied the necessary evidence; these
final sample corrections have no remaining paper-access issue.

## Field change counts

- data_type: 169
- species: 11
- year: 22
- citation: 167
- doi: 25
- notes: 171
- tissue: 107
- predicts: 120
- training_target: 123
- unit: 78
- model_type: 173
- platform: 99
- population: 147
- journal: 45
- last_author: 39
- n_features: 6

## Validation

- 173 clocks and 2768 audited fields materialized with no unresolved evidence.
- The dry-run matched all 173 registry, ledger, notebook, weight, aggregate, and
  baseline-fingerprint clock identities.
- All 173 prediction-state fingerprints matched the immutable baseline.
- SHA-256 bytes, `readlink` targets, and mutation-relevant `lstat` fields
  (`st_dev`, `st_ino`, `st_mode`, `st_nlink`, `st_uid`, `st_gid`, `st_size`,
  `st_mtime_ns`, `st_ctime_ns`, and `st_flags`) were unchanged for all 347
  repository artifacts (173 notebooks, 173 weights, and one aggregate). Access
  time is intentionally excluded because verification reads can update it.
- All controlled values passed the canonical vocabulary and resolved-evidence
  checks; all 1,557 paper comments are single physical lines (maximum rendered
  assignment length: 277 characters).
- The final evidence-sample correction chain was revalidated across the
  vocabulary, registry, evidence ledger, source notebooks, serialized model
  metadata, aggregate catalogue, and generated documentation artifacts.

## Hugging Face publication

- Not performed by this one-off materialization step.
