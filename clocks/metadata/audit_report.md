# Clock Metadata Source Audit

## Scope

173 clocks across 71 DOI families.

## Controlled-vocabulary decisions

- Tissue, platform, prediction target, training target, and unit use multi-valued controlled terms.
- Source-specific detail remains in the evidence ledger.

## Evidence status counts

- author-confirmed: 6
- code-confirmed: 325
- paper-confirmed: 2378
- supplement-confirmed: 59
- unresolved: 0

## Access issues

- altumage: EPIC compatibility is explicit, but the training datasets themselves used only 27K and 450K arrays; platform therefore excludes EPIC under the audit's training/selection rule.
- bocklandt: The paper's validated predictor uses two CpGs, whereas the packaged Pyaging Bocklandt model exposes only cg09809672. The metadata proposal follows the paper and flags the implementation mismatch.
- bohlin: The official paper and supplement report a 96-CpG primary ultrasound model, but the third-party coefficient file used by pyaging contains 251 CpGs.
- bohlin: No author-hosted 251-CpG coefficient model was located; the implemented coefficient set cannot be mapped to one of the paper’s six reported LASSO variants.
- cpgptgrimage3: The 2024 manuscript anonymizes mortality-cohort identities and does not report their age ranges or exact array mix.
- cpgptgrimage3: The names CpGPTGrimAge3 and CpGPTPCGrimAge3 and their compact feature formulas are documented in author-maintained implementation code/tutorials but not described by name in the assigned manuscript.
- cpgptpcgrimage3: The 2024 manuscript anonymizes mortality-cohort identities and does not report their age ranges or exact array mix.
- cpgptpcgrimage3: The names CpGPTGrimAge3 and CpGPTPCGrimAge3 and their compact feature formulas are documented in author-maintained implementation code/tutorials but not described by name in the assigned manuscript.
- cvdwesterman: Paper final cross-study learner uses a 1,305-CpG union, whereas the packaged pyaging score has 235 coefficients and a sigmoid; the mapping provenance is not documented by the paper.
- deconvolutebloodepicbcell: Publisher version-of-record methods required subscription; the complete open bioRxiv manuscript and author repository were inspected instead.
- deconvolutebloodepicbcell: The assigned paper does not report the age range of purified EPIC reference-panel donors; population remains age-unspecified.
- deconvolutebloodepicbcell: The reporting-summary supplement contains no model coefficients or reference-donor age details.
- deconvolutebloodepiccd4tcell: Publisher version-of-record methods required subscription; the complete open bioRxiv manuscript and author repository were inspected instead.
- deconvolutebloodepiccd4tcell: The assigned paper does not report the age range of purified EPIC reference-panel donors; population remains age-unspecified.
- deconvolutebloodepiccd4tcell: The reporting-summary supplement contains no model coefficients or reference-donor age details.
- deconvolutebloodepiccd8tcell: Publisher version-of-record methods required subscription; the complete open bioRxiv manuscript and author repository were inspected instead.
- deconvolutebloodepiccd8tcell: The assigned paper does not report the age range of purified EPIC reference-panel donors; population remains age-unspecified.
- deconvolutebloodepiccd8tcell: The reporting-summary supplement contains no model coefficients or reference-donor age details.
- deconvolutebloodepicmonocyte: Publisher version-of-record methods required subscription; the complete open bioRxiv manuscript and author repository were inspected instead.
- deconvolutebloodepicmonocyte: The assigned paper does not report the age range of purified EPIC reference-panel donors; population remains age-unspecified.
- deconvolutebloodepicmonocyte: The reporting-summary supplement contains no model coefficients or reference-donor age details.
- deconvolutebloodepicneutrophil: Publisher version-of-record methods required subscription; the complete open bioRxiv manuscript and author repository were inspected instead.
- deconvolutebloodepicneutrophil: The assigned paper does not report the age range of purified EPIC reference-panel donors; population remains age-unspecified.
- deconvolutebloodepicneutrophil: The reporting-summary supplement contains no model coefficients or reference-donor age details.
- deconvolutebloodepicnkcell: Publisher version-of-record methods required subscription; the complete open bioRxiv manuscript and author repository were inspected instead.
- deconvolutebloodepicnkcell: The assigned paper does not report the age range of purified EPIC reference-panel donors; population remains age-unspecified.
- deconvolutebloodepicnkcell: The reporting-summary supplement contains no model coefficients or reference-donor age details.
- downsyndrome: The paper reports an EWAS, not a trained prediction clock; pyaging derives a zero-intercept weighted sum from Supplementary Data 2 beta_overall estimates.
- dunedinpoam38: The paper's scientific unit is a rate (years of biological aging per calendar year), correcting the current generic “years” label.
- dunedinpoam38: EPIC validation/compatibility was not treated as a training platform.
- epitoc3: The assigned 2020 Genome Medicine paper defines epiTOC2 but does not name or define epiTOC3; the 170-CpG epiTOC3 model is only confirmed in the official EpiMitClocks code.
- garagnani: No official author prediction code or published one-CpG linear formula was located; implementation-specific output, unit, model, and feature-count claims remain unresolved.
- hannum: Original Molecular Cell Table S3 supplement was not directly accessible; NCBI OA API reports PMC3780611 as not open access. Paper text and packaged model code were accessible.
- hepatoxu: The publisher main-text PDF required subscription; bibliographic facts came from the publisher page and model details from the publicly accessible original supplement.
- hypoclock: The 2018 Nature Genetics/PMC supplementary attachment bundle was not retrievable through the Europe PMC bundle endpoint (HTTP 404), and direct legacy attachment paths returned HTML rather than the files. The structured full text, supplementary captions, authors’ official probe annotations, the 2020 HypoClock supplement/code, and the packaged model were available and used.
- lin: The Lin supplementary PDF downloaded successfully, but two independent local PDF engines hung during text extraction/rendering; coefficient count and executable form were therefore cross-checked against the packaged pyaging model, while the paper HTML supplied the training and outcome evidence.
- reedbmi: No official author clock implementation was identified. The downstream coefficient file used by pyaging contains 134 CpGs plus a zero intercept, whereas the paper describes a 135-CpG methylation score.
- replitalinorm: The paper does not name the 218-CpG upstream normalizer “RepliTaliNorm” as a standalone clock; Pyaging exposes it separately. Its metadata therefore describes its actual construction role, not the final RepliTali target.
- senchronoage: The paper/supplement and methylCIPHER README report 188 CpGs, but the official author-lab coefficient object and packaged Pyaging model both contain 187 methylation inputs. n_features records the executable-model count.
- sencultureage: The paper/supplement and methylCIPHER README report 141 CpGs, but the official author-lab coefficient object and packaged Pyaging model both contain 142 methylation inputs. n_features records the executable-model count.
- sencultureage: The paper does not state a user-facing unit; inspection of the no-postprocess Pyaging linear model makes the returned scale a log-odds score.
- senmortalityage: The paper/supplement and methylCIPHER README report 89 CpGs, but the official author-lab coefficient object and packaged Pyaging model both contain 91 methylation inputs. n_features records the executable-model count.
- senmortalityage: The paper does not label a user-facing unit; the no-postprocess Pyaging model returns the Cox linear predictor (log-hazard score), not a hazard ratio.
- twelvecelldeconvolutebloodepicbas: Publisher version-of-record methods required subscription; the open bioRxiv manuscript and author repository were inspected instead.
- twelvecelldeconvolutebloodepicbas: The assigned paper does not describe the twelve-cell model’s feature-selection optimization or reference-donor age range; training_target and population remain unresolved.
- twelvecelldeconvolutebloodepicbas: The reporting-summary supplement contains no model coefficients or twelve-cell construction details.
- twelvecelldeconvolutebloodepicbmem: Publisher version-of-record methods required subscription; the open bioRxiv manuscript and author repository were inspected instead.
- twelvecelldeconvolutebloodepicbmem: The assigned paper does not describe the twelve-cell model’s feature-selection optimization or reference-donor age range; training_target and population remain unresolved.
- twelvecelldeconvolutebloodepicbmem: The reporting-summary supplement contains no model coefficients or twelve-cell construction details.
- twelvecelldeconvolutebloodepicbnv: Publisher version-of-record methods required subscription; the open bioRxiv manuscript and author repository were inspected instead.
- twelvecelldeconvolutebloodepicbnv: The assigned paper does not describe the twelve-cell model’s feature-selection optimization or reference-donor age range; training_target and population remain unresolved.
- twelvecelldeconvolutebloodepicbnv: The reporting-summary supplement contains no model coefficients or twelve-cell construction details.
- twelvecelldeconvolutebloodepiccd4mem: Publisher version-of-record methods required subscription; the open bioRxiv manuscript and author repository were inspected instead.
- twelvecelldeconvolutebloodepiccd4mem: The assigned paper does not describe the twelve-cell model’s feature-selection optimization or reference-donor age range; training_target and population remain unresolved.
- twelvecelldeconvolutebloodepiccd4mem: The reporting-summary supplement contains no model coefficients or twelve-cell construction details.
- twelvecelldeconvolutebloodepiccd4nv: Publisher version-of-record methods required subscription; the open bioRxiv manuscript and author repository were inspected instead.
- twelvecelldeconvolutebloodepiccd4nv: The assigned paper does not describe the twelve-cell model’s feature-selection optimization or reference-donor age range; training_target and population remain unresolved.
- twelvecelldeconvolutebloodepiccd4nv: The reporting-summary supplement contains no model coefficients or twelve-cell construction details.
- twelvecelldeconvolutebloodepiccd8mem: Publisher version-of-record methods required subscription; the open bioRxiv manuscript and author repository were inspected instead.
- twelvecelldeconvolutebloodepiccd8mem: The assigned paper does not describe the twelve-cell model’s feature-selection optimization or reference-donor age range; training_target and population remain unresolved.
- twelvecelldeconvolutebloodepiccd8mem: The reporting-summary supplement contains no model coefficients or twelve-cell construction details.
- twelvecelldeconvolutebloodepiccd8nv: Publisher version-of-record methods required subscription; the open bioRxiv manuscript and author repository were inspected instead.
- twelvecelldeconvolutebloodepiccd8nv: The assigned paper does not describe the twelve-cell model’s feature-selection optimization or reference-donor age range; training_target and population remain unresolved.
- twelvecelldeconvolutebloodepiccd8nv: The reporting-summary supplement contains no model coefficients or twelve-cell construction details.
- twelvecelldeconvolutebloodepiceos: Publisher version-of-record methods required subscription; the open bioRxiv manuscript and author repository were inspected instead.
- twelvecelldeconvolutebloodepiceos: The assigned paper does not describe the twelve-cell model’s feature-selection optimization or reference-donor age range; training_target and population remain unresolved.
- twelvecelldeconvolutebloodepiceos: The reporting-summary supplement contains no model coefficients or twelve-cell construction details.
- twelvecelldeconvolutebloodepicmono: Publisher version-of-record methods required subscription; the open bioRxiv manuscript and author repository were inspected instead.
- twelvecelldeconvolutebloodepicmono: The assigned paper does not describe the twelve-cell model’s feature-selection optimization or reference-donor age range; training_target and population remain unresolved.
- twelvecelldeconvolutebloodepicmono: The reporting-summary supplement contains no model coefficients or twelve-cell construction details.
- twelvecelldeconvolutebloodepicneu: Publisher version-of-record methods required subscription; the open bioRxiv manuscript and author repository were inspected instead.
- twelvecelldeconvolutebloodepicneu: The assigned paper does not describe the twelve-cell model’s feature-selection optimization or reference-donor age range; training_target and population remain unresolved.
- twelvecelldeconvolutebloodepicneu: The reporting-summary supplement contains no model coefficients or twelve-cell construction details.
- twelvecelldeconvolutebloodepicnk: Publisher version-of-record methods required subscription; the open bioRxiv manuscript and author repository were inspected instead.
- twelvecelldeconvolutebloodepicnk: The assigned paper does not describe the twelve-cell model’s feature-selection optimization or reference-donor age range; training_target and population remain unresolved.
- twelvecelldeconvolutebloodepicnk: The reporting-summary supplement contains no model coefficients or twelve-cell construction details.
- twelvecelldeconvolutebloodepictreg: Publisher version-of-record methods required subscription; the open bioRxiv manuscript and author repository were inspected instead.
- twelvecelldeconvolutebloodepictreg: The assigned paper does not describe the twelve-cell model’s feature-selection optimization or reference-donor age range; training_target and population remain unresolved.
- twelvecelldeconvolutebloodepictreg: The reporting-summary supplement contains no model coefficients or twelve-cell construction details.
- yingadaptage: The paper reports AdaptAge as 1,000 sites, whereas the packaged coefficient model has 999 CpG inputs. The reason for the discrepancy is unresolved; n_features records actual packaged methylation inputs.
- yingcausage: The paper reports CausAge as 586 sites, whereas the packaged coefficient model has 585 CpG inputs. The reason for the discrepancy is unresolved; n_features records actual packaged methylation inputs.
- yingdamage: The paper reports DamAge as 1,090 sites, whereas the packaged coefficient model has 1,089 CpG inputs. The reason for the discrepancy is unresolved; n_features records actual packaged methylation inputs.
- zhangmortality: The paper’s score is an integer count of ten CpGs crossing cohort-specific quartile thresholds; pyaging instead implements a continuous weighted sum of raw beta values.
- zhangmortality: No author-provided code supporting the pyaging continuous formula was located; the paper states that SAS analysis code is available only on request.

## Source contradictions and adjudications

- cpgptgrimage3: cpgptgrimage3 and cpgptpcgrimage3 were trained in blood with the 450k array to predict mortality. it was trained in the FHS cohort with methylation data. (Direct author clarification in Codex task, 2026-07-18). The paper anonymizes the mortality cohorts; direct author clarification identifies the Framingham Heart Study (FHS) adult whole-blood methylation cohort measured on the Illumina 450K array.
- cpgptpcgrimage3: cpgptgrimage3 and cpgptpcgrimage3 were trained in blood with the 450k array to predict mortality. it was trained in the FHS cohort with methylation data. (Direct author clarification in Codex task, 2026-07-18). The paper anonymizes the mortality cohorts; direct author clarification identifies the Framingham Heart Study (FHS) adult whole-blood methylation cohort measured on the Illumina 450K array.

## Changed-value summary

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
- n_features: 7

## Validation

- 173 clocks and 2768 audited fields materialized with no unresolved evidence.

## Hugging Face publication

- Not performed by this one-off materialization step.
