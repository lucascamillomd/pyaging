# Proteomic source audit for 0.5.5

Checked on 2026-10-02. This note records the remaining asset requirements for
ProtAge, PAOPAC, and ipfP3GPT from the
[six-clock clinical-trial comparison](https://doi.org/10.1038/s41587-026-03286-y).
None currently has a verified, portable, publicly distributable implementation
ready to add to pyaging from the sources audited here.

| Clock | Pinned author source | Finding |
| --- | --- | --- |
| ProtAge | [proteomic-age-ukb, `7ed719dc`](https://github.com/miargentieri/proteomic-age-ukb/tree/7ed719dc0c175a2bf227710bfd85f51f19b4c8b0) | Training code and protein annotations are public; the trained clock and fixed reference normalization assets are absent. |
| PAOPAC | [PAOPAC, `a166bc76`](https://github.com/JackieHanLab/PAOPAC/tree/a166bc763ffd1395a96c6557db889706fcc9ef85) | A public model download exists, but the inference implementation is a CPython 3.9 Windows binary. The portable input and prediction contract remains unverified. |
| ipfP3GPT | [proteoclock, `d078b246`](https://github.com/Insilico-org/proteoclock/commit/d078b246c9a834bb1eaa54a79381b6c6c14245f7) | The authors withdrew the weights and require execution within the UK Biobank Research Analysis Platform. |

## ProtAge

The original publication is
[Argentieri et al., Nature Medicine, 2024](https://doi.org/10.1038/s41591-024-03164-7).
It describes a LightGBM model using 204 Olink Explore 3072 plasma proteins.
Use that model description in metadata; the comparison paper calls ProtAge a
deep-learning clock, which conflicts with the original paper and author code.

The pinned [README](https://github.com/miargentieri/proteomic-age-ukb/blob/7ed719dc0c175a2bf227710bfd85f51f19b4c8b0/README.md)
explicitly says ProtAge and ProtAge20 models are unavailable in the repository
and directs early-access requests to the author. The public repository has no
releases or tags. Its tracked files and history contain no trained ProtAge
weights or saved reference scalers. The clinical-trial paper reports obtaining
weights through direct collaboration.

The author [protein table](https://github.com/miargentieri/proteomic-age-ukb/blob/7ed719dc0c175a2bf227710bfd85f51f19b4c8b0/files/ProtAge_proteins_2023-12-29.csv)
contains 204 distinct gene names, protein names, UniProt IDs, and
`mean_SHAP_value`. Those SHAP values describe model contributions; they are not
regression coefficients and cannot reproduce the trained trees.

Normalization also needs an explicit version contract. The public
[training script](https://github.com/miargentieri/proteomic-age-ukb/blob/7ed719dc0c175a2bf227710bfd85f51f19b4c8b0/code/UKB-proteomic-age-model-both-sexes.py#L208)
fits a separate `MinMaxScaler` for each protein, then subtracts the median of
the scaled values. It repeats fitting within the external CKB cohort. The
clinical-trial methods instead specify trained UK Biobank scalers and population
medians. Refitting on each submitted batch would therefore not reproduce that
trial workflow and would make a sample's prediction depend on the other samples.

The [imputation script](https://github.com/miargentieri/proteomic-age-ukb/blob/7ed719dc0c175a2bf227710bfd85f51f19b4c8b0/code/UKB-olink-imputation.py)
uses `miceforest`, one dataset, five iterations, and random seed 3456 during
data preparation. It does not establish an inference policy for absent assay
columns in new data. Neither zero filling nor pyaging's default KNN imputation
can be assumed equivalent.

The [LICENSE](https://github.com/miargentieri/proteomic-age-ukb/blob/7ed719dc0c175a2bf227710bfd85f51f19b4c8b0/LICENSE)
contains standard MIT text, while the README specifies academic/noncommercial
use and a separate commercial license. Obtain explicit model redistribution
terms before publishing any supplied assets.

Required before implementation:

- An authorized trained LightGBM export with the exact ordered assay identifiers.
- The matching per-protein scaling parameters and reference medians, plus the
  assay platform, specimen type, and upstream NPX normalization specification.
- The supported missing-value and absent-protein policy, including any required
  imputer or reference values.
- Redistribution terms and an author-generated input/output fixture that tests
  normalization and prediction together.

## PAOPAC

The original report is
[Xu et al., bioRxiv, 2026](https://doi.org/10.64898/2026.04.24.720503).
The public [v1.0.0 release](https://github.com/JackieHanLab/PAOPAC/releases/tag/v1.0.0)
points to commit `d65afe8610063d6d5ec02decb501c90ef70030c2` and includes
`model.bin`. GitHub reports 41,668,984 bytes and SHA-256
`5b0c5dadbac0fc382c51151a8845ff6a7ac1da31f89c0579731b02338c10c3f1`.
This is the release's reported digest, not a locally verified checksum. A
partial download exposed a Fernet-shaped header; the full download was not
needed for this audit. No decryption or extraction of embedded keys was attempted.
The repository's own `model/model.bin` is only a one-byte placeholder.

The pinned [README](https://github.com/JackieHanLab/PAOPAC/blob/a166bc763ffd1395a96c6557db889706fcc9ef85/README.md)
and [tutorial](https://github.com/JackieHanLab/PAOPAC/blob/a166bc763ffd1395a96c6557db889706fcc9ef85/tutorial.ipynb)
accept samples in rows, protein names in columns, Olink NPX values, and a
sample-aligned metadata frame with numeric `Age`. They call
`OrganAgePredictor().predict(df_protein, df_meta)`. The inference module is
`pre.cp39-win_amd64.pyd`, restricted to CPython 3.9 on 64-bit Windows. There is
no Python implementation in the seven public commits, and the notebook has no
reference predictions.

The public interface does not specify the complete ordered feature list,
assay identifiers, normalization parameters, missing-data behavior, or output
schema. Olink NPX denotes relative expression; a matching protein name alone
does not establish compatibility with another assay platform or concentration
unit. The repository [license](https://github.com/JackieHanLab/PAOPAC/blob/a166bc763ffd1395a96c6557db889706fcc9ef85/LICENSE)
is MIT.

Required before a portable implementation:

- An author-provided source implementation or portable model export with
  documented loading and reuse terms.
- The ordered assay identifiers, platform/version, normalization contract,
  reference statistics, and missing-feature rules.
- The meaning and units of every output, including how chronological age enters
  prediction or any subsequent correction.
- Representative inputs and reference outputs from the released implementation,
  including missing-value and feature-order cases.

## ipfP3GPT

The author's [restriction notice](https://github.com/Insilico-org/proteoclock/blob/d078b246c9a834bb1eaa54a79381b6c6c14245f7/README.md)
states that UK Biobank requested removal of `galkin_2025` / ipfP3GPT weights
because they may disclose participant-level information. The notice limits use
to registered researchers inside UK Biobank RAP and disallows local downloading,
copying, or storage. Commit `d078b246` removes `best_clock.pt` and adds the
corresponding missing-weight error.

Do not recover these weights from older commits, mirrors, or caches, or publish
a local-weight workaround. Future work requires an author-supported RAP workflow
and its preprocessing specification. A distributable local implementation would
require an explicit replacement release that permits that use.
