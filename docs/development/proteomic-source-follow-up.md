# Original-author proteomic source follow-up

Checked 2026-10-02 after the 0.5.5 release. Research-use restrictions are recorded
in metadata; they are not treated as a reason to omit an otherwise reproducible
model. This audit supersedes the PAOPAC availability assessment in the
[0.5.5 source audit](proteomic-source-audit-0.5.5.md).

## Healthspan Proteomic Score

`hps` implements [Kuo et al., PNAS (2025)](https://doi.org/10.1073/pnas.2414086122)
from [the original HPS repository](https://github.com/kuo-lab-uchc/HPS/tree/6eec9ac5a091e34fece69fa4652ebd9af22c1839).
The builder pins the commit and SHA-256 of `HPS.R`, extracts its fitted numerical
parameters, and constructs an independent Torch implementation. The
[August 2025 correction](https://doi.org/10.1073/pnas.2520058122) concerns author
affiliations, not the model.

The 87 inputs are 86 Olink Explore 3072 plasma NPX proteins and age in years.
`nppb` and `ntprobnp` are separate assays. Output is a probability from zero to
one of remaining free of the paper's disease/death endpoint over ten years;
higher is healthier. It is not years of healthy life or an age residual.

The original function performs no extra scaling or imputation. Absent required
columns prevent scoring, and present NaNs propagate. The paper's upstream
multiUS kNN imputation (k=10) is a separate preparation step. NPX represents
normalized log2 abundance; matching protein names alone does not establish
cross-platform, batch, serum/plasma or panel comparability.

The native R 4.5.3 reference executes the unmodified author function on ten
synthetic profiles, including explicit NA, reversed features and extreme
probability limits. The maximum absolute difference is
`4.440892098500626e-16` probability units. Fixtures and source hashes are in
`tests/data/hps/`; no participant data or author R source is vendored.

The repository has no code license. The article is CC BY-NC-ND 4.0; pyaging
records research use and original terms without relicensing the author code or
claiming author approval. OpenAlex reported 21 citations on the audit date.

## PAOPAC availability correction

The original [v1.0.0 release](https://github.com/JackieHanLab/PAOPAC/releases/tag/v1.0.0)
contains the actual trained models. The complete `model.bin` is 41,668,984 bytes,
with SHA-256
`5b0c5dadbac0fc382c51151a8845ff6a7ac1da31f89c0579731b02338c10c3f1`.
The earlier audit only obtained a partial download and did not execute the
Windows interface. The complete release and its MIT-licensed loader permit a
portable extraction of the fitted numerical trees.

The Nature Biotechnology trial's [official supplementary workbook](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41587-026-03286-y/MediaObjects/41587_2026_3286_MOESM3_ESM.xlsx)
identifies `Han_2025(Conventional)` in worksheet S2, cell F4. Its SHA-256 is
`ad7f0d3bc87b7ad5916b03ce2befaea803476716cec74965a4782cdc202a3b0b`.
This establishes the Conventional endpoint rather than an average of organs.

The portable `paopac` model contains the Conventional model's 800 fitted
numerical trees. It accepts 189 protein predictors, the optional protein-matrix
`TDI` input, and chronological `age` for cohort correction. The original MIT
license accompanies the artifact. No tuning objects or participant records are
retained in the converted model.

The released interface exponentiates NPX, fills missing linear abundances with
zero, fits a StandardScaler within the submitted cohort, predicts using the
original trees, and subtracts a LOWESS fit of predicted-minus-chronological age
against chronological age. LOWESS uses `frac=0.75` and `return_sorted=False`.
These details are established by execution of the original compiled interface;
the preprint's full methods were inaccessible during this audit. They must not
be described as verified from the paper. Matching this public release does not
establish that the trial used the identical release or undocumented preparation.

Exact duplicate NPX names are averaged before exponentiation; differently cased
names that collide only after uppercasing are rejected, matching the native
interface. Metadata covariates are not joined to the tree input. In particular,
absent protein-matrix TDI is filled with zero, whereas supplied protein-matrix
TDI undergoes the same exponential transform as the other inputs.

The native reference executes the unchanged Windows CPython 3.9 extension with
the original trees evaluated by LightGBM. Only the pickle loader is replaced by
a safe adapter holding those exact trees. Test fixtures include scaled inputs,
raw tree outputs and final corrected ages for complete data, TDI, missingness,
duplicates with and without NaNs, missing age values, and degenerate cohorts.
The maximum independently reviewed public-pipeline discrepancy is
`1.5e-13` years across batch sizes 1, 7 and 1000. The submitted cohort, not the
minibatch, controls normalization and correction. Raw AnnData and cached model
state remain unchanged. CUDA is supported by the tree implementation but was
not available for this validation.

A singleton returns its supplied chronological age. All-missing protein
cohorts can likewise collapse to chronological ages; equal-age LOWESS ties can
depend on row order. These reproduced author behaviors are not evidence of a
meaningful individual estimate and are documented in the public input guide.

## ProtAge: fitted assets still absent

The [author README](https://github.com/miargentieri/proteomic-age-ukb/blob/7ed719dc0c175a2bf227710bfd85f51f19b4c8b0/README.md),
updated September 28, 2026, still says the ProtAge and ProtAge20 models are not
publicly supplied and a noncommercial Python package is being developed.

The renewed audit checked all current original repository files, branches,
tags and releases, the author's related repositories, public model/package
searches, and the original Nature Medicine supplementary workbook. The workbook
has 22 sheets containing protein lists and downstream association results, not
the fitted trees or training-reference normalization parameters. SHAP values
cannot substitute for the 204-protein LightGBM model.

A working conversion requires that trained model plus its ordered assay names,
fitted scaling and median-centering parameters, missing-value policy and an
input/output reference. Research-only metadata can preserve the terms once
these assets are supplied; it cannot reconstruct them.

## ipfP3GPT: withdrawn checkpoint and RAP execution

The [current author notice](https://github.com/Insilico-org/proteoclock/blob/d078b246c9a834bb1eaa54a79381b6c6c14245f7/README.md)
states that UK Biobank requested removal of the proteomic age checkpoint and
requires registered-researcher execution within UK Biobank RAP.

The renewed audit inspected the public OSF project through its API, original
journal inference notebooks, current official repositories and model listings.
The supplementary age notebook needs `best_clock.pt`, a UKB scaler and feature
ordering. The OSF package and notebooks predate the June 2026 withdrawal; no
replacement release, executable RAP endpoint, applet or project/file identifier
was found. Withdrawn weights were not recovered from older copies.

The public Precious3GPT model is a distinct generative omics transformer, not a
replacement for this proteomic age network. An authorized RAP execution
context with the exact model and preparation assets, or a new author release,
is needed for a testable implementation. A research-only flag does not provide
that access. No incomplete predictor is registered under either missing name.
