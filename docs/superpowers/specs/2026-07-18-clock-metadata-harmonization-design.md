# Clock Metadata Harmonization and Source Audit — Design Spec

**Date:** 2026-07-18
**Status:** Approved in conversation
**Scope:** All 173 model objects currently distributed by pyaging

## Goal

Audit every clock's curated metadata against its original publication and
supporting sources, replace inconsistent free text with controlled,
multi-valued terms, preserve source-specific detail in the implementation
notebooks and an evidence ledger, publish the corrected model objects and
aggregate metadata, and rebuild the Clock Explorer so the harmonized values are
easy to filter.

The audit covers every package model, including standalone age estimators,
component biomarkers, cell-deconvolution models, PC variants, sex-specific
models, disease classifiers, and other non-age outputs.

## Current State

The package contains 173 serialized model objects associated with 71 unique DOI
values. The existing aggregate metadata has complete keys for the main curated
fields, but several fields contain highly variable prose:

- `tissue` has 94 distinct values;
- `predicts` has 124 distinct values;
- `unit` has 19 values whose semantics and specificity vary;
- `platform` has 11 values, some of which combine multiple platforms in one
  string.

The documentation generator reads `all_clock_metadata.pt` from Hugging Face and
emits static JSON and CSV. The Clock Explorer currently treats categorical
values as scalars. The update script treats the aggregate metadata as the
curated source for most fields while obtaining runtime fields from the
serialized model objects. The notebooks also contain metadata assignments, so
the same information is represented in several places without a reviewable
text registry.

## Architecture

### Canonical registry

Add a committed, human-reviewable JSON registry under `clocks/metadata/`.
It contains one record for each of the 173 clock names and every curated
metadata field used by the package and documentation. It becomes the canonical
review artifact for this migration.

Add a separate committed controlled-vocabulary file beside the registry. It
defines the allowed values, capitalization, aliases, and field-specific
semantics. At minimum, it governs:

- data type;
- species;
- training tissue;
- training platform;
- prediction target;
- training target;
- prediction unit;
- model type;
- training population categories.

The existing public metadata keys remain in place wherever possible. This work
changes controlled categorical values from scalar strings to arrays rather than
renaming established fields unnecessarily.

### Evidence ledger

Add a committed, machine-readable evidence ledger with field-level provenance.
Every clock and audited field records:

- the exact source wording or a concise faithful transcription;
- the normalized controlled value or values;
- DOI and source URL;
- source type: paper, supplement, or official author code/model;
- page, section, table, figure, or file location when available;
- evidence status;
- reviewer assignment;
- an explanatory note when interpretation was required.

Allowed evidence statuses are:

- `paper-confirmed`;
- `supplement-confirmed`;
- `code-confirmed`;
- `unresolved`.

An inference may be recorded in the explanatory note, but it remains
`unresolved` unless the user explicitly accepts it. Unresolved values are
blocked from publication.

### One-off synchronization

Use a temporary migration utility to apply the approved registry to notebooks,
serialized model objects, and the aggregate metadata. The utility is not a
maintained package feature and is removed after the migration. The committed
registry, vocabulary, evidence ledger, notebooks, model objects, tests, and
aggregate remain.

The utility edits only metadata code cells in notebooks. It does not execute
notebooks, download training data, rebuild coefficients, or refit models. For
the controlled fields, each metadata assignment retains the source-specific
detail as an inline comment on the same line:

```python
model.metadata["tissue"] = ["whole blood"]  # Paper: peripheral whole-blood leukocytes from FHS
```

The utility patches only metadata in `.pt` objects. It captures a logical
fingerprint of each object's class, preprocess/postprocess configuration,
reference values, feature identifiers, coefficients, tensors, and other
prediction-relevant state before the mutation and requires the same fingerprint
afterward.

## Metadata Semantics

### Training tissue

`tissue` means the biological material used to fit the model or select its
features. It does not include a tissue merely because the paper validated the
model there or because pyaging can technically apply it there.

The value is always an array of controlled terms. Models trained across tissues
list the source-confirmed tissues. Broad pan-tissue models may additionally
include the `multi-tissue` umbrella term to support convenient filtering, but
the umbrella does not replace the confirmed tissue terms.

### Training platform

`platform` means the assay platforms that supplied data used for feature
selection or model fitting. It excludes platforms mentioned only for
validation, transfer, imputation, or technical compatibility.

The value is always an array of controlled terms. Combined strings such as
`Illumina 27K/450K/EPIC` become separate values.

### Prediction target

`predicts` identifies the scientific construct represented by the value
returned to the user. It uses stable, filterable terms such as `chronological
age`, `gestational age`, `mortality risk`, `pace of aging`, `grip strength`,
`interleukin-6`, and `cell-type proportion`. It is always an array.

Clock names, sex strata, biomarker transformations, and other exact detail stay
in the notebook comment, evidence ledger, and notes rather than generating
near-duplicate filter terms.

### Training target

Add `training_target` when it prevents a meaningful conflation between the
outcome used for fitting and the meaning of the returned value. For example, a
clock may be trained using mortality or time-to-event outcomes but return an
age-scaled prediction.

`training_target` is always an array. It may match `predicts` for direct
regression clocks.

### Prediction unit

`unit` means the physical or statistical unit of the value returned by the
pyaging model after postprocessing. It does not mean the unit suggested by the
clock's name, the raw regression scale before postprocessing, or the unit of an
intermediate training variable.

It is always an array for schema consistency. Controlled examples include
`years`, `months`, `weeks`, `days`, `grams per milliliter`, `meters per second`,
`proportion`, `probability`, `hazard score`, `population doublings`, and
`unitless`. Exact transformations such as logarithms and score definitions are
retained in comments, notes, and evidence.

### Other audited fields

The source audit also verifies data type, species, training population, model
type, feature count, publication year, citation, DOI, journal, last author, and
explanatory notes. The evidence ledger records these checks even when no value
changes.

## Paper Audit Workflow

### Assignment

Group all models by their 71 DOI values. Keep every model from one paper in the
same assignment. Parallel subagents receive non-overlapping paper batches and
write separate ledger shards; they do not edit notebooks, model objects, or
shared metadata.

Each subagent receives:

- its DOI and clock-name assignments;
- current metadata for those clocks;
- the controlled-vocabulary draft;
- the evidence-ledger schema;
- explicit instructions to inspect the original source rather than infer from
  the clock name;
- an output path unique to that assignment.

Every clock receives its own field-level evidence records even when several
clocks share a publication.

### Source precedence

Reviewers inspect the paper's methods and results and the relevant supplement.
When the manuscript does not fully specify a value, they inspect official
author code, coefficient files, or model repositories.

Evidence precedence is:

1. original paper;
2. original supplement;
3. official author code or released model;
4. explicit unresolved inference.

Open-access sources may be inspected with the available research/browser tools.
Comet may be used for papers that require the user's authenticated browser
session. Parallel agents must not contend for the same interactive browser
session.

### Access and ambiguity handling

Record paywalls, authentication failures, unavailable supplements, broken
archives, contradictory sources, and ambiguous definitions in one blocker
report. Do not guess or silently reuse current metadata. No unresolved field is
applied to notebooks, model objects, the aggregate, or Hugging Face.

### Reconciliation

After first-pass review:

1. merge the non-overlapping ledger shards;
2. validate every expected clock and field is present;
3. reconcile synonyms and vocabulary drift across reviewers;
4. adjudicate source contradictions centrally;
5. spot-check large multi-model families and a sample of single-model papers;
6. produce a proposed-change report before mutation.

## Data Flow

The approved data flow is:

```text
papers / supplements / author code
                |
                v
       evidence-ledger shards
                |
                v
   reconciled evidence + blocker report
                |
                v
  controlled vocabulary + canonical registry
                |
        temporary migration utility
        /           |              \
       v            v               v
 notebooks     model objects   aggregate metadata
      \             |             /          \
       \            |            /            v
        v           v           v      local docs validation
       Hugging Face model objects
                 |
                 v
       Hugging Face aggregate (last)
                 |
                 v
     published docs regeneration
```

The temporary migration runs only after the evidence ledger has no
publication-blocking unresolved values or the user has explicitly resolved
them.

## Documentation Changes

Update the Clock Explorer's pure data logic and rendering to understand
array-valued categorical fields:

- facet generation counts each distinct array member once per clock;
- multiple selected values are ORed within a field;
- selections across fields are ANDed;
- search indexes each array member;
- sorting uses a deterministic joined representation;
- tables and cards display a readable, deterministic join;
- CSV uses a stable delimiter that round-trips unambiguously.

Update the documentation generator so validation can consume an explicit local
aggregate path. Its production/default behavior continues to download the
published aggregate from Hugging Face.

Regenerate the committed `clocks.json` and `clock_glossary.csv`, rebuild the
Sphinx documentation, and verify the catalogue at desktop and narrow viewport
widths.

## Validation

### Schema and evidence

Automated checks require:

- exactly 173 unique lowercase clock names matching the distributed weights;
- every required curated field on every registry record;
- arrays for the controlled multi-valued fields;
- every array value present in its controlled vocabulary;
- no empty arrays where a field is required;
- complete field-level evidence coverage;
- valid DOI/source references;
- no unresolved evidence in publishable metadata;
- deterministic serialization and ordering.

### Artifact consistency

Require exact curated-metadata agreement among:

- canonical registry;
- notebook metadata assignments;
- serialized model-object metadata;
- `all_clock_metadata.pt`;
- generated documentation JSON and CSV.

Runtime-only fields such as version, preprocessing, postprocessing, and
reference values retain their established generation rules and are compared
with type-aware normalization.

### Behavioral invariants

Before and after model-object patching:

- logical prediction-state fingerprints are identical;
- feature identifiers and counts are identical;
- coefficients/tensors and reference values are identical;
- preprocess and postprocess configuration is identical;
- gold-standard predictions remain unchanged.

Run focused metadata tests, Clock Explorer JavaScript tests, documentation
generator tests, prediction gold standards, the broader package suite where
practical, and a warning-clean Sphinx build.

## Hugging Face Publication

Hugging Face remains the production data host. Publication occurs only after
local validation:

1. verify the authenticated account and public repository;
2. upload all changed model objects;
3. download and verify remote model metadata and logical fingerprints;
4. upload `all_clock_metadata.pt` last as the publication boundary;
5. download and compare the remote aggregate to the validated local aggregate;
6. record the resulting repository revision;
7. regenerate committed documentation artifacts from the published aggregate
   and rebuild the documentation.

Uploading the aggregate last prevents clients from discovering metadata that
describes model objects not yet available remotely.

## Deliverables

- controlled-vocabulary definition;
- canonical metadata registry covering all 173 clocks;
- field-level evidence ledger;
- source-access and unresolved-issue report;
- proposed-change and final-change summaries;
- notebooks with controlled arrays and precise same-line comments;
- metadata-updated model objects with unchanged prediction state;
- regenerated aggregate metadata;
- array-aware Clock Explorer and documentation artifacts;
- automated consistency and regression tests;
- Hugging Face revision and remote-verification results.

## Non-Goals

- Refitting or redesigning any clock;
- changing coefficients, feature sets, preprocessing, postprocessing, or
  prediction behavior;
- treating validation or compatible platforms as training platforms;
- silently resolving inaccessible or contradictory evidence;
- maintaining the one-off synchronization utility as a package feature;
- redesigning documentation outside metadata presentation and filtering.
