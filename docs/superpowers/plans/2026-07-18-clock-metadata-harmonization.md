# Clock Metadata Harmonization and Source Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` to execute engineering tasks and
> `superpowers:dispatching-parallel-agents` for the independent paper-review
> batches. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Source-audit and harmonize metadata for all 173 pyaging model objects,
publish the corrected artifacts to Hugging Face, and make the documentation
catalogue filter array-valued controlled terms correctly.

**Architecture:** A committed JSON registry is the curated source of truth, a
controlled-vocabulary JSON file defines allowed normalized values, and a JSONL
evidence ledger records field-level provenance. Twelve non-overlapping
paper-review batches cover the 71 unique DOI families. A temporary migration
utility updates notebook metadata comments and serialized model metadata without
changing prediction state; maintained validators, aggregate generation, and
documentation code keep the committed artifacts consistent.

**Tech Stack:** Python 3.9+, PyTorch, JSON/JSONL, Jupyter notebook JSON, pytest,
vanilla JavaScript, Node `assert`, pandas, Sphinx, Hugging Face Hub CLI

---

## File map

### Create

- `clocks/metadata/controlled_vocabulary.json` — allowed values, aliases, field
  semantics, and display order.
- `clocks/metadata/clock_metadata.json` — canonical curated records for all 173
  clocks.
- `clocks/metadata/evidence_ledger.jsonl` — one source-audit record per clock,
  including field-level evidence.
- `clocks/metadata/audit_report.md` — access issues, reconciliation decisions,
  changed-value summary, and final publication record.
- `clocks/metadata/validate_metadata.py` — maintained schema, evidence, notebook,
  weight, aggregate, and fingerprint validation helpers.
- `tests/test_clock_metadata.py` — hermetic registry/vocabulary/evidence and
  cross-artifact tests.
- `clocks/metadata/_audit_tools.py` — temporary audit-manifest, shard merge,
  proposed-change, notebook patch, weight patch, and aggregate migration tool;
  remove before final commit.

### Modify

- `.gitignore` — allow the two committed JSON files under `clocks/metadata/`.
- `clocks/update_all_clocks.py` — read curated values from the JSON registry,
  merge runtime-only fields, and emit the aggregate deterministically.
- `tests/test_update_all_clocks.py` — specify registry-backed regeneration.
- `docs/source/make_clock_data.py` — accept an explicit local aggregate and
  preserve arrays in JSON/CSV output.
- `docs/source/test_make_clock_data.py` — specify local-input and array output.
- `docs/_static/clock_explorer_core.js` — facet, filter, search, sort, and CSV
  support for arrays.
- `docs/_static/tests/clock_explorer_core.test.js` — array behavior tests.
- `docs/_static/clock_explorer.js` — deterministic array display and the new
  `training_target` field.
- `docs/_static/clocks.json` — regenerated catalogue artifact.
- `docs/_static/clock_glossary.csv` — regenerated catalogue artifact.
- `clocks/notebooks/*.ipynb` except `template.ipynb` — controlled arrays and
  source-specific same-line comments in metadata cells.
- `docs/source/clock_notebooks/*.ipynb` except `template.ipynb` — refreshed
  documentation copies of the implementation notebooks.
- `clocks/huggingface/README.md` — document controlled array fields and evidence
  artifacts.

### Modify locally and publish to Hugging Face, but do not Git-add

- `clocks/weights/*.pt` — enriched, corrected metadata only; intentionally
  Git-ignored.
- `clocks/metadata/all_clock_metadata.pt` — regenerated runtime aggregate;
  intentionally Git-ignored.

### Temporary external paths

- `/tmp/pyaging-clock-metadata-audit/manifest.json` — deterministic DOI
  assignments.
- `/tmp/pyaging-clock-metadata-audit/batch-01.json` through
  `/tmp/pyaging-clock-metadata-audit/batch-12.json` — balanced paper batches.
- `/tmp/pyaging-clock-metadata-audit/shard-01.json` through
  `/tmp/pyaging-clock-metadata-audit/shard-12.json` — isolated subagent output.
- `/tmp/pyaging-clock-metadata-audit/baseline_fingerprints.json` — logical model
  fingerprints before mutation.
- `/tmp/pyaging-clock-metadata-audit/proposed_changes.json` — pre-mutation
  review report.

## Canonical schema

`controlled_vocabulary.json` has this top-level shape:

```json
{
  "schema_version": 1,
  "array_fields": ["tissue", "platform", "predicts", "training_target", "unit"],
  "fields": {
    "tissue": {
      "description": "Biological material used for feature selection or model fitting.",
      "values": ["whole blood", "multi-tissue"],
      "aliases": {"peripheral whole-blood leukocytes": "whole blood"}
    }
  }
}
```

The initial `values` arrays are seeds only; Task 4 replaces them with the
complete reconciled vocabulary derived from all 71 papers. Vocabulary values
are lower-case except official platform and species names.

Each `clock_metadata.json` record preserves the existing public fields and adds
`training_target`. These five fields are always nonempty arrays:

```json
{
  "horvath2013": {
    "clock_name": "horvath2013",
    "data_type": "methylation",
    "species": "Homo sapiens",
    "year": 2013,
    "approved_by_author": "⌛",
    "citation": "Horvath, Steve. \"DNA methylation age of human tissues and cell types.\" Genome Biology 14 (2013): R115.",
    "doi": "https://doi.org/10.1186/gb-2013-14-10-r115",
    "research_only": null,
    "notes": "Pan-tissue DNA methylation clock for chronological age.",
    "tissue": ["multi-tissue"],
    "predicts": ["chronological age"],
    "training_target": ["chronological age"],
    "unit": ["years"],
    "model_type": "elastic net",
    "platform": ["Illumina 27K", "Illumina 450K"],
    "population": "all ages",
    "journal": "Genome Biology",
    "last_author": "Steve Horvath",
    "n_features": 353,
    "citations": 7318,
    "citations_date": "2026-07-05"
  }
}
```

The example illustrates types and normalization, not a pre-audited final claim.
The audit determines the final values and source-specific notebook comment.

Each line in `evidence_ledger.jsonl` has one complete clock record:

```json
{
  "clock_name": "horvath2013",
  "doi": "https://doi.org/10.1186/gb-2013-14-10-r115",
  "reviewer": "paper-audit-01",
  "sources": [
    {
      "id": "paper",
      "type": "paper",
      "url": "https://doi.org/10.1186/gb-2013-14-10-r115",
      "accessed": "2026-07-18"
    }
  ],
  "fields": {
    "tissue": {
      "value": ["multi-tissue"],
      "source_text": "Short faithful transcription of the training material.",
      "source_id": "paper",
      "locator": "Methods: Training data",
      "status": "paper-confirmed",
      "note": ""
    }
  },
  "access_issues": []
}
```

The complete record contains evidence for `data_type`, `species`, `year`,
`citation`, `doi`, `notes`, `tissue`, `predicts`, `training_target`, `unit`,
`model_type`, `platform`, `population`, `journal`, `last_author`, and
`n_features`. Citation counts, citation date, approval status, and
`research_only` are administrative fields and are preserved from the current
registry rather than attributed to the original paper.

### Task 1: Establish failing schema and consistency tests

**Files:**

- Modify: `.gitignore`
- Create: `tests/test_clock_metadata.py`
- Create: `clocks/metadata/validate_metadata.py`
- Create: `clocks/metadata/controlled_vocabulary.json`
- Create: `clocks/metadata/clock_metadata.json`
- Create: `clocks/metadata/evidence_ledger.jsonl`

- [ ] **Step 1: Allow only the canonical JSON artifacts through the blanket
  JSON ignore**

Append:

```gitignore
# Audited clock metadata sources are intentionally versioned.
!clocks/metadata/controlled_vocabulary.json
!clocks/metadata/clock_metadata.json
```

- [ ] **Step 2: Seed the canonical registry mechanically from the current
  aggregate**

Create the two JSON files with sorted keys and UTF-8 output. Convert only the
five array fields mechanically:

```python
ARRAY_FIELDS = ("tissue", "platform", "predicts", "training_target", "unit")

def as_seed_array(field, entry):
    if field == "training_target":
        return [entry["predicts"]]
    value = entry[field]
    return value if isinstance(value, list) else [value]
```

Write one provisional JSONL record per clock with `status: "unresolved"` for
every audited field. This makes the red tests reflect incomplete research
rather than missing files.

Seed each vocabulary `values` array from the distinct mechanically converted
registry values so structural validation passes before scientific
harmonization. Seed `aliases` as an empty object. Task 4 replaces verbose seed
terms with the reconciled controlled vocabulary.

- [ ] **Step 3: Write validator tests**

Add tests with these exact public helper contracts:

```python
from pathlib import Path

import pytest
import torch

from clocks.metadata.validate_metadata import (
    ARRAY_FIELDS,
    AUDITED_FIELDS,
    load_json,
    load_ledger,
    validate_artifact_consistency,
    validate_evidence,
    validate_registry,
)

ROOT = Path(__file__).parents[1]
METADATA = ROOT / "clocks" / "metadata"


def test_registry_has_every_implementation_notebook():
    registry = load_json(METADATA / "clock_metadata.json")
    notebook_names = {
        path.stem
        for path in (ROOT / "clocks" / "notebooks").glob("*.ipynb")
        if path.stem != "template"
    }
    assert set(registry) == notebook_names
    assert len(registry) == 173


def test_registry_uses_controlled_arrays():
    registry = load_json(METADATA / "clock_metadata.json")
    vocabulary = load_json(METADATA / "controlled_vocabulary.json")
    validate_registry(registry, vocabulary)
    for name, record in registry.items():
        assert record["clock_name"] == name
        for field in ARRAY_FIELDS:
            assert isinstance(record[field], list)
            assert record[field]


def test_evidence_is_complete_and_resolved():
    registry = load_json(METADATA / "clock_metadata.json")
    ledger = load_ledger(METADATA / "evidence_ledger.jsonl")
    validate_evidence(registry, ledger)
    assert all(
        evidence["status"] != "unresolved"
        for record in ledger.values()
        for field in AUDITED_FIELDS
        for evidence in [record["fields"][field]]
    )


@pytest.mark.full_catalog
def test_local_runtime_artifacts_match_registry():
    validate_artifact_consistency(ROOT)
```

- [ ] **Step 4: Implement only JSON loading and structural validation**

In `validate_metadata.py`, define:

```python
ARRAY_FIELDS = ("tissue", "platform", "predicts", "training_target", "unit")
CONTROLLED_SCALAR_FIELDS = ("data_type", "species", "model_type", "population")
AUDITED_FIELDS = (
    "data_type", "species", "year", "citation", "doi", "notes", "tissue",
    "predicts", "training_target", "unit", "model_type", "platform",
    "population", "journal", "last_author", "n_features",
)
EVIDENCE_STATUSES = {
    "paper-confirmed", "supplement-confirmed", "code-confirmed", "unresolved",
}
ADMIN_FIELDS = {
    "approved_by_author", "research_only", "citations", "citations_date",
}
```

`load_json(path)` uses `json.loads(path.read_text(encoding="utf-8"))`.
`load_ledger(path)` parses nonblank JSONL lines, rejects duplicate clock names,
and returns a dictionary keyed by `clock_name`.

`validate_registry()` checks lowercase keys, matching `clock_name`, required
fields, exact list types for `ARRAY_FIELDS`, nonempty unique values, vocabulary
membership for `ARRAY_FIELDS` and `CONTROLLED_SCALAR_FIELDS`, integer
`year`/`n_features`, DOI URLs beginning with `https://doi.org/`, and
deterministic alphabetical keys.

`validate_evidence()` checks one ledger record per registry clock, every
`AUDITED_FIELDS` member, allowed status, nonempty `source_text`, valid
`source_id`, nonempty locator, and exact equality between each evidence
`value` and the registry value.

Leave `validate_artifact_consistency()` raising
`NotImplementedError("artifact consistency is implemented after migration")`.

- [ ] **Step 5: Run the focused tests and confirm research and consistency are
  red**

Run:

```bash
.venv/bin/pytest tests/test_clock_metadata.py -q
.venv/bin/pytest -o addopts="" -m full_catalog \
  tests/test_clock_metadata.py::test_local_runtime_artifacts_match_registry -q
```

Expected: registry structure passes; evidence resolution fails because the seed
ledger is unresolved. The explicit full-catalog invocation fails with
`NotImplementedError`.

- [ ] **Step 6: Commit the test harness and seed files**

```bash
git add .gitignore clocks/metadata/controlled_vocabulary.json \
  clocks/metadata/clock_metadata.json clocks/metadata/evidence_ledger.jsonl \
  clocks/metadata/validate_metadata.py tests/test_clock_metadata.py
git commit -m "test: define audited clock metadata schema"
```

### Task 2: Build deterministic paper assignments and audit templates

**Files:**

- Create temporarily: `clocks/metadata/_audit_tools.py`
- Create temporarily: `/tmp/pyaging-clock-metadata-audit/*.json`
- Modify: `clocks/metadata/audit_report.md`

- [ ] **Step 1: Implement `build_manifest` in the temporary tool**

The command:

```bash
.venv/bin/python clocks/metadata/_audit_tools.py build-manifest \
  --registry clocks/metadata/clock_metadata.json \
  --output-dir /tmp/pyaging-clock-metadata-audit \
  --batches 12
```

must group records by normalized DOI, sort families by descending clock count
then DOI, and greedily place each family into the batch with the lowest current
clock count. It writes a manifest and 12 batch files. Each batch entry contains
`doi`, `clock_names`, `current_metadata`, and the exact audited-field list.

Implement these command functions:

```python
def normalize_doi(value):
    value = value.strip()
    return "https://doi.org/" + value.split("doi.org/", 1)[-1]


def assign_families(registry, batch_count):
    families = {}
    for name, record in registry.items():
        families.setdefault(normalize_doi(record["doi"]), []).append(name)
    bins = [{"clock_count": 0, "families": []} for _ in range(batch_count)]
    for doi, names in sorted(families.items(), key=lambda item: (-len(item[1]), item[0])):
        target = min(enumerate(bins), key=lambda item: (item[1]["clock_count"], item[0]))[1]
        target["families"].append({"doi": doi, "clock_names": sorted(names)})
        target["clock_count"] += len(names)
    return bins
```

- [ ] **Step 2: Add `validate-shard`**

The command:

```bash
.venv/bin/python clocks/metadata/_audit_tools.py validate-shard \
  --batch /tmp/pyaging-clock-metadata-audit/batch-01.json \
  --shard /tmp/pyaging-clock-metadata-audit/shard-01.json
```

must reject missing/extra DOIs, missing/extra clocks, missing audited fields,
unknown statuses, evidence values with wrong types, unknown `source_id`, blank
source text, and blank locators.

- [ ] **Step 3: Generate and inspect the assignments**

Run the manifest command and verify:

```bash
.venv/bin/python -c 'import json; p="/tmp/pyaging-clock-metadata-audit/manifest.json"; m=json.load(open(p)); assert m["paper_count"] == 71; assert m["clock_count"] == 173; assert len(m["batches"]) == 12; print([(b["batch"], b["paper_count"], b["clock_count"]) for b in m["batches"]])'
```

Expected: 12 nonempty batches, exactly 71 papers, exactly 173 clocks, and no DOI
split across batches.

- [ ] **Step 4: Start the audit report**

Create `audit_report.md` with fixed sections:

```markdown
# Clock Metadata Source Audit

## Scope
173 clocks across 71 DOI families.

## Controlled-vocabulary decisions

## Access issues

## Source contradictions and adjudications

## Changed-value summary

## Validation

## Hugging Face publication
```

- [ ] **Step 5: Commit the assignment tooling and report scaffold**

Do not commit `/tmp` files.

```bash
git add clocks/metadata/_audit_tools.py clocks/metadata/audit_report.md
git commit -m "chore: prepare clock metadata paper audit"
```

### Task 3: Execute all 12 source-review batches with parallel subagents

**Files:**

- Read: `/tmp/pyaging-clock-metadata-audit/batch-01.json` through
  `/tmp/pyaging-clock-metadata-audit/batch-12.json`
- Create: `/tmp/pyaging-clock-metadata-audit/shard-01.json` through
  `/tmp/pyaging-clock-metadata-audit/shard-12.json`
- Modify after each wave: `clocks/metadata/audit_report.md`

- [ ] **Step 1: Dispatch wave 1**

Launch three independent subagents for batches 01–03 with `fork_turns="none"`.
Use this prompt, substituting only the two-digit batch number:

```text
Audit every clock in /tmp/pyaging-clock-metadata-audit/batch-01.json against
its original paper. Read the batch file, the schema in
docs/superpowers/specs/2026-07-18-clock-metadata-harmonization-design.md, and
clocks/metadata/controlled_vocabulary.json.

For every DOI, open and inspect the original paper's methods/results and the
relevant supplement. Use official author code/model files only when the paper
and supplement do not establish a field. Do not infer values from clock names
or current metadata. Open-access sources may use web/research tools. If a
source needs the user's authenticated Comet session, record an access issue
instead of guessing.

Produce /tmp/pyaging-clock-metadata-audit/shard-01.json with one complete
record per assigned clock. Each audited field must include value, short
faithful source_text, source_id, locator, status, and note. Record every URL
and access date. Use only paper-confirmed, supplement-confirmed,
code-confirmed, or unresolved. Keep proposed controlled terms concise; record
exact source detail in source_text. Do not edit repository files.

Before finishing, run:
.venv/bin/python clocks/metadata/_audit_tools.py validate-shard
  --batch /tmp/pyaging-clock-metadata-audit/batch-01.json
  --shard /tmp/pyaging-clock-metadata-audit/shard-01.json

Return a summary of papers reviewed, clocks covered, access issues,
contradictions, and proposed new vocabulary terms.
```

- [ ] **Step 2: Validate and reconcile wave 1**

Run `validate-shard` for shards 01–03. Read all summaries. Add access issues,
contradictions, and vocabulary proposals to `audit_report.md`. Do not normalize
away disagreements yet.

- [ ] **Step 3: Dispatch and reconcile waves 2–4**

Repeat the exact procedure with three agents per wave:

- wave 2: batches 04–06;
- wave 3: batches 07–09;
- wave 4: batches 10–12.

Validate every shard before launching the next wave. If an agent returns an
invalid shard, send a focused follow-up specifying the validator error and do
not reassign that batch to another agent.

- [ ] **Step 4: Prove complete first-pass coverage**

Run:

```bash
for n in 01 02 03 04 05 06 07 08 09 10 11 12; do
  .venv/bin/python clocks/metadata/_audit_tools.py validate-shard \
    --batch "/tmp/pyaging-clock-metadata-audit/batch-${n}.json" \
    --shard "/tmp/pyaging-clock-metadata-audit/shard-${n}.json"
done
```

Expected: all 12 validators exit 0; union coverage is 71 DOI families and 173
clock names.

### Task 4: Reconcile the audit and finalize controlled values

**Files:**

- Modify: `clocks/metadata/controlled_vocabulary.json`
- Modify: `clocks/metadata/clock_metadata.json`
- Modify: `clocks/metadata/evidence_ledger.jsonl`
- Modify: `clocks/metadata/audit_report.md`
- Modify temporarily: `clocks/metadata/_audit_tools.py`

- [ ] **Step 1: Merge validated shards without normalization**

Implement and run:

```bash
.venv/bin/python clocks/metadata/_audit_tools.py merge-shards \
  --manifest /tmp/pyaging-clock-metadata-audit/manifest.json \
  --shards /tmp/pyaging-clock-metadata-audit \
  --output /tmp/pyaging-clock-metadata-audit/merged.json
```

The merge rejects duplicate/missing clocks and writes clocks alphabetically.

- [ ] **Step 2: Generate the vocabulary reconciliation report**

Implement and run:

```bash
.venv/bin/python clocks/metadata/_audit_tools.py vocabulary-report \
  --merged /tmp/pyaging-clock-metadata-audit/merged.json \
  --vocabulary clocks/metadata/controlled_vocabulary.json \
  --output /tmp/pyaging-clock-metadata-audit/vocabulary_report.json
```

The report groups case-folded values, punctuation variants, singular/plural
variants, current aliases, and proposed new terms by field. Review each group
manually. Do not merge biologically distinct materials such as `whole blood`,
`PBMC`, `cord blood`, and `dried blood spot`; do not merge raw assay platforms
with derived compatibility.

- [ ] **Step 3: Apply vocabulary decisions to the merged audit**

Update `controlled_vocabulary.json` with the final allowed values and aliases.
Use:

```bash
.venv/bin/python clocks/metadata/_audit_tools.py normalize-merged \
  --merged /tmp/pyaging-clock-metadata-audit/merged.json \
  --vocabulary clocks/metadata/controlled_vocabulary.json \
  --output /tmp/pyaging-clock-metadata-audit/normalized.json
```

The command may apply only explicit aliases present in the vocabulary file. It
must fail on unknown values.

- [ ] **Step 4: Perform the independent reconciliation review**

Dispatch one fresh review subagent with no paper-batch assignment. Its prompt:

```text
Review /tmp/pyaging-clock-metadata-audit/normalized.json and
clocks/metadata/controlled_vocabulary.json for cross-paper consistency.
Inspect the original sources for every unresolved item, all families with five
or more clocks, and a deterministic sample of ten single-clock papers selected
by alphabetical DOI order at positions 1, 5, 10, 15, 20, 25, 30, 35, 40, and
42. Check training tissue versus validation tissue, training platform versus
compatibility, returned prediction versus fitting target, and postprocessed
unit versus raw model scale. Do not edit repository files. Write findings to
/tmp/pyaging-clock-metadata-audit/reconciliation_review.json.
```

Reconcile every finding against the cited source.

- [ ] **Step 5: Stop for authenticated access when needed**

If any record remains `unresolved`, present the exact DOI, clock names, fields,
failed URLs, and required source to the user. Use Comet only for the affected
sources. Resume only after the evidence is obtained or the user explicitly
accepts a documented interpretation.

- [ ] **Step 6: Materialize the registry and ledger**

Implement and run:

```bash
.venv/bin/python clocks/metadata/_audit_tools.py materialize \
  --normalized /tmp/pyaging-clock-metadata-audit/normalized.json \
  --current clocks/metadata/clock_metadata.json \
  --registry clocks/metadata/clock_metadata.json \
  --ledger clocks/metadata/evidence_ledger.jsonl \
  --report clocks/metadata/audit_report.md
```

Administrative fields remain from the current registry. Audited fields come
only from normalized evidence. JSON keys and JSONL records are alphabetical.

- [ ] **Step 7: Run schema/evidence tests**

```bash
.venv/bin/pytest tests/test_clock_metadata.py::test_registry_has_every_implementation_notebook \
  tests/test_clock_metadata.py::test_registry_uses_controlled_arrays \
  tests/test_clock_metadata.py::test_evidence_is_complete_and_resolved -q
```

Expected: all three pass; only the not-yet-implemented artifact consistency test
remains red.

- [ ] **Step 8: Commit the audited sources**

```bash
git add clocks/metadata/controlled_vocabulary.json \
  clocks/metadata/clock_metadata.json clocks/metadata/evidence_ledger.jsonl \
  clocks/metadata/audit_report.md
git commit -m "data: audit and harmonize all clock metadata"
```

### Task 5: Make aggregate regeneration registry-backed

**Files:**

- Modify: `clocks/update_all_clocks.py`
- Modify: `tests/test_update_all_clocks.py`

- [ ] **Step 1: Replace binary-curation tests with registry tests**

Update fixtures to write JSON and assert:

```python
def test_regeneration_reads_curated_fields_from_registry(tmp_path):
    registry_path = tmp_path / "clock_metadata.json"
    registry_path.write_text(
        json.dumps({
            "clock": {
                "clock_name": "clock",
                "notes": "Audited notes",
                "tissue": ["whole blood"],
                "platform": ["Illumina 450K"],
                "predicts": ["chronological age"],
                "training_target": ["chronological age"],
                "unit": ["years"],
            }
        }),
        encoding="utf-8",
    )
    result = update_all_clocks.load_curated_metadata(registry_path)
    assert result["clock"]["tissue"] == ["whole blood"]
```

Retain the preflight/no-partial-save tests, but pass both `registry_path` and
`metadata_path`.

- [ ] **Step 2: Run focused tests and confirm failure**

```bash
.venv/bin/pytest tests/test_update_all_clocks.py -q
```

Expected: failures because `load_curated_metadata` still uses `torch.load` and
`regenerate_clock_metadata` lacks `registry_path`.

- [ ] **Step 3: Implement registry-backed regeneration**

Change the signature to:

```python
def regenerate_clock_metadata(
    version,
    weights_dir=Path("weights"),
    registry_path=Path("metadata/clock_metadata.json"),
    metadata_path=Path("metadata/all_clock_metadata.pt"),
):
```

Load the registry with `json.loads`, validate its top-level dictionary and clock
names, and merge only `RUNTIME_METADATA_FIELDS` from generated weights. Save the
aggregate only after all weight preflight and merge checks succeed.

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_update_all_clocks.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add clocks/update_all_clocks.py tests/test_update_all_clocks.py
git commit -m "refactor: generate aggregate from audited registry"
```

### Task 6: Add local aggregate input and array-safe docs generation

**Files:**

- Modify: `docs/source/make_clock_data.py`
- Modify: `docs/source/test_make_clock_data.py`

- [ ] **Step 1: Write failing generator tests**

Add:

```python
def test_generate_accepts_local_metadata_and_preserves_arrays(tmp_path, monkeypatch):
    source = tmp_path / "all_clock_metadata.pt"
    torch.save({
        "clock": {
            "approved_by_author": "⌛",
            "tissue": ["whole blood", "cord blood"],
            "platform": ["Illumina 450K", "Illumina EPIC"],
            "predicts": ["chronological age"],
            "training_target": ["chronological age"],
            "unit": ["years"],
        }
    }, source)
    static = tmp_path / "static"
    monkeypatch.setattr(make_clock_data, "STATIC", str(static))

    assert make_clock_data.generate(metadata_path=source) == 1
    row = json.loads((static / "clocks.json").read_text())[0]
    assert row["tissue"] == ["whole blood", "cord blood"]
    csv_text = (static / "clock_glossary.csv").read_text()
    assert "whole blood | cord blood" in csv_text
```

Add `training_target` to required fields and expected CSV headers.

- [ ] **Step 2: Run and confirm failure**

```bash
.venv/bin/pytest docs/source/test_make_clock_data.py -q
```

Expected: failure because `generate()` has no `metadata_path` parameter and
`training_target` is absent from `FIELDS`.

- [ ] **Step 3: Implement local input and stable CSV arrays**

Use:

```python
def generate(metadata_path=None):
    os.makedirs(STATIC, exist_ok=True)
    if metadata_path is None:
        metadata_path = download_hf_file("all_clock_metadata.pt", STATIC)
    meta = torch.load(metadata_path, weights_only=False)
```

Add `training_target` to `FIELDS` after `predicts`. Preserve lists in JSON. For
CSV only, map lists to `" | ".join(map(str, value))`; do not mutate JSON rows.
The command-line entry point accepts `--metadata-path` with `argparse`.

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest docs/source/test_make_clock_data.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add docs/source/make_clock_data.py docs/source/test_make_clock_data.py
git commit -m "docs: generate catalogue from array metadata"
```

### Task 7: Make Clock Explorer facets and rendering array-aware

**Files:**

- Modify: `docs/_static/clock_explorer_core.js`
- Modify: `docs/_static/tests/clock_explorer_core.test.js`
- Modify: `docs/_static/clock_explorer.js`

- [ ] **Step 1: Convert test fixtures to arrays and specify semantics**

Use arrays for the five array fields. Add assertions:

```javascript
const multi = [
  {
    clock_name: "multi",
    tissue: ["whole blood", "cord blood"],
    platform: ["Illumina 450K", "Illumina EPIC"],
    predicts: ["chronological age"],
    training_target: ["chronological age"],
    unit: ["years"],
  },
  {
    clock_name: "cord",
    tissue: ["cord blood"],
    platform: ["Illumina EPIC"],
    predicts: ["gestational age"],
    training_target: ["gestational age"],
    unit: ["weeks"],
  },
];
const mf = core.computeFacets(multi);
assert.strictEqual(mf.tissue.find((x) => x.value === "cord blood").count, 2);
assert.strictEqual(mf.tissue.find((x) => x.value === "whole blood").count, 1);
assert.deepStrictEqual(
  core.filterClocks(multi, { tissue: ["whole blood"], platform: ["Illumina EPIC"] }, "")
    .map((c) => c.clock_name),
  ["multi"]
);
assert.deepStrictEqual(
  core.filterClocks(multi, { tissue: ["whole blood", "cord blood"] }, "")
    .map((c) => c.clock_name),
  ["multi", "cord"]
);
assert.strictEqual(core.formatValue(["whole blood", "cord blood"]), "whole blood; cord blood");
assert.ok(core.toCSV([multi[0]], ["tissue"]).includes('"whole blood | cord blood"'));
```

Add `training_target` to `FACET_FIELDS`.

- [ ] **Step 2: Run and confirm failure**

```bash
node docs/_static/tests/clock_explorer_core.test.js
```

Expected: array facets collapse into comma-joined strings, filters fail, and
`formatValue` is undefined.

- [ ] **Step 3: Implement normalized values**

Add:

```javascript
function valuesOf(value) {
  if (value === null || value === undefined || value === "") return [];
  return Array.isArray(value) ? value : [value];
}

function formatValue(value, delimiter) {
  return valuesOf(value).map(String).join(delimiter || "; ");
}
```

`computeFacets` iterates over a case-insensitively deduplicated `valuesOf(c[f])`
set for each clock. `filterClocks` matches when any selected value equals any
clock value within the same field. Search concatenates `valuesOf(val)`. String
sorting compares `formatValue(value).toLowerCase()`. `toCSV` uses `" | "` for
arrays before escaping. Export `valuesOf` and `formatValue`.

- [ ] **Step 4: Update DOM rendering**

In `clock_explorer.js`, add `training_target` to labels and detail fields.
Change:

```javascript
function fmt(v) {
  return core.formatValue(v) || "—";
}
```

When building card badges, flatten array fields with `core.valuesOf`. Keep
approval rendering scalar.

- [ ] **Step 5: Run checks**

```bash
node docs/_static/tests/clock_explorer_core.test.js
node --check docs/_static/clock_explorer.js
```

Expected: pass and syntax exit 0.

- [ ] **Step 6: Commit**

```bash
git add docs/_static/clock_explorer_core.js \
  docs/_static/tests/clock_explorer_core.test.js docs/_static/clock_explorer.js
git commit -m "docs: filter multi-valued clock metadata"
```

### Task 8: Dry-run the one-off notebook and model migration

**Files:**

- Modify temporarily: `clocks/metadata/_audit_tools.py`
- Create temporarily: `/tmp/pyaging-clock-metadata-audit/baseline_fingerprints.json`
- Create temporarily: `/tmp/pyaging-clock-metadata-audit/proposed_changes.json`
- Modify: `clocks/metadata/audit_report.md`

- [ ] **Step 1: Implement logical fingerprints**

Fingerprint prediction-relevant state with:

```python
def tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
    }


def model_fingerprint(model):
    return {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "state_dict": {
            key: tensor_digest(value)
            for key, value in sorted(model.state_dict().items())
        },
        "features": normalize_runtime_value(model.features),
        "base_model_features": normalize_runtime_value(model.base_model_features),
        "reference_values": normalize_runtime_value(model.reference_values),
        "preprocess_name": model.preprocess_name,
        "preprocess_dependencies": normalize_runtime_value(model.preprocess_dependencies),
        "postprocess_name": model.postprocess_name,
        "postprocess_dependencies": normalize_runtime_value(model.postprocess_dependencies),
        "version": model.version,
    }
```

`normalize_runtime_value` handles `None`, strings, numbers, lists, tuples,
dictionaries with sorted keys, NumPy arrays/scalars, pandas Index/Series, and
PyTorch tensors. Unknown types raise rather than stringify.

- [ ] **Step 2: Capture all baseline fingerprints**

```bash
.venv/bin/python clocks/metadata/_audit_tools.py fingerprint \
  --weights clocks/weights \
  --output /tmp/pyaging-clock-metadata-audit/baseline_fingerprints.json
```

Expected: 173 fingerprints.

- [ ] **Step 3: Implement notebook-line rendering**

For each audited record, replace assignment lines in the single metadata cell.
Render strings with `json.dumps(value, ensure_ascii=False)` so quoting is
deterministic. Render the five array fields as Python list literals. For every
controlled field in `ARRAY_FIELDS + CONTROLLED_SCALAR_FIELDS`, append the
evidence `source_text` as `# Paper: ...`, replacing newlines with spaces and
escaping any comment-breaking control characters.

Fail if a notebook has zero or multiple metadata cells, if any curated field is
missing, if `template.ipynb` is selected, or if a proposed clock lacks a
notebook and weight.

- [ ] **Step 4: Implement dry-run change reporting**

Run:

```bash
.venv/bin/python clocks/metadata/_audit_tools.py migrate \
  --registry clocks/metadata/clock_metadata.json \
  --ledger clocks/metadata/evidence_ledger.jsonl \
  --notebooks clocks/notebooks \
  --weights clocks/weights \
  --aggregate clocks/metadata/all_clock_metadata.pt \
  --baseline /tmp/pyaging-clock-metadata-audit/baseline_fingerprints.json \
  --report /tmp/pyaging-clock-metadata-audit/proposed_changes.json \
  --dry-run
```

The report lists old/new values per field, notebook lines to change, model
metadata keys to add/change, aggregate changes, unchanged fields, and
fingerprint verification. Dry-run writes no repository artifact.

- [ ] **Step 5: Review the proposed change report**

Check all 173 clocks appear exactly once, no runtime configuration changes, no
comments exceed a readable single line, no controlled value is absent from the
vocabulary, and every proposed metadata change has resolved evidence. Summarize
counts and notable corrections in `audit_report.md`.

- [ ] **Step 6: Commit the reviewed audit report**

```bash
git add clocks/metadata/audit_report.md
git commit -m "docs: summarize proposed clock metadata corrections"
```

### Task 9: Apply the one-off migration and prove model behavior is unchanged

**Files:**

- Modify: `clocks/notebooks/*.ipynb`
- Modify locally/Hugging Face only: `clocks/weights/*.pt`
- Modify locally/Hugging Face only: `clocks/metadata/all_clock_metadata.pt`
- Modify: `clocks/metadata/validate_metadata.py`
- Modify: `tests/test_clock_metadata.py`
- Delete: `clocks/metadata/_audit_tools.py`

- [ ] **Step 1: Apply the reviewed migration**

Run the Task 8 migrate command without `--dry-run`. It writes notebooks and
weights only after validating every input, then calls registry-backed aggregate
generation.

- [ ] **Step 2: Verify fingerprints immediately**

```bash
.venv/bin/python clocks/metadata/_audit_tools.py verify-fingerprints \
  --weights clocks/weights \
  --baseline /tmp/pyaging-clock-metadata-audit/baseline_fingerprints.json
```

Expected: 173/173 logical fingerprints match.

- [ ] **Step 3: Implement permanent artifact consistency**

Complete `validate_artifact_consistency(root)`:

- parse each notebook's metadata cell with `ast` and compare curated values to
  the registry;
- require same-line `# Paper:` comments for all fields in
  `ARRAY_FIELDS + CONTROLLED_SCALAR_FIELDS`;
- load each `.pt`, compare its curated `model.metadata` to the registry, and
  compare feature count to `n_features`;
- load `all_clock_metadata.pt`, compare curated values to the registry, and
  verify runtime-only fields separately;
- require identical clock-name sets across all four artifacts.

Add a test that every model metadata dictionary contains the controlled fields
and that the aggregate has `training_target`.

- [ ] **Step 4: Run metadata, updater, and prediction tests**

```bash
.venv/bin/pytest tests/test_clock_metadata.py tests/test_update_all_clocks.py -q
.venv/bin/pytest -o addopts="" -m full_catalog \
  tests/test_clock_metadata.py::test_local_runtime_artifacts_match_registry -q
.venv/bin/pytest tests/predict/test_gold_standard.py -q
```

Expected: schema tests, the local 173-object consistency test, and gold-standard
numeric predictions all pass.

- [ ] **Step 5: Remove the one-off tool**

Delete `clocks/metadata/_audit_tools.py` with `apply_patch`. Confirm no committed
code imports it:

```bash
rg -n '_audit_tools' . -g '!docs/superpowers/plans/2026-07-18-clock-metadata-harmonization.md'
```

Expected: no matches.

- [ ] **Step 6: Commit the migration**

```bash
git add clocks/notebooks clocks/metadata/validate_metadata.py \
  tests/test_clock_metadata.py
git add -u clocks/metadata/_audit_tools.py
git commit -m "data: apply audited metadata to all clocks"
```

Confirm the ignored binaries are not accidentally staged:

```bash
if git diff --cached --name-only | rg -q '^(clocks/weights/|clocks/metadata/all_clock_metadata\.pt$)'; then
  echo "Ignored runtime binaries must not be committed"
  exit 1
fi
```

### Task 10: Regenerate and validate the documentation catalogue

**Files:**

- Modify: `docs/_static/clocks.json`
- Modify: `docs/_static/clock_glossary.csv`
- Modify: `docs/source/clock_notebooks/*.ipynb`
- Modify: `clocks/huggingface/README.md`

- [ ] **Step 1: Refresh documented notebooks and regenerate locally from the
  validated aggregate**

```bash
cp clocks/notebooks/*.ipynb docs/source/clock_notebooks/
.venv/bin/python docs/source/make_clock_data.py \
  --metadata-path clocks/metadata/all_clock_metadata.pt
```

Expected: `generated 173 clocks`.

- [ ] **Step 2: Document the schema**

In `clocks/huggingface/README.md`, state that `tissue`, `platform`, `predicts`,
`training_target`, and `unit` are arrays of controlled terms; define their
training/output semantics; point to the pyaging repository's canonical registry
and evidence ledger; retain the security and mixed-provenance warnings.

- [ ] **Step 3: Run generated-artifact and JavaScript tests**

```bash
.venv/bin/pytest docs/source/test_make_clock_data.py tests/test_clock_metadata.py -q
node docs/_static/tests/clock_explorer_core.test.js
```

Expected: all pass.

- [ ] **Step 4: Build Sphinx with warnings as errors**

```bash
.venv/bin/sphinx-build -W --keep-going -b html docs/source docs/_build/html
```

Expected: exit 0 with no warnings introduced by the metadata migration.

- [ ] **Step 5: Serve and visually inspect**

Serve:

```bash
.venv/bin/python -m http.server 8765 --directory docs/_build/html
```

Inspect `clock_glossary.html` at approximately 1440×900 and 390×844. Verify
multi-value cells, individual facet choices, OR-within/AND-across filters,
search, cards, expanded details, sorting, and CSV download. Stop the server
after inspection.

- [ ] **Step 6: Commit**

```bash
git add docs/_static/clocks.json docs/_static/clock_glossary.csv \
  docs/source/clock_notebooks clocks/huggingface/README.md
git commit -m "docs: publish harmonized clock catalogue"
```

### Task 11: Run full local verification and independent review

**Files:**

- Verify all changed files
- Modify only if a review finding is confirmed

- [ ] **Step 1: Run formatting and focused static checks**

```bash
git diff --check HEAD~6
node --check docs/_static/clock_explorer_core.js
node --check docs/_static/clock_explorer.js
.venv/bin/ruff check clocks/update_all_clocks.py \
  clocks/metadata/validate_metadata.py docs/source/make_clock_data.py \
  tests/test_clock_metadata.py tests/test_update_all_clocks.py \
  docs/source/test_make_clock_data.py
```

Expected: all exit 0.

- [ ] **Step 2: Run the offline suite**

```bash
.venv/bin/pytest -m "not full_catalog and not online" tests/ docs/source/test_make_clock_data.py -q
.venv/bin/pytest -o addopts="" -m full_catalog \
  tests/test_clock_metadata.py::test_local_runtime_artifacts_match_registry -q
node docs/_static/tests/clock_explorer_core.test.js
.venv/bin/sphinx-build -W --keep-going -b html docs/source docs/_build/html
```

Expected: all tests pass and Sphinx exits 0.

- [ ] **Step 3: Request independent code/data review**

Use `superpowers:requesting-code-review`. Ask one reviewer to check schema and
code behavior and a separate reviewer to check a deterministic 10% sample of
evidence-to-registry-to-notebook chains. Resolve only findings confirmed
against source evidence and rerun the affected tests.

- [ ] **Step 4: Update final local validation report**

Record exact test counts, Sphinx result, fingerprint count, changed-field
counts, access resolutions, and remaining nonblocking caveats under
`audit_report.md` → `Validation`.

- [ ] **Step 5: Commit review fixes/report**

```bash
git add clocks/metadata/audit_report.md
git add -u
git commit -m "test: verify harmonized clock metadata"
```

If there are no changes, do not create an empty commit.

### Task 12: Publish model objects and aggregate metadata to Hugging Face

**Files:**

- Modify: `clocks/metadata/audit_report.md`
- Regenerate: `docs/_static/clocks.json`
- Regenerate: `docs/_static/clock_glossary.csv`

- [ ] **Step 1: Verify identity and repository visibility**

```bash
make verify-hf-data-repo-public
```

Expected: authenticated user `lucascamillomd` and public repository
`lucascamillomd/pyaging-data`.

- [ ] **Step 2: Upload model objects first**

```bash
uv run hf upload lucascamillomd/pyaging-data clocks/weights . \
  --type model --commit-message "Harmonize audited clock metadata"
```

Expected: upload succeeds; aggregate is not included in this commit.

- [ ] **Step 3: Verify remote model objects**

Download every remote root-level clock `.pt` into a temporary Hugging Face
snapshot/cache and run the permanent registry/model consistency and logical
fingerprint checks against the remote files. Require 173/173 clock names,
metadata matches, and fingerprints match before continuing.

- [ ] **Step 4: Upload aggregate metadata last**

```bash
uv run hf upload lucascamillomd/pyaging-data \
  clocks/metadata/all_clock_metadata.pt all_clock_metadata.pt \
  --type model --commit-message "Publish audited aggregate clock metadata"
```

- [ ] **Step 5: Verify remote aggregate and record revision**

Use `hf_hub_download(..., force_download=True)` for
`all_clock_metadata.pt`, compare it to the local aggregate with
`validate_artifact_consistency`, and obtain:

```bash
uv run hf models info lucascamillomd/pyaging-data --format json
```

Record the returned SHA in `audit_report.md`.

- [ ] **Step 6: Regenerate docs from the published default source**

Run without `--metadata-path`:

```bash
.venv/bin/python docs/source/make_clock_data.py
.venv/bin/pytest docs/source/test_make_clock_data.py tests/test_clock_metadata.py -q
git diff --exit-code -- docs/_static/clocks.json docs/_static/clock_glossary.csv
```

Expected: remote regeneration is byte-identical to the committed local
artifacts.

- [ ] **Step 7: Commit publication record**

```bash
git add clocks/metadata/audit_report.md
git commit -m "docs: record audited metadata publication"
```

### Task 13: Final verification and handoff

**Files:**

- Verify all repository and remote artifacts

- [ ] **Step 1: Apply `superpowers:verification-before-completion`**

Run fresh:

```bash
.venv/bin/pytest -m "not full_catalog and not online" tests/ docs/source/test_make_clock_data.py -q
.venv/bin/pytest -o addopts="" -m full_catalog \
  tests/test_clock_metadata.py::test_local_runtime_artifacts_match_registry -q
node docs/_static/tests/clock_explorer_core.test.js
.venv/bin/sphinx-build -W --keep-going -b html docs/source docs/_build/html
git diff --check
git status --short
```

Expected: tests and docs pass; diff check is clean; status contains only the
intended branch state.

- [ ] **Step 2: Deliver the audit summary**

Report:

- 173/173 clocks and 71/71 DOI families reviewed;
- evidence-status counts by paper/supplement/code;
- every inaccessible source and how it was resolved;
- old/new vocabulary cardinalities;
- changed-field counts and scientifically important corrections;
- 173/173 model fingerprints unchanged;
- exact automated test and docs-build results;
- Hugging Face repository revision;
- links to the registry, vocabulary, evidence ledger, audit report, and rebuilt
  catalogue artifacts.

- [ ] **Step 3: Apply `superpowers:finishing-a-development-branch`**

Offer the user the skill's merge/PR/keep/discard choices. Do not merge, push the
Git branch, or open a pull request without the user's selected handoff.
