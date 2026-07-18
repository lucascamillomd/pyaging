import copy
import json
from pathlib import Path

import pytest

from clocks.metadata.validate_metadata import (
    ARRAY_FIELDS,
    AUDITED_FIELDS,
    load_json,
    load_ledger,
    validate_artifact_consistency,
    validate_evidence,
    validate_registry,
    validate_vocabulary,
)

ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = ROOT / "clocks" / "metadata"
REGISTRY_PATH = METADATA_DIR / "clock_metadata.json"
VOCABULARY_PATH = METADATA_DIR / "controlled_vocabulary.json"
LEDGER_PATH = METADATA_DIR / "evidence_ledger.jsonl"


@pytest.fixture(scope="module")
def registry():
    return load_json(REGISTRY_PATH)


@pytest.fixture(scope="module")
def vocabulary():
    return load_json(VOCABULARY_PATH)


@pytest.fixture(scope="module")
def ledger():
    return load_ledger(LEDGER_PATH)


def test_registry_has_every_implementation_notebook(registry):
    notebook_names = {
        path.stem for path in (ROOT / "clocks" / "notebooks").glob("*.ipynb") if path.name != "template.ipynb"
    }
    assert len(registry) == 173
    assert set(registry) == notebook_names


def test_registry_uses_controlled_arrays(registry, vocabulary):
    validate_registry(registry, vocabulary)
    for clock_name, record in registry.items():
        for field in ARRAY_FIELDS:
            assert isinstance(record[field], list), f"{clock_name}.{field}"
            assert record[field], f"{clock_name}.{field}"


def test_evidence_is_complete_and_resolved(registry, ledger):
    validate_evidence(registry, ledger)
    for clock_name, record in ledger.items():
        for field in AUDITED_FIELDS:
            assert record["fields"][field]["status"] != "unresolved", f"{clock_name}.{field}"


@pytest.mark.full_catalog
def test_local_runtime_artifacts_match_registry():
    validate_artifact_consistency(ROOT)


def test_load_ledger_rejects_duplicate_clock_names(tmp_path):
    path = tmp_path / "ledger.jsonl"
    record = {"clock_name": "clock"}
    path.write_text("\n".join((json.dumps(record), json.dumps(record))), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate.*clock"):
        load_ledger(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_load_json_rejects_non_finite_numbers(tmp_path, constant):
    path = tmp_path / "metadata.json"
    path.write_text(f'{{"value": {constant}}}', encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite"):
        load_json(path)


def test_load_json_rejects_nested_duplicate_keys_with_file_context(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text('{"outer": {"value": 1, "value": 2}}', encoding="utf-8")

    with pytest.raises(ValueError, match=r"metadata\.json.*duplicate key.*value"):
        load_json(path)


@pytest.mark.parametrize(
    ("line", "context"),
    [
        ("{", r"line 1"),
        ("null", r"line 1.*object"),
        ("[]", r"line 1.*object"),
        ('{"other": "value"}', r"line 1.*clock_name"),
        ('{"clock_name": 3}', r"line 1.*clock_name"),
        ('{"clock_name": ""}', r"line 1.*clock_name"),
        ('{"clock_name": "clock", "value": NaN}', r"line 1.*non-finite"),
    ],
)
def test_load_ledger_rejects_malformed_records(tmp_path, line, context):
    path = tmp_path / "ledger.jsonl"
    path.write_text(line, encoding="utf-8")

    with pytest.raises(ValueError, match=context):
        load_ledger(path)


def test_load_ledger_requires_alphabetical_order(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text('{"clock_name": "z"}\n{"clock_name": "a"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"line 2.*alphabetical"):
        load_ledger(path)


def test_load_ledger_rejects_nested_duplicate_keys_with_line_context(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text(
        '{"clock_name": "a"}\n{"clock_name": "b", "nested": {"value": 1, "value": 2}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"line 2.*duplicate key.*value"):
        load_ledger(path)


@pytest.mark.parametrize(
    ("mutation", "context"),
    [
        (lambda record: record.__setitem__("tissue", "blood"), r"example\.tissue"),
        (lambda record: record.__setitem__("year", True), r"example\.year"),
        (lambda record: record.__setitem__("data_type", "unknown"), r"example\.data_type"),
        (lambda record: record.pop("approved_by_author"), r"example\.approved_by_author"),
        (lambda record: record.__setitem__("citations", True), r"example\.citations"),
        (lambda record: record.__setitem__("citation", {"raw": "citation"}), r"example\.citation"),
    ],
)
def test_validate_registry_rejects_representative_invalid_values(registry, vocabulary, mutation, context):
    invalid_registry = {"example": copy.deepcopy(next(iter(registry.values())))}
    invalid_registry["example"]["clock_name"] = "example"
    mutation(invalid_registry["example"])

    with pytest.raises(ValueError, match=context):
        validate_registry(invalid_registry, vocabulary)


@pytest.mark.parametrize(
    ("mutation", "context"),
    [
        (lambda value: value.__setitem__("schema_version", 2), r"vocabulary\.schema_version"),
        (lambda value: value.__setitem__("array_fields", ["tissue"]), r"vocabulary\.array_fields"),
        (lambda value: value["fields"].pop("species"), r"vocabulary\.fields"),
        (
            lambda value: value["fields"]["species"].__setitem__("description", ""),
            r"vocabulary\.species\.description",
        ),
        (
            lambda value: value["fields"]["species"].__setitem__("values", []),
            r"vocabulary\.species\.values.*nonempty",
        ),
        (
            lambda value: value["fields"]["species"].__setitem__("values", ["a", "a"]),
            r"vocabulary\.species\.values.*unique",
        ),
        (
            lambda value: value["fields"]["species"].__setitem__("values", [True]),
            r"vocabulary\.species\.values",
        ),
        (
            lambda value: value["fields"]["species"].__setitem__("aliases", {"human": "missing"}),
            r"vocabulary\.species\.aliases",
        ),
    ],
)
def test_validate_vocabulary_rejects_malformed_schema(vocabulary, mutation, context):
    invalid_vocabulary = copy.deepcopy(vocabulary)
    mutation(invalid_vocabulary)

    with pytest.raises(ValueError, match=context):
        validate_vocabulary(invalid_vocabulary)


def test_validate_vocabulary_preserves_unique_declared_order(vocabulary):
    declared_order_vocabulary = copy.deepcopy(vocabulary)
    declared_order_vocabulary["fields"]["species"]["values"].reverse()

    validate_vocabulary(declared_order_vocabulary)


@pytest.mark.parametrize(
    ("mutation", "context"),
    [
        (
            lambda record: record["fields"]["year"].__setitem__("status", "guessed"),
            r"example\.year.*status",
        ),
        (
            lambda record: record["fields"]["year"].__setitem__("status", []),
            r"example\.year.*status",
        ),
        (
            lambda record: record["fields"]["year"].__setitem__("source_id", "missing"),
            r"example\.year.*source_id",
        ),
        (
            lambda record: record["fields"]["year"].__setitem__("source_id", {}),
            r"example\.year.*source_id",
        ),
        (
            lambda record: record["fields"]["year"].__setitem__("value", 1900),
            r"example\.year.*value",
        ),
        (
            lambda record: record["fields"]["year"].__setitem__("value", True),
            r"example\.year.*value",
        ),
        (
            lambda record: record["fields"]["year"].__setitem__("value", float(record["fields"]["year"]["value"])),
            r"example\.year.*value",
        ),
        (lambda record: record.__setitem__("clock_name", "wrong"), r"example\.clock_name"),
        (lambda record: record.__setitem__("doi", "https://doi.org/wrong"), r"example\.doi"),
        (lambda record: record.__setitem__("reviewer", ""), r"example\.reviewer"),
        (lambda record: record.__setitem__("sources", []), r"example\.sources"),
        (
            lambda record: record["sources"].append(copy.deepcopy(record["sources"][0])),
            r"example\.sources.*duplicate",
        ),
        (
            lambda record: record["sources"][0].__setitem__("type", "website"),
            r"example\.sources.*type",
        ),
        (
            lambda record: record["sources"][0].__setitem__("type", []),
            r"example\.sources.*type",
        ),
        (
            lambda record: record["sources"][0].__setitem__("url", "http://example.test"),
            r"example\.sources.*url",
        ),
        (
            lambda record: record["sources"][0].__setitem__("url", "https://"),
            r"example\.sources.*url",
        ),
        (
            lambda record: record["sources"][0].__setitem__("accessed", "2026-7-18"),
            r"example\.sources.*accessed",
        ),
    ],
)
def test_validate_evidence_rejects_representative_invalid_values(registry, ledger, mutation, context):
    registry_record = copy.deepcopy(next(iter(registry.values())))
    ledger_record = copy.deepcopy(ledger[registry_record["clock_name"]])
    registry_record["clock_name"] = "example"
    ledger_record["clock_name"] = "example"
    invalid_registry = {"example": registry_record}
    invalid_ledger = {"example": ledger_record}
    mutation(ledger_record)

    with pytest.raises(ValueError, match=context):
        validate_evidence(invalid_registry, invalid_ledger)


@pytest.mark.parametrize(
    ("status", "source_type"),
    [
        ("paper-confirmed", "code"),
        ("supplement-confirmed", "paper"),
        ("code-confirmed", "paper"),
        ("author-confirmed", "paper"),
    ],
)
def test_resolved_evidence_status_must_match_source_type(registry, ledger, status, source_type):
    registry_record = copy.deepcopy(next(iter(registry.values())))
    clock_name = registry_record["clock_name"]
    ledger_record = copy.deepcopy(ledger[clock_name])
    evidence = ledger_record["fields"]["year"]
    ledger_record["reviewer"] = "reviewer"
    ledger_record["sources"].append(
        {
            "id": "mismatched-source",
            "type": source_type,
            "url": "https://example.org/source",
            "accessed": "2026-07-18",
        }
    )
    evidence["source_id"] = "mismatched-source"
    evidence["status"] = status
    evidence["locator"] = "page 1"

    with pytest.raises(ValueError, match=rf"{clock_name}\.year.*{status}.*{source_type}"):
        validate_evidence({clock_name: registry_record}, {clock_name: ledger_record})


def test_author_confirmed_evidence_accepts_author_communication_source(registry, ledger):
    registry_record = copy.deepcopy(next(iter(registry.values())))
    clock_name = registry_record["clock_name"]
    ledger_record = copy.deepcopy(ledger[clock_name])
    evidence = ledger_record["fields"]["year"]
    ledger_record["reviewer"] = "metadata-audit"
    ledger_record["sources"].append(
        {
            "id": "author-clarification",
            "type": "author communication",
            "url": "https://github.com/lcamillo/CpGPT",
            "accessed": "2026-07-18",
        }
    )
    evidence.update(
        {
            "status": "author-confirmed",
            "source_id": "author-clarification",
            "source_text": "Direct author clarification",
            "locator": "Direct author clarification in Codex task, 2026-07-18",
        }
    )

    validate_evidence({clock_name: registry_record}, {clock_name: ledger_record})


def test_resolved_evidence_rejects_provisional_assignment(registry, ledger):
    registry_record = copy.deepcopy(next(iter(registry.values())))
    clock_name = registry_record["clock_name"]
    ledger_record = copy.deepcopy(ledger[clock_name])
    ledger_record["reviewer"] = "unassigned"
    ledger_record["fields"]["year"]["status"] = "paper-confirmed"

    with pytest.raises(ValueError, match=rf"{clock_name}\..*reviewer"):
        validate_evidence({clock_name: registry_record}, {clock_name: ledger_record})


def test_resolved_evidence_rejects_provisional_locator(registry, ledger):
    registry_record = copy.deepcopy(next(iter(registry.values())))
    clock_name = registry_record["clock_name"]
    ledger_record = copy.deepcopy(ledger[clock_name])
    ledger_record["reviewer"] = "reviewer"
    ledger_record["fields"]["year"]["status"] = "paper-confirmed"
    ledger_record["fields"]["year"]["locator"] = "pending source audit"

    with pytest.raises(ValueError, match=rf"{clock_name}\.year.*locator"):
        validate_evidence({clock_name: registry_record}, {clock_name: ledger_record})


def test_resolved_evidence_rejects_provisional_source_text(registry, ledger):
    registry_record = copy.deepcopy(next(iter(registry.values())))
    clock_name = registry_record["clock_name"]
    ledger_record = copy.deepcopy(ledger[clock_name])
    evidence = ledger_record["fields"]["year"]
    ledger_record["reviewer"] = "reviewer"
    evidence["status"] = "paper-confirmed"
    evidence["locator"] = "page 1"
    evidence["source_text"] = "pending source audit"

    with pytest.raises(ValueError, match=rf"{clock_name}\.year.*source_text"):
        validate_evidence({clock_name: registry_record}, {clock_name: ledger_record})
