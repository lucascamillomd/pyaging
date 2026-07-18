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


@pytest.mark.parametrize(
    ("mutation", "context"),
    [
        (lambda record: record.__setitem__("tissue", "blood"), r"example\.tissue"),
        (lambda record: record.__setitem__("year", True), r"example\.year"),
        (lambda record: record.__setitem__("data_type", "unknown"), r"example\.data_type"),
        (lambda record: record.pop("approved_by_author"), r"example\.approved_by_author"),
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
        (
            lambda record: record["fields"]["year"].__setitem__("status", "guessed"),
            r"example\.year.*status",
        ),
        (
            lambda record: record["fields"]["year"].__setitem__("source_id", "missing"),
            r"example\.year.*source_id",
        ),
        (
            lambda record: record["fields"]["year"].__setitem__("value", 1900),
            r"example\.year.*value",
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
