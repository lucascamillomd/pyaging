import json
from pathlib import Path

ARRAY_FIELDS = ("tissue", "platform", "predicts", "training_target", "unit")
CONTROLLED_SCALAR_FIELDS = ("data_type", "species", "model_type", "population")
AUDITED_FIELDS = (
    "data_type",
    "species",
    "year",
    "citation",
    "doi",
    "notes",
    "tissue",
    "predicts",
    "training_target",
    "unit",
    "model_type",
    "platform",
    "population",
    "journal",
    "last_author",
    "n_features",
)
EVIDENCE_STATUSES = {"paper-confirmed", "supplement-confirmed", "code-confirmed", "unresolved"}
ADMIN_FIELDS = {"approved_by_author", "research_only", "citations", "citations_date"}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_ledger(path):
    ledger = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        clock_name = record.get("clock_name")
        if clock_name in ledger:
            raise ValueError(f"duplicate clock_name {clock_name!r} at line {line_number}")
        ledger[clock_name] = record
    return ledger


def _fail(clock_name, field, message):
    raise ValueError(f"{clock_name}.{field}: {message}")


def validate_registry(registry, vocabulary):
    if type(registry) is not dict:
        raise ValueError("registry: expected top-level object")
    if list(registry) != sorted(registry):
        raise ValueError("registry: clock keys must be in alphabetical order")

    vocabulary_fields = vocabulary.get("fields", {})
    for clock_name, record in registry.items():
        if not isinstance(clock_name, str) or not clock_name or clock_name != clock_name.lower():
            _fail(clock_name, "clock_name", "key must be nonempty lowercase text")
        if type(record) is not dict:
            _fail(clock_name, "record", "expected object")
        if record.get("clock_name") != clock_name:
            _fail(clock_name, "clock_name", "must exactly match registry key")

        for field in (*AUDITED_FIELDS, *ADMIN_FIELDS):
            if field not in record:
                _fail(clock_name, field, "required curated field is missing")

        for field in ARRAY_FIELDS:
            value = record[field]
            if type(value) is not list:
                _fail(clock_name, field, "expected a list")
            if not value:
                _fail(clock_name, field, "must not be empty")
            if len(value) != len({json.dumps(item, sort_keys=True) for item in value}):
                _fail(clock_name, field, "values must be unique")
            allowed = vocabulary_fields.get(field, {}).get("values", [])
            for item in value:
                if item not in allowed:
                    _fail(clock_name, field, f"{item!r} is not in the controlled vocabulary")

        for field in CONTROLLED_SCALAR_FIELDS:
            value = record[field]
            allowed = vocabulary_fields.get(field, {}).get("values", [])
            if value not in allowed:
                _fail(clock_name, field, f"{value!r} is not in the controlled vocabulary")

        for field in ("year", "n_features"):
            if type(record[field]) is not int:
                _fail(clock_name, field, "expected an integer")

        if not isinstance(record["doi"], str) or not record["doi"].startswith("https://doi.org/"):
            _fail(clock_name, "doi", "must start with https://doi.org/")


def validate_evidence(registry, ledger):
    if set(registry) != set(ledger):
        missing = sorted(set(registry) - set(ledger))
        extra = sorted(set(ledger) - set(registry))
        raise ValueError(f"ledger clock set mismatch: missing={missing}, extra={extra}")

    for clock_name, registry_record in registry.items():
        ledger_record = ledger[clock_name]
        sources = ledger_record.get("sources")
        if type(sources) is not list:
            _fail(clock_name, "sources", "expected a list")
        source_ids = {source.get("id") for source in sources if type(source) is dict}
        fields = ledger_record.get("fields")
        if type(fields) is not dict:
            _fail(clock_name, "fields", "expected an object")

        for field in AUDITED_FIELDS:
            if field not in fields:
                _fail(clock_name, field, "evidence is missing")
            evidence = fields[field]
            if evidence.get("status") not in EVIDENCE_STATUSES:
                _fail(clock_name, field, f"status {evidence.get('status')!r} is not allowed")
            if not isinstance(evidence.get("source_text"), str) or not evidence["source_text"].strip():
                _fail(clock_name, field, "source_text must be nonempty")
            if not isinstance(evidence.get("locator"), str) or not evidence["locator"].strip():
                _fail(clock_name, field, "locator must be nonempty")
            if evidence.get("source_id") not in source_ids:
                _fail(clock_name, field, f"source_id {evidence.get('source_id')!r} is not defined")
            if evidence.get("value") != registry_record[field]:
                _fail(clock_name, field, "value does not exactly match registry")


def validate_artifact_consistency(root):
    raise NotImplementedError("artifact consistency is implemented after migration")
