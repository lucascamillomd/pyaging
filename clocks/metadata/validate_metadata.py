import json
import math
import re
from datetime import date
from pathlib import Path
from urllib.parse import unquote, urlsplit

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
EVIDENCE_STATUSES = {
    "paper-confirmed",
    "supplement-confirmed",
    "code-confirmed",
    "author-confirmed",
    "unresolved",
}
ADMIN_FIELDS = {"approved_by_author", "research_only", "citations", "citations_date"}
SOURCE_TYPES = {"paper", "supplement", "code", "author communication"}
CONFIRMED_SOURCE_TYPES = {
    "paper-confirmed": "paper",
    "supplement-confirmed": "supplement",
    "code-confirmed": "code",
    "author-confirmed": "author communication",
}
PROVISIONAL_SOURCE_TEXT = {"pending source audit", "unresolved", "unknown"}
NONEMPTY_STRING_FIELDS = {
    "data_type",
    "species",
    "model_type",
    "population",
    "journal",
    "last_author",
    "notes",
}


def _reject_non_finite(value):
    raise ValueError(f"non-finite JSON number {value!r} is not allowed")


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r} is not allowed")
        result[key] = value
    return result


def _parse_json(text, context):
    try:
        return json.loads(
            text,
            parse_constant=_reject_non_finite,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{context}: invalid JSON at column {error.colno}: {error.msg}") from error
    except ValueError as error:
        raise ValueError(f"{context}: {error}") from error


def load_json(path):
    path = Path(path)
    return _parse_json(path.read_text(encoding="utf-8"), str(path))


def load_ledger(path):
    ledger = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = _parse_json(line, f"line {line_number}")
        if type(record) is not dict:
            raise ValueError(f"line {line_number}: expected an object")
        clock_name = record.get("clock_name")
        if type(clock_name) is not str or not clock_name.strip():
            raise ValueError(f"line {line_number}: clock_name must be a nonempty string")
        if clock_name in ledger:
            raise ValueError(f"duplicate clock_name {clock_name!r} at line {line_number}")
        if ledger and clock_name < next(reversed(ledger)):
            raise ValueError(f"line {line_number}: clock_name records must be in alphabetical order")
        ledger[clock_name] = record
    return ledger


def _fail(clock_name, field, message):
    raise ValueError(f"{clock_name}.{field}: {message}")


def _same_json_value(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, float) and (not math.isfinite(left) or not math.isfinite(right)):
        return False
    if type(left) is list:
        return len(left) == len(right) and all(_same_json_value(a, b) for a, b in zip(left, right))
    if type(left) is dict:
        return left.keys() == right.keys() and all(_same_json_value(left[key], right[key]) for key in left)
    return left == right


def normalize_doi(value):
    """Return a strict DOI as a lowercase canonical resolver URL."""
    if type(value) is not str or not value.strip():
        raise ValueError("DOI must be a nonempty string")
    value = value.strip()
    prefix = "https://doi.org/"
    if value.casefold().startswith(prefix):
        try:
            parsed = urlsplit(value)
        except ValueError as error:
            raise ValueError(f"DOI URL {value!r} is invalid") from error
        if (
            parsed.scheme.casefold() != "https"
            or parsed.netloc.casefold() != "doi.org"
            or parsed.query
            or parsed.fragment
            or not parsed.path.startswith("/")
        ):
            raise ValueError(f"DOI URL {value!r} must be an unadorned https://doi.org/ URL")
        core = parsed.path[1:]
    else:
        if "://" in value or value.casefold().startswith("doi.org/"):
            raise ValueError(f"DOI {value!r} must be bare or use the https://doi.org/ prefix")
        core = value
    if re.search(r"%(?![0-9A-Fa-f]{2})", core):
        raise ValueError(f"DOI {value!r} contains an invalid percent escape")
    try:
        core = unquote(core, errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"DOI {value!r} contains invalid UTF-8 percent escapes") from error
    if re.search(r"%[0-9A-Fa-f]{2}", core):
        raise ValueError(f"DOI {value!r} must not contain double-encoded percent escapes")
    if re.fullmatch(r"10\.[0-9]{4,9}/[^\s?#]+", core) is None:
        raise ValueError(
            f"DOI {value!r} must match 10.<4-9 digits>/<nonempty suffix> without whitespace, query, or fragment"
        )
    return prefix + core.lower()


def validate_audited_value(field, value, context):
    """Validate canonical audited-value shape without checking vocabulary membership."""
    if field not in AUDITED_FIELDS:
        raise ValueError(f"{context}: unknown audited field {field!r}")
    if field in ARRAY_FIELDS:
        if type(value) is not list or not value:
            raise ValueError(f"{context}: expected a nonempty list")
        if any(type(item) is not str or not item.strip() for item in value):
            raise ValueError(f"{context}: expected nonempty string values")
        if len(value) != len(set(value)):
            raise ValueError(f"{context}: values must be unique")
    elif field in NONEMPTY_STRING_FIELDS:
        if type(value) is not str or not value.strip():
            raise ValueError(f"{context}: expected a nonempty string")
    elif field in ("year", "n_features"):
        if type(value) is not int:
            raise ValueError(f"{context}: expected an integer")
    elif field == "citation":
        if type(value) is str:
            valid = bool(value.strip())
        elif type(value) is list:
            valid = (
                bool(value)
                and all(type(item) is str and item.strip() for item in value)
                and len(value) == len(set(value))
            )
        else:
            valid = False
        if not valid:
            raise ValueError(f"{context}: expected a nonempty string or unique nonempty string list")
    elif field == "doi":
        if type(value) is not str or not value.strip().casefold().startswith("https://doi.org/"):
            raise ValueError(f"{context}: expected an https://doi.org/ DOI URL")
        try:
            normalize_doi(value)
        except ValueError as error:
            raise ValueError(f"{context}: {error}") from error


def validate_vocabulary(vocabulary):
    if type(vocabulary) is not dict:
        raise ValueError("vocabulary: expected top-level object")
    if type(vocabulary.get("schema_version")) is not int or vocabulary["schema_version"] != 1:
        raise ValueError("vocabulary.schema_version: expected integer 1")
    if vocabulary.get("array_fields") != list(ARRAY_FIELDS) or type(vocabulary["array_fields"]) is not list:
        raise ValueError(f"vocabulary.array_fields: expected exactly {list(ARRAY_FIELDS)!r}")

    fields = vocabulary.get("fields")
    if type(fields) is not dict:
        raise ValueError("vocabulary.fields: expected an object")
    required_fields = set(ARRAY_FIELDS) | set(CONTROLLED_SCALAR_FIELDS)
    missing = sorted(required_fields - set(fields))
    if missing:
        raise ValueError(f"vocabulary.fields: missing controlled fields {missing}")

    for field, descriptor in fields.items():
        context = f"vocabulary.{field}"
        if type(descriptor) is not dict:
            raise ValueError(f"{context}: expected an object")
        description = descriptor.get("description")
        if type(description) is not str or not description.strip():
            raise ValueError(f"{context}.description: expected a nonempty string")
        values = descriptor.get("values")
        if type(values) is not list:
            raise ValueError(f"{context}.values: expected a list")
        if not values:
            raise ValueError(f"{context}.values: expected a nonempty list")
        if any(type(value) is not str or not value.strip() for value in values):
            raise ValueError(f"{context}.values: expected nonempty string values")
        if len(values) != len(set(values)):
            raise ValueError(f"{context}.values: values must be unique")
        aliases = descriptor.get("aliases")
        if type(aliases) is not dict:
            raise ValueError(f"{context}.aliases: expected an object")
        for alias, target in aliases.items():
            if type(alias) is not str or not alias.strip():
                raise ValueError(f"{context}.aliases: alias names must be nonempty strings")
            if type(target) is not str or target not in values:
                raise ValueError(f"{context}.aliases: alias {alias!r} must reference an existing value")


def _validate_registry_field_types(clock_name, record):
    for field in ("approved_by_author", "citations_date"):
        if type(record[field]) is not str or not record[field].strip():
            _fail(clock_name, field, "expected a nonempty string")

    if type(record["citations"]) is not int:
        _fail(clock_name, "citations", "expected an integer")
    if record["research_only"] is not None and type(record["research_only"]) is not bool:
        _fail(clock_name, "research_only", "expected a boolean or null")


def validate_registry(registry, vocabulary):
    validate_vocabulary(vocabulary)
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
        for field in AUDITED_FIELDS:
            validate_audited_value(field, record[field], f"{clock_name}.{field}")
        _validate_registry_field_types(clock_name, record)

        for field in ARRAY_FIELDS:
            value = record[field]
            allowed = vocabulary_fields.get(field, {}).get("values", [])
            for item in value:
                if not any(_same_json_value(item, allowed_item) for allowed_item in allowed):
                    _fail(clock_name, field, f"{item!r} is not in the controlled vocabulary")

        for field in CONTROLLED_SCALAR_FIELDS:
            value = record[field]
            allowed = vocabulary_fields.get(field, {}).get("values", [])
            if not any(_same_json_value(value, allowed_item) for allowed_item in allowed):
                _fail(clock_name, field, f"{value!r} is not in the controlled vocabulary")


def validate_evidence(registry, ledger):
    if set(registry) != set(ledger):
        missing = sorted(set(registry) - set(ledger))
        extra = sorted(set(ledger) - set(registry))
        raise ValueError(f"ledger clock set mismatch: missing={missing}, extra={extra}")

    for clock_name, registry_record in registry.items():
        ledger_record = ledger[clock_name]
        if type(ledger_record) is not dict:
            _fail(clock_name, "record", "expected an object")
        if ledger_record.get("clock_name") != clock_name:
            _fail(clock_name, "clock_name", "must exactly match registry key")
        if not _same_json_value(ledger_record.get("doi"), registry_record["doi"]):
            _fail(clock_name, "doi", "must exactly match registry DOI")
        reviewer = ledger_record.get("reviewer")
        if type(reviewer) is not str or not reviewer.strip():
            _fail(clock_name, "reviewer", "must be a nonempty string")

        sources = ledger_record.get("sources")
        if type(sources) is not list or not sources:
            _fail(clock_name, "sources", "expected a nonempty list")
        source_types = {}
        for index, source in enumerate(sources):
            source_field = f"sources[{index}]"
            if type(source) is not dict:
                _fail(clock_name, source_field, "expected an object")
            source_id = source.get("id")
            if type(source_id) is not str or not source_id.strip():
                _fail(clock_name, f"{source_field}.id", "must be a nonempty string")
            if source_id in source_types:
                _fail(clock_name, "sources", f"duplicate source id {source_id!r}")
            source_type = source.get("type")
            if type(source_type) is not str or source_type not in SOURCE_TYPES:
                _fail(clock_name, f"{source_field}.type", f"must be one of {sorted(SOURCE_TYPES)}")
            source_types[source_id] = source_type
            url = source.get("url")
            parsed_url = urlsplit(url) if type(url) is str else None
            if parsed_url is None or parsed_url.scheme != "https" or not parsed_url.netloc:
                _fail(clock_name, f"{source_field}.url", "must be an https URL")
            accessed = source.get("accessed")
            try:
                parsed_accessed = date.fromisoformat(accessed) if type(accessed) is str else None
            except ValueError:
                parsed_accessed = None
            if parsed_accessed is None or parsed_accessed.isoformat() != accessed:
                _fail(clock_name, f"{source_field}.accessed", "must use ISO YYYY-MM-DD")

        fields = ledger_record.get("fields")
        if type(fields) is not dict:
            _fail(clock_name, "fields", "expected an object")

        for field in AUDITED_FIELDS:
            if field not in fields:
                _fail(clock_name, field, "evidence is missing")
            evidence = fields[field]
            if type(evidence) is not dict:
                _fail(clock_name, field, "evidence must be an object")
            status = evidence.get("status")
            if type(status) is not str or not status.strip():
                _fail(clock_name, field, "status must be a nonempty string")
            if status not in EVIDENCE_STATUSES:
                _fail(clock_name, field, f"status {status!r} is not allowed")
            source_text = evidence.get("source_text")
            if type(source_text) is not str or not source_text.strip():
                _fail(clock_name, field, "source_text must be nonempty")
            locator = evidence.get("locator")
            if type(locator) is not str or not locator.strip():
                _fail(clock_name, field, "locator must be nonempty")
            source_id = evidence.get("source_id")
            if type(source_id) is not str or not source_id.strip():
                _fail(clock_name, field, "source_id must be a nonempty string")
            if source_id not in source_types:
                _fail(clock_name, field, f"source_id {source_id!r} is not defined")
            if not _same_json_value(evidence.get("value"), registry_record[field]):
                _fail(clock_name, field, "value does not exactly match registry")
            expected_source_type = CONFIRMED_SOURCE_TYPES.get(status)
            if expected_source_type is not None and source_types[source_id] != expected_source_type:
                _fail(
                    clock_name,
                    field,
                    f"{status} requires source type {expected_source_type!r}, got {source_types[source_id]!r}",
                )
            if status != "unresolved":
                if reviewer.strip().casefold() == "unassigned":
                    _fail(clock_name, field, "resolved evidence requires an assigned reviewer")
                if locator.strip().casefold() == "pending source audit":
                    _fail(clock_name, field, "resolved evidence requires a specific locator")
                normalized_source_text = source_text.strip().casefold()
                if normalized_source_text in PROVISIONAL_SOURCE_TEXT or (
                    normalized_source_text.startswith("no current ") and normalized_source_text.endswith(" recorded.")
                ):
                    _fail(clock_name, field, "resolved evidence cannot use provisional source_text")


def validate_artifact_consistency(root):
    raise NotImplementedError("artifact consistency is implemented after migration")
