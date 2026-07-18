"""Temporary helpers for partitioning and validating the clock metadata paper audit."""

import argparse
import json
import os
import re
import tempfile
from contextlib import suppress
from datetime import date
from pathlib import Path
from urllib.parse import unquote, urlsplit

try:
    from .validate_metadata import (
        ARRAY_FIELDS,
        AUDITED_FIELDS,
        CONFIRMED_SOURCE_TYPES,
        CONTROLLED_SCALAR_FIELDS,
        EVIDENCE_STATUSES,
        PROVISIONAL_SOURCE_TEXT,
        SOURCE_TYPES,
        load_json,
    )
except ImportError:  # Direct script execution.
    from validate_metadata import (
        ARRAY_FIELDS,
        AUDITED_FIELDS,
        CONFIRMED_SOURCE_TYPES,
        CONTROLLED_SCALAR_FIELDS,
        EVIDENCE_STATUSES,
        PROVISIONAL_SOURCE_TEXT,
        SOURCE_TYPES,
        load_json,
    )


def normalize_doi(value):
    """Return a DOI in canonical resolver URL form."""
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
    if re.fullmatch(r"10\.[0-9]{4,9}/[^\s?#]+", core) is None:
        raise ValueError(
            f"DOI {value!r} must match 10.<4-9 digits>/<nonempty suffix> without whitespace, query, or fragment"
        )
    return prefix + core


def _validate_batch_count(batch_count):
    if type(batch_count) is not int or batch_count <= 0:
        raise ValueError("batch_count must be a positive integer")


def _families(registry):
    if type(registry) is not dict:
        raise ValueError("registry: expected a top-level object")
    grouped = {}
    for clock_name, record in registry.items():
        if type(clock_name) is not str or not clock_name:
            raise ValueError("registry: clock names must be nonempty strings")
        if type(record) is not dict:
            raise ValueError(f"{clock_name}: registry record must be an object")
        try:
            doi = normalize_doi(record.get("doi"))
        except ValueError as error:
            raise ValueError(f"{clock_name}.doi: {error}") from error
        grouped.setdefault(doi, []).append(clock_name)
    families = [{"doi": doi, "clock_names": sorted(clock_names)} for doi, clock_names in grouped.items()]
    return sorted(families, key=lambda family: (-len(family["clock_names"]), family["doi"]))


def assign_families(registry, batch_count):
    """Greedily assign whole DOI families to the least-loaded batch."""
    _validate_batch_count(batch_count)
    width = max(2, len(str(batch_count)))
    bins = [
        {"batch": str(index + 1).zfill(width), "clock_count": 0, "paper_count": 0, "families": []}
        for index in range(batch_count)
    ]
    for family in _families(registry):
        target = min(bins, key=lambda batch: (batch["clock_count"], int(batch["batch"])))
        target["families"].append(family)
        target["clock_count"] += len(family["clock_names"])
        target["paper_count"] += 1
    return bins


def _write_json_atomic(path, value):
    path = Path(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary:
            json.dump(value, temporary, indent=2, ensure_ascii=False)
            temporary.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def build_manifest(registry_path, output_dir, batch_count=12):
    """Build deterministic paper assignments and batch audit templates."""
    _validate_batch_count(batch_count)
    registry = load_json(registry_path)
    assignments = assign_families(registry, batch_count)
    expected_clocks = set(registry)
    seen_clocks = set()
    seen_dois = set()
    batch_payloads = []
    manifest_batches = []

    for assignment in assignments:
        papers = []
        for family in assignment["families"]:
            doi = family["doi"]
            if doi in seen_dois:
                raise ValueError(f"manifest: duplicate DOI family {doi!r}")
            seen_dois.add(doi)
            current_metadata = {}
            for clock_name in family["clock_names"]:
                if clock_name in seen_clocks:
                    raise ValueError(f"manifest: duplicate clock {clock_name!r}")
                seen_clocks.add(clock_name)
                current_metadata[clock_name] = registry[clock_name]
            papers.append(
                {
                    "doi": doi,
                    "clock_names": family["clock_names"],
                    "current_metadata": current_metadata,
                    "audited_fields": list(AUDITED_FIELDS),
                }
            )
        batch = assignment["batch"]
        batch_payloads.append(
            (
                f"batch-{batch}.json",
                {
                    "schema_version": 1,
                    "batch": batch,
                    "paper_count": assignment["paper_count"],
                    "clock_count": assignment["clock_count"],
                    "papers": papers,
                },
            )
        )
        manifest_batches.append(
            {
                "batch": batch,
                "paper_count": assignment["paper_count"],
                "clock_count": assignment["clock_count"],
                "path": f"batch-{batch}.json",
            }
        )

    if seen_clocks != expected_clocks:
        missing = sorted(expected_clocks - seen_clocks)
        extra = sorted(seen_clocks - expected_clocks)
        raise ValueError(f"manifest clock set mismatch: missing={missing}, extra={extra}")

    manifest = {
        "schema_version": 1,
        "paper_count": len(seen_dois),
        "clock_count": len(seen_clocks),
        "batches": manifest_batches,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in batch_payloads:
        _write_json_atomic(output_dir / filename, payload)
    _write_json_atomic(output_dir / "manifest.json", manifest)
    return manifest


def _require_exact_keys(value, expected, context):
    if type(value) is not dict:
        raise ValueError(f"{context}: expected an object")
    actual = set(value)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise ValueError(f"{context}: key mismatch: missing={missing}, extra={extra}")


def _load_batch_assignments(batch):
    _require_exact_keys(
        batch,
        {"schema_version", "batch", "paper_count", "clock_count", "papers"},
        "batch",
    )
    if type(batch["schema_version"]) is not int or batch["schema_version"] != 1:
        raise ValueError("batch.schema_version: expected integer 1")
    batch_id = batch["batch"]
    if type(batch_id) is not str or not batch_id.isdigit() or len(batch_id) < 2:
        raise ValueError("batch.batch: expected a zero-padded batch string")
    if type(batch["papers"]) is not list:
        raise ValueError("batch.papers: expected a list")

    assignments = {}
    dois = set()
    for index, paper in enumerate(batch["papers"]):
        context = f"batch.papers[{index}]"
        _require_exact_keys(
            paper,
            {"doi", "clock_names", "current_metadata", "audited_fields"},
            context,
        )
        doi = normalize_doi(paper["doi"])
        if doi != paper["doi"]:
            raise ValueError(f"{context}.doi: must be normalized")
        if doi in dois:
            raise ValueError(f"{context}.doi: duplicate DOI {doi!r}")
        dois.add(doi)
        clock_names = paper["clock_names"]
        metadata = paper["current_metadata"]
        if (
            type(clock_names) is not list
            or not clock_names
            or any(type(name) is not str or not name for name in clock_names)
        ):
            raise ValueError(f"{context}.clock_names: expected a nonempty list of nonempty strings")
        if clock_names != sorted(clock_names) or len(clock_names) != len(set(clock_names)):
            raise ValueError(f"{context}.clock_names: expected unique alphabetical names")
        if type(metadata) is not dict or set(metadata) != set(clock_names):
            raise ValueError(f"{context}.current_metadata: clock set mismatch")
        if paper["audited_fields"] != list(AUDITED_FIELDS):
            raise ValueError(f"{context}.audited_fields: expected the fixed audited field list")
        for clock_name in clock_names:
            if clock_name in assignments:
                raise ValueError(f"{context}: duplicate clock {clock_name!r}")
            current = metadata[clock_name]
            if type(current) is not dict:
                raise ValueError(f"{context}.current_metadata.{clock_name}: expected an object")
            if normalize_doi(current.get("doi")) != doi:
                raise ValueError(f"{context}.current_metadata.{clock_name}.doi: DOI mismatch")
            for field in AUDITED_FIELDS:
                if field not in current:
                    raise ValueError(f"{context}.current_metadata.{clock_name}: missing field {field!r}")
            assignments[clock_name] = (doi, current)
    if type(batch["paper_count"]) is not int or batch["paper_count"] != len(dois):
        raise ValueError("batch.paper_count: does not match papers")
    if type(batch["clock_count"]) is not int or batch["clock_count"] != len(assignments):
        raise ValueError("batch.clock_count: does not match assigned clocks")
    return batch_id, assignments, len(dois)


def _validate_source(clock_name, source, index, source_types):
    context = f"{clock_name}.sources[{index}]"
    _require_exact_keys(source, {"id", "type", "url", "accessed"}, context)
    source_id = source["id"]
    if type(source_id) is not str or not source_id.strip():
        raise ValueError(f"{context}.id: expected a nonempty string")
    if source_id in source_types:
        raise ValueError(f"{clock_name}.sources: duplicate source id {source_id!r}")
    source_type = source["type"]
    if type(source_type) is not str or source_type not in SOURCE_TYPES:
        raise ValueError(f"{context}.type: must be one of {sorted(SOURCE_TYPES)}")
    url = source["url"]
    parsed = urlsplit(url) if type(url) is str else None
    if parsed is None or parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"{context}.url: must be an https URL")
    accessed = source["accessed"]
    try:
        parsed_date = date.fromisoformat(accessed) if type(accessed) is str else None
    except ValueError:
        parsed_date = None
    if parsed_date is None or parsed_date.isoformat() != accessed:
        raise ValueError(f"{context}.accessed: must use ISO YYYY-MM-DD")
    source_types[source_id] = source_type


def _validate_evidence_value(clock_name, field, value, current):
    if field in ARRAY_FIELDS:
        valid = type(value) is list
    elif field in ("year", "n_features"):
        valid = type(value) is int
    elif field == "notes":
        valid = value is None or type(value) is str
    elif field in CONTROLLED_SCALAR_FIELDS:
        valid = type(value) is str
    elif type(current) is list:
        valid = type(value) is list
    else:
        valid = type(value) is type(current)
    if not valid:
        raise ValueError(f"{clock_name}.{field}.value: wrong value type")


def _validate_evidence_field(clock_name, field, evidence, current, reviewer, source_types):
    context = f"{clock_name}.{field}"
    _require_exact_keys(
        evidence,
        {"value", "source_text", "source_id", "locator", "status", "note"},
        context,
    )
    _validate_evidence_value(clock_name, field, evidence["value"], current)
    for key in ("source_text", "source_id", "locator"):
        if type(evidence[key]) is not str or not evidence[key].strip():
            raise ValueError(f"{context}.{key}: expected a nonempty string")
    if type(evidence["note"]) is not str:
        raise ValueError(f"{context}.note: expected a string")
    status = evidence["status"]
    if type(status) is not str or status not in EVIDENCE_STATUSES:
        raise ValueError(f"{context}.status: {status!r} is not allowed")
    source_id = evidence["source_id"]
    if source_id not in source_types:
        raise ValueError(f"{context}.source_id: {source_id!r} is not defined")
    expected_source_type = CONFIRMED_SOURCE_TYPES.get(status)
    if expected_source_type is not None and source_types[source_id] != expected_source_type:
        raise ValueError(
            f"{context}: {status} requires source type {expected_source_type!r}, got {source_types[source_id]!r}"
        )
    if status != "unresolved":
        if reviewer.strip().casefold() == "unassigned":
            raise ValueError(f"{context}: resolved evidence requires an assigned reviewer")
        if evidence["locator"].strip().casefold() == "pending source audit":
            raise ValueError(f"{context}: resolved evidence requires a specific locator")
        source_text = evidence["source_text"].strip().casefold()
        if source_text in PROVISIONAL_SOURCE_TEXT or (
            source_text.startswith("no current ") and source_text.endswith(" recorded.")
        ):
            raise ValueError(f"{context}: resolved evidence cannot use provisional source_text")


def validate_shard(batch_path, shard_path):
    """Validate one review shard against its generated assignment batch."""
    batch = load_json(batch_path)
    shard = load_json(shard_path)
    batch_id, assignments, paper_count = _load_batch_assignments(batch)
    _require_exact_keys(shard, {"schema_version", "batch", "reviewer", "records"}, "shard")
    if type(shard["schema_version"]) is not int or shard["schema_version"] != 1:
        raise ValueError("shard.schema_version: expected integer 1")
    if shard["batch"] != batch_id:
        raise ValueError(f"shard.batch: expected {batch_id!r}")
    expected_reviewer = f"paper-audit-{batch_id}"
    if shard["reviewer"] != expected_reviewer:
        raise ValueError(f"shard.reviewer: expected {expected_reviewer!r}")
    records = shard["records"]
    if type(records) is not list:
        raise ValueError("shard.records: expected a list")
    names = [record.get("clock_name") if type(record) is dict else None for record in records]
    if not all(type(name) is str for name in names):
        raise ValueError("shard.records: each record must have a string clock_name")
    if names != sorted(names):
        raise ValueError("shard.records: records must be in alphabetical order")
    if len(names) != len(set(names)):
        raise ValueError("shard.records: clock names must be unique")
    if set(names) != set(assignments):
        missing = sorted(set(assignments) - set(names))
        extra = sorted(set(names) - set(assignments))
        raise ValueError(f"shard clock set mismatch: missing={missing}, extra={extra}")

    for record in records:
        clock_name = record["clock_name"]
        _require_exact_keys(
            record,
            {"clock_name", "doi", "reviewer", "sources", "fields", "access_issues"},
            clock_name,
        )
        assigned_doi, current = assignments[clock_name]
        if record["doi"] != assigned_doi:
            raise ValueError(f"{clock_name}.doi: DOI must exactly match assigned DOI")
        if record["reviewer"] != shard["reviewer"]:
            raise ValueError(f"{clock_name}.reviewer: must match shard reviewer")
        sources = record["sources"]
        if type(sources) is not list or not sources:
            raise ValueError(f"{clock_name}.sources: expected a nonempty list")
        source_types = {}
        for index, source in enumerate(sources):
            _validate_source(clock_name, source, index, source_types)
        fields = record["fields"]
        if type(fields) is not dict or set(fields) != set(AUDITED_FIELDS):
            missing = sorted(set(AUDITED_FIELDS) - set(fields)) if type(fields) is dict else []
            extra = sorted(set(fields) - set(AUDITED_FIELDS)) if type(fields) is dict else []
            raise ValueError(f"{clock_name}.fields: field mismatch: missing={missing}, extra={extra}")
        for field in AUDITED_FIELDS:
            _validate_evidence_field(
                clock_name,
                field,
                fields[field],
                current[field],
                record["reviewer"],
                source_types,
            )
        access_issues = record["access_issues"]
        if type(access_issues) is not list or any(type(issue) is not str for issue in access_issues):
            raise ValueError(f"{clock_name}.access_issues: expected a list of strings")
    return {"batch": batch_id, "paper_count": paper_count, "clock_count": len(assignments)}


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-manifest")
    build.add_argument("--registry", required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--batches", type=int, default=12)
    validate = subparsers.add_parser("validate-shard")
    validate.add_argument("--batch", required=True)
    validate.add_argument("--shard", required=True)
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    if arguments.command == "build-manifest":
        manifest = build_manifest(arguments.registry, arguments.output_dir, arguments.batches)
        print(
            f"built {len(manifest['batches'])} batches for "
            f"{manifest['paper_count']} papers and {manifest['clock_count']} clocks"
        )
    else:
        summary = validate_shard(arguments.batch, arguments.shard)
        print(f"validated batch {summary['batch']}: {summary['paper_count']} papers, {summary['clock_count']} clocks")


if __name__ == "__main__":
    main()
