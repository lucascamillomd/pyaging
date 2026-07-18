"""Temporary helpers for partitioning and validating the clock metadata paper audit."""

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
from collections import Counter
from contextlib import suppress
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

try:
    from .validate_metadata import (
        ARRAY_FIELDS,
        AUDITED_FIELDS,
        CONFIRMED_SOURCE_TYPES,
        CONTROLLED_SCALAR_FIELDS,
        EVIDENCE_STATUSES,
        PROVISIONAL_SOURCE_TEXT,
        SOURCE_TYPES,
        _parse_json,
        load_json,
        normalize_doi,
        validate_audited_value,
        validate_registry,
        validate_vocabulary,
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
        _parse_json,
        load_json,
        normalize_doi,
        validate_audited_value,
        validate_registry,
        validate_vocabulary,
    )


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
    for _, payload in batch_payloads:
        _load_batch_assignments(payload)

    output_dir = Path(output_dir)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"output directory {output_dir}: expected a directory")
        if any(output_dir.iterdir()):
            raise ValueError(f"output directory {output_dir}: must be empty")
    else:
        output_dir.mkdir(parents=True)
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
                validate_audited_value(
                    field,
                    current[field],
                    f"{context}.current_metadata.{clock_name}.{field}",
                )
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


def _validate_evidence_field(clock_name, field, evidence, reviewer, source_types):
    context = f"{clock_name}.{field}"
    _require_exact_keys(
        evidence,
        {"value", "source_text", "source_id", "locator", "status", "note"},
        context,
    )
    validate_audited_value(field, evidence["value"], f"{context}.value")
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


def _validate_shard_objects(batch, shard, batch_path="batch", shard_path="shard"):
    """Validate already-loaded shard objects so callers consume the checked bytes."""
    try:
        batch_id, assignments, paper_count = _load_batch_assignments(batch)
    except ValueError as error:
        raise ValueError(f"{batch_path}: {error}") from error
    _require_exact_keys(shard, {"schema_version", "batch", "reviewer", "records"}, str(shard_path))
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
        assigned_doi, _current = assignments[clock_name]
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
                record["reviewer"],
                source_types,
            )
        access_issues = record["access_issues"]
        if type(access_issues) is not list or any(type(issue) is not str for issue in access_issues):
            raise ValueError(f"{clock_name}.access_issues: expected a list of strings")
    return {"batch": batch_id, "paper_count": paper_count, "clock_count": len(assignments)}


def validate_shard(batch_path, shard_path):
    """Validate one review shard against its generated assignment batch."""
    batch = load_json(batch_path)
    shard = load_json(shard_path)
    return _validate_shard_objects(batch, shard, batch_path, shard_path)


def _refuse_existing(paths):
    for path in map(Path, paths):
        if path.exists():
            raise ValueError(f"output path already exists: {path}")
        if not path.parent.is_dir():
            raise ValueError(f"output parent is not an existing directory: {path.parent}")


def _load_manifest(path):
    manifest = load_json(path)
    _require_exact_keys(
        manifest,
        {"schema_version", "paper_count", "clock_count", "batches"},
        "manifest",
    )
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("manifest.schema_version: expected integer 1")
    for count_name in ("paper_count", "clock_count"):
        if type(manifest[count_name]) is not int or manifest[count_name] < 0:
            raise ValueError(f"manifest.{count_name}: expected a nonnegative integer")
    batches = manifest["batches"]
    if type(batches) is not list or not batches:
        raise ValueError("manifest.batches: expected a nonempty list")
    seen_ids = set()
    seen_paths = set()
    for index, descriptor in enumerate(batches):
        context = f"manifest.batches[{index}]"
        _require_exact_keys(descriptor, {"batch", "paper_count", "clock_count", "path"}, context)
        batch_id = descriptor["batch"]
        if type(batch_id) is not str or not batch_id.isdigit() or len(batch_id) < 2:
            raise ValueError(f"{context}.batch: expected a zero-padded batch string")
        expected_path = f"batch-{batch_id}.json"
        if descriptor["path"] != expected_path:
            raise ValueError(f"{context}.path: expected {expected_path!r}")
        if batch_id in seen_ids:
            raise ValueError(f"{context}.batch: duplicate batch {batch_id!r}")
        if descriptor["path"] in seen_paths:
            raise ValueError(f"{context}.path: duplicate path {descriptor['path']!r}")
        seen_ids.add(batch_id)
        seen_paths.add(descriptor["path"])
        for count_name in ("paper_count", "clock_count"):
            if type(descriptor[count_name]) is not int or descriptor[count_name] < 0:
                raise ValueError(f"{context}.{count_name}: expected a nonnegative integer")
    if [descriptor["batch"] for descriptor in batches] != sorted(seen_ids):
        raise ValueError("manifest.batches: declarations must be in batch order")
    if sum(item["paper_count"] for item in batches) != manifest["paper_count"]:
        raise ValueError("manifest.paper_count: does not match batch declarations")
    if sum(item["clock_count"] for item in batches) != manifest["clock_count"]:
        raise ValueError("manifest.clock_count: does not match batch declarations")
    return manifest


def _validate_record(record):
    if type(record) is not dict:
        raise ValueError("merged.records: each record must be an object")
    clock_name = record.get("clock_name")
    if type(clock_name) is not str or not clock_name:
        raise ValueError("merged.records: each record must have a nonempty clock_name")
    _require_exact_keys(
        record,
        {"clock_name", "doi", "reviewer", "sources", "fields", "access_issues"},
        clock_name,
    )
    if normalize_doi(record["doi"]) != record["doi"]:
        raise ValueError(f"{clock_name}.doi: must be normalized")
    reviewer = record["reviewer"]
    if type(reviewer) is not str or not reviewer.strip():
        raise ValueError(f"{clock_name}.reviewer: expected a nonempty string")
    sources = record["sources"]
    if type(sources) is not list or not sources:
        raise ValueError(f"{clock_name}.sources: expected a nonempty list")
    source_types = {}
    for index, source in enumerate(sources):
        _validate_source(clock_name, source, index, source_types)
    fields = record["fields"]
    if type(fields) is not dict or set(fields) != set(AUDITED_FIELDS):
        raise ValueError(f"{clock_name}.fields: expected the fixed audited field set")
    for field in AUDITED_FIELDS:
        _validate_evidence_field(clock_name, field, fields[field], reviewer, source_types)
    if type(fields["doi"]["value"]) is not str or fields["doi"]["value"] != record["doi"]:
        raise ValueError(f"{clock_name}.doi.value: must exactly match record DOI")
    access_issues = record["access_issues"]
    if type(access_issues) is not list or any(type(issue) is not str for issue in access_issues):
        raise ValueError(f"{clock_name}.access_issues: expected a list of strings")


def _load_merged(path):
    merged = load_json(path)
    _require_exact_keys(
        merged,
        {"schema_version", "paper_count", "clock_count", "records"},
        "merged",
    )
    if type(merged["schema_version"]) is not int or merged["schema_version"] != 1:
        raise ValueError("merged.schema_version: expected integer 1")
    records = merged["records"]
    if type(records) is not list:
        raise ValueError("merged.records: expected a list")
    names = []
    dois = set()
    for record in records:
        _validate_record(record)
        names.append(record["clock_name"])
        dois.add(record["doi"])
    if names != sorted(names):
        raise ValueError("merged.records: records must be in alphabetical order")
    if len(names) != len(set(names)):
        raise ValueError("merged.records: duplicate clock_name")
    if type(merged["clock_count"]) is not int or merged["clock_count"] != len(records):
        raise ValueError("merged.clock_count: does not match records")
    if type(merged["paper_count"]) is not int or merged["paper_count"] != len(dois):
        raise ValueError("merged.paper_count: does not match distinct record DOIs")
    return merged


def merge_shards(manifest_path, shards_dir, output_path):
    """Strictly reconcile all and only the batches declared by a manifest."""
    output_path = Path(output_path)
    _refuse_existing([output_path])
    manifest = _load_manifest(manifest_path)
    shards_dir = Path(shards_dir)
    if not shards_dir.is_dir():
        raise ValueError(f"shards directory does not exist: {shards_dir}")
    declared_batches = {item["path"] for item in manifest["batches"]}
    declared_shards = {f"shard-{item['batch']}.json" for item in manifest["batches"]}
    actual_batches = {path.name for path in shards_dir.glob("batch-*.json") if path.is_file()}
    actual_shards = {path.name for path in shards_dir.glob("shard-*.json") if path.is_file()}
    missing_batches = sorted(declared_batches - actual_batches)
    extra_batches = sorted(actual_batches - declared_batches)
    missing_shards = sorted(declared_shards - actual_shards)
    extra_shards = sorted(actual_shards - declared_shards)
    if missing_batches or extra_batches or missing_shards or extra_shards:
        raise ValueError(
            "shard file set mismatch: "
            f"missing batch={missing_batches}, extra batch={extra_batches}, "
            f"missing shard={missing_shards}, extra shard={extra_shards}"
        )

    records = []
    seen_names = set()
    doi_batches = {}
    for descriptor in manifest["batches"]:
        batch_id = descriptor["batch"]
        batch_path = shards_dir / descriptor["path"]
        shard_path = shards_dir / f"shard-{batch_id}.json"
        batch = load_json(batch_path)
        shard = load_json(shard_path)
        summary = _validate_shard_objects(batch, shard, batch_path, shard_path)
        for key in ("paper_count", "clock_count"):
            if summary[key] != descriptor[key]:
                raise ValueError(
                    f"batch {batch_id} {key}: manifest declares {descriptor[key]}, file has {summary[key]}"
                )
        if (
            batch["batch"] != batch_id
            or batch["paper_count"] != descriptor["paper_count"]
            or batch["clock_count"] != descriptor["clock_count"]
        ):
            raise ValueError(f"batch {batch_id}: declaration does not match batch file")
        for paper in batch["papers"]:
            prior_batch = doi_batches.setdefault(paper["doi"], batch_id)
            if prior_batch != batch_id:
                raise ValueError(f"DOI split across batches: {paper['doi']!r} appears in {prior_batch} and {batch_id}")
        for record in shard["records"]:
            name = record["clock_name"]
            if name in seen_names:
                raise ValueError(f"duplicate clock across shards: {name!r}")
            seen_names.add(name)
            records.append(record)

    records.sort(key=lambda record: record["clock_name"])
    if len(records) != manifest["clock_count"]:
        raise ValueError("manifest.clock_count: merged coverage does not match")
    if len(doi_batches) != manifest["paper_count"]:
        raise ValueError("manifest.paper_count: merged DOI coverage does not match")
    merged = {
        "schema_version": 1,
        "paper_count": len(doi_batches),
        "clock_count": len(records),
        "records": records,
    }
    _load_merged_value(merged)
    _write_json_new_atomic(output_path, merged)
    return merged


def _load_merged_value(merged):
    """Validate an in-memory merged value with the same contract as _load_merged."""
    descriptor, name = tempfile.mkstemp(prefix=".audit-merged-", suffix=".json")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(merged, stream, ensure_ascii=False)
        return _load_merged(name)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(name)


def _load_vocabulary(path):
    vocabulary = load_json(path)
    _require_exact_keys(vocabulary, {"schema_version", "array_fields", "fields"}, "vocabulary")
    validate_vocabulary(vocabulary)
    expected_fields = set(ARRAY_FIELDS) | set(CONTROLLED_SCALAR_FIELDS)
    if set(vocabulary["fields"]) != expected_fields:
        raise ValueError("vocabulary.fields: expected exactly the controlled fields")
    for field, descriptor in vocabulary["fields"].items():
        _require_exact_keys(descriptor, {"description", "values", "aliases"}, f"vocabulary.{field}")
        collisions = set(descriptor["values"]) & set(descriptor["aliases"])
        if collisions:
            raise ValueError(f"vocabulary.{field}: aliases collide with canonical values {sorted(collisions)}")
    return vocabulary


def _observations(merged, field):
    grouped = {}
    for record in merged["records"]:
        evidence = record["fields"][field]
        proposed = evidence["value"] if field in ARRAY_FIELDS else [evidence["value"]]
        for value in proposed:
            bucket = grouped.setdefault(
                value,
                {"value": value, "count": 0, "clock_names": set(), "statuses": set(), "source_terms": set()},
            )
            bucket["count"] += 1
            bucket["clock_names"].add(record["clock_name"])
            bucket["statuses"].add(evidence["status"])
            bucket["source_terms"].add(evidence["source_text"])
    result = []
    for value in sorted(grouped):
        bucket = grouped[value]
        result.append(
            {
                "value": value,
                "count": bucket["count"],
                "clock_names": sorted(bucket["clock_names"]),
                "statuses": sorted(bucket["statuses"]),
                "source_terms": sorted(bucket["source_terms"]),
            }
        )
    return result


def _candidate_key(value):
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized = re.sub(
        r"[\u00ad\u2010-\u2015\u2212\u2e3a\u2e3b\ufe58\ufe63\uff0d]",
        "-",
        normalized,
    )
    normalized = "".join(
        " "
        if character == "-"
        and index > 0
        and index + 1 < len(normalized)
        and normalized[index - 1].isalpha()
        and normalized[index + 1].isalpha()
        else character
        for index, character in enumerate(normalized)
    )
    normalized = " ".join(normalized.split())
    normalized = re.sub(r"\s*([^\w\s])\s*", r"\1", normalized, flags=re.UNICODE)
    safe_plurals = {
        "cells": "cell",
        "tissues": "tissue",
        "assays": "assay",
        "arrays": "array",
        "probes": "probe",
        "cpgs": "cpg",
        "samples": "sample",
        "adults": "adult",
        "marks": "mark",
        "scores": "score",
    }
    final_token = re.fullmatch(r"(.*?)([^\W\d_]+)([^\w]*)", normalized, flags=re.UNICODE)
    if final_token and final_token.group(2) in safe_plurals:
        normalized = final_token.group(1) + safe_plurals[final_token.group(2)] + final_token.group(3)
    return normalized


def vocabulary_report(merged_path, vocabulary_path, output_path):
    """Classify proposed controlled values without changing them."""
    output_path = Path(output_path)
    _refuse_existing([output_path])
    merged = _load_merged(merged_path)
    vocabulary = _load_vocabulary(vocabulary_path)
    report = {"schema_version": 1, "fields": {}}
    for field in (*ARRAY_FIELDS, *CONTROLLED_SCALAR_FIELDS):
        descriptor = vocabulary["fields"][field]
        canonical = set(descriptor["values"])
        aliases = descriptor["aliases"]
        observations = _observations(merged, field)
        exact_known = [item for item in observations if item["value"] in canonical]
        alias_known = []
        unknown = []
        for item in observations:
            if item["value"] in canonical:
                continue
            if item["value"] in aliases:
                alias_known.append({**item, "canonical_value": aliases[item["value"]]})
            else:
                unknown.append(item)
        groups = {}
        for item in observations:
            groups.setdefault(_candidate_key(item["value"]), []).append(item)
        candidate_groups = [
            {"normalized_key": key, "values": values} for key, values in sorted(groups.items()) if len(values) > 1
        ]
        report["fields"][field] = {
            "existing_values": descriptor["values"],
            "exact_known": exact_known,
            "alias_known": alias_known,
            "unknown_values": unknown,
            "candidate_groups": candidate_groups,
        }
    _write_json_new_atomic(output_path, report)
    return report


def normalize_merged(merged_path, vocabulary_path, output_path):
    """Apply canonical values and explicit exact aliases; never infer mappings."""
    output_path = Path(output_path)
    _refuse_existing([output_path])
    merged = copy.deepcopy(_load_merged(merged_path))
    vocabulary = _load_vocabulary(vocabulary_path)
    unknown = {}
    duplicate_errors = []
    for record in merged["records"]:
        name = record["clock_name"]
        for field in (*ARRAY_FIELDS, *CONTROLLED_SCALAR_FIELDS):
            evidence = record["fields"][field]
            descriptor = vocabulary["fields"][field]
            canonical = set(descriptor["values"])
            aliases = descriptor["aliases"]
            proposed = evidence["value"] if field in ARRAY_FIELDS else [evidence["value"]]
            normalized = []
            for value in proposed:
                if value in canonical:
                    target = value
                elif value in aliases:
                    target = aliases[value]
                else:
                    unknown.setdefault((field, value), set()).add(name)
                    target = value
                normalized.append(target)
            if field in ARRAY_FIELDS and len(normalized) != len(set(normalized)):
                duplicate_errors.append(f"{name}.{field}: duplicate canonical value after alias normalization")
            evidence["value"] = normalized if field in ARRAY_FIELDS else normalized[0]
    if unknown:
        details = "; ".join(
            f"{field}={value!r}: {', '.join(sorted(names))}" for (field, value), names in sorted(unknown.items())
        )
        raise ValueError(f"unknown controlled values: {details}")
    if duplicate_errors:
        raise ValueError("; ".join(duplicate_errors))
    _load_merged_value(merged)
    _write_json_new_atomic(output_path, merged)
    return merged


def _load_merged_snapshot(path):
    path = Path(path)
    try:
        content = path.read_bytes()
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: invalid UTF-8") from error
    merged = _parse_json(text, str(path))
    _load_merged_value(merged)
    return merged, hashlib.sha256(content).hexdigest()


def _review_snapshot(reconciled, field_decisions):
    coverage = {}
    observed_counts = {}
    canonical_counts = {}
    alias_counts = {}
    override_counts = {}
    unmapped_errors = []
    for field in (*ARRAY_FIELDS, *CONTROLLED_SCALAR_FIELDS):
        descriptor = field_decisions[field]
        canonical = set(descriptor["canonical_values"])
        aliases = descriptor["aliases"]
        overrides = descriptor["per_clock_overrides"]
        observations = {}
        for record in reconciled["records"]:
            value = record["fields"][field]["value"]
            values = value if field in ARRAY_FIELDS else [value]
            for source_value in values:
                observations.setdefault(source_value, set()).add(record["clock_name"])

        proof = []
        unmapped_values = []
        for source_value in sorted(observations):
            direct_clocks = []
            override_clocks = []
            targets = set()
            source_unmapped = False
            for clock_name in sorted(observations[source_value]):
                if clock_name in overrides:
                    override_clocks.append(clock_name)
                    targets.update(overrides[clock_name]["canonical_values"])
                    continue
                direct_clocks.append(clock_name)
                mapping_count = int(source_value in canonical) + int(source_value in aliases)
                if mapping_count != 1:
                    source_unmapped = True
                    unmapped_errors.append(
                        f"{clock_name}.{field}={source_value!r}: expected exactly one mapping, found {mapping_count}"
                    )
                else:
                    targets.add(aliases.get(source_value, source_value))
            if source_unmapped:
                unmapped_values.append(source_value)
            proof.append(
                {
                    "source_value": source_value,
                    "clock_count": len(observations[source_value]),
                    "direct_clock_count": len(direct_clocks),
                    "override_clock_count": len(override_clocks),
                    "mapping": (
                        "per-clock override"
                        if override_clocks
                        else "canonical"
                        if source_value in canonical
                        else "alias"
                    ),
                    "canonical_values": sorted(targets),
                }
            )
        coverage[field] = {
            "observed_value_count": len(observations),
            "covered_value_count": len(observations) - len(unmapped_values),
            "unmapped_values": unmapped_values,
            "canonical_value_count": len(canonical),
            "proof": proof,
        }
        observed_counts[field] = len(observations)
        canonical_counts[field] = len(canonical)
        alias_counts[field] = len(aliases)
        override_counts[field] = len(overrides)
    counts = {
        "fields": len(field_decisions),
        "observed_values": observed_counts,
        "canonical_values": canonical_counts,
        "aliases": alias_counts,
        "per_clock_overrides": override_counts,
        "unmapped_values": sum(len(descriptor["unmapped_values"]) for descriptor in coverage.values()),
    }
    return coverage, counts, unmapped_errors


def _coverage_snapshot(coverage):
    if type(coverage) is not dict:
        raise ValueError("decisions.coverage: expected an object")
    result = {}
    for field, descriptor in coverage.items():
        _require_exact_keys(
            descriptor,
            {
                "observed_value_count",
                "covered_value_count",
                "unmapped_values",
                "canonical_value_count",
                "proof",
            },
            f"decisions.coverage.{field}",
        )
        proof = descriptor["proof"]
        if type(proof) is not list:
            raise ValueError(f"decisions.coverage.{field}.proof: expected a list")
        proof_by_value = {}
        for index, item in enumerate(proof):
            context = f"decisions.coverage.{field}.proof[{index}]"
            _require_exact_keys(
                item,
                {
                    "source_value",
                    "clock_count",
                    "direct_clock_count",
                    "override_clock_count",
                    "mapping",
                    "canonical_values",
                },
                context,
            )
            source_value = item["source_value"]
            if type(source_value) is not str or not source_value.strip():
                raise ValueError(f"{context}.source_value: expected a nonempty string")
            if source_value in proof_by_value:
                raise ValueError(f"{context}.source_value: duplicate value {source_value!r}")
            canonical_values = item["canonical_values"]
            if (
                type(canonical_values) is not list
                or any(type(value) is not str or not value.strip() for value in canonical_values)
                or len(canonical_values) != len(set(canonical_values))
            ):
                raise ValueError(f"{context}.canonical_values: expected unique strings")
            proof_by_value[source_value] = {
                **item,
                "canonical_values": sorted(canonical_values),
            }
        result[field] = {
            **descriptor,
            "proof": proof_by_value,
            "unmapped_values": (
                sorted(descriptor["unmapped_values"])
                if type(descriptor["unmapped_values"]) is list
                else descriptor["unmapped_values"]
            ),
        }
    return result


def _validate_vocabulary_decisions(
    decisions,
    reconciled_path,
    reconciled_sha256,
    reconciled,
    vocabulary,
):
    controlled_fields = (*ARRAY_FIELDS, *CONTROLLED_SCALAR_FIELDS)
    _require_exact_keys(
        decisions,
        {
            "schema_version",
            "source",
            "rationale",
            "decisions",
            "ambiguities",
            "coverage",
            "counts",
        },
        "decisions",
    )
    if type(decisions["schema_version"]) is not int or decisions["schema_version"] != 1:
        raise ValueError("decisions.schema_version: expected integer 1")
    rationale = decisions["rationale"]
    if type(rationale) is not list or not rationale:
        raise ValueError("decisions.rationale: expected a nonempty list")
    if any(type(item) is not str or not item.strip() for item in rationale):
        raise ValueError("decisions.rationale: expected nonempty strings")
    if len(rationale) != len(set(rationale)):
        raise ValueError("decisions.rationale: entries must be unique")
    if type(decisions["ambiguities"]) is not list:
        raise ValueError("decisions.ambiguities: expected a list")
    if decisions["ambiguities"]:
        raise ValueError("decisions.ambiguities: unresolved ambiguities must be adjudicated first")
    source = decisions["source"]
    _require_exact_keys(
        source,
        {"reconciled", "sha256", "paper_count", "clock_count"},
        "decisions.source",
    )
    if type(source["reconciled"]) is not str or not source["reconciled"].strip():
        raise ValueError("decisions.source.reconciled: expected a nonempty path")
    if Path(source["reconciled"]).resolve() != Path(reconciled_path).resolve():
        raise ValueError("decisions.source.reconciled: does not match the reconciled input")
    if type(source["sha256"]) is not str or re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) is None:
        raise ValueError("decisions.source.sha256: expected a lowercase SHA-256 digest")
    if source["sha256"] != reconciled_sha256:
        raise ValueError("decisions.source.sha256: does not match the reconciled input")
    for count_name in ("paper_count", "clock_count"):
        if type(source[count_name]) is not int or source[count_name] != reconciled[count_name]:
            raise ValueError(f"decisions.source.{count_name}: does not match the reconciled input")

    field_decisions = decisions["decisions"]
    if type(field_decisions) is not dict or set(field_decisions) != set(controlled_fields):
        raise ValueError("decisions.decisions: expected exactly the controlled fields")
    records_by_name = {record["clock_name"]: record for record in reconciled["records"]}
    validated = {}
    for field in controlled_fields:
        context = f"decisions.{field}"
        descriptor = field_decisions[field]
        _require_exact_keys(
            descriptor,
            {"canonical_values", "aliases", "per_clock_overrides"},
            context,
        )
        canonical_values = descriptor["canonical_values"]
        if (
            type(canonical_values) is not list
            or not canonical_values
            or any(type(value) is not str or not value.strip() for value in canonical_values)
            or len(canonical_values) != len(set(canonical_values))
        ):
            raise ValueError(f"{context}.canonical_values: expected unique nonempty strings")
        canonical = set(canonical_values)
        aliases = descriptor["aliases"]
        if type(aliases) is not dict:
            raise ValueError(f"{context}.aliases: expected an object")
        for alias, target in aliases.items():
            if type(alias) is not str or not alias.strip():
                raise ValueError(f"{context}.aliases: expected nonempty alias strings")
            if alias in canonical:
                raise ValueError(f"{context}: canonical value {alias!r} has more than exactly one mapping")
            if type(target) is not str or target not in canonical:
                raise ValueError(f"{context}.aliases: alias {alias!r} must reference a canonical value")

        vocabulary_descriptor = vocabulary["fields"][field]
        if vocabulary_descriptor["values"] != canonical_values or vocabulary_descriptor["aliases"] != aliases:
            raise ValueError(f"vocabulary.{field}: does not match the vocabulary decisions")

        overrides = descriptor["per_clock_overrides"]
        if type(overrides) is not dict:
            raise ValueError(f"{context}.per_clock_overrides: expected an object")
        if overrides and field not in ARRAY_FIELDS:
            raise ValueError(f"{context}.per_clock_overrides: cannot override a scalar field")
        for clock_name, override in overrides.items():
            override_context = f"{context}.per_clock_overrides.{clock_name}"
            if clock_name not in records_by_name:
                raise ValueError(f"{override_context}: references an unknown clock")
            _require_exact_keys(
                override,
                {"expected_value", "canonical_values"},
                override_context,
            )
            expected_value = override["expected_value"]
            validate_audited_value(field, expected_value, f"{override_context}.expected_value")
            current = records_by_name[clock_name]["fields"][field]["value"]
            if expected_value != current:
                raise ValueError(f"{override_context}.expected_value: does not match the current reviewed value")
            targets = override["canonical_values"]
            if (
                type(targets) is not list
                or not targets
                or any(type(target) is not str or not target.strip() for target in targets)
            ):
                raise ValueError(f"{override_context}: expected nonempty canonical strings")
            if len(targets) != len(set(targets)):
                raise ValueError(f"{override_context}: target values must be unique")
            unknown_targets = sorted(set(targets) - canonical)
            if unknown_targets:
                raise ValueError(f"{override_context}: targets are not canonical: {unknown_targets}")
        validated[field] = {
            "canonical": canonical,
            "aliases": aliases,
            "overrides": {clock_name: override["canonical_values"] for clock_name, override in overrides.items()},
        }
    expected_coverage, expected_counts, unmapped = _review_snapshot(
        reconciled,
        field_decisions,
    )
    if unmapped:
        raise ValueError("observed controlled value has no mapping: " + "; ".join(unmapped))
    if _coverage_snapshot(decisions["coverage"]) != _coverage_snapshot(expected_coverage):
        raise ValueError("decisions.coverage: does not match the reconciled review snapshot")
    if decisions["counts"] != expected_counts:
        raise ValueError("decisions.counts: does not match the reconciled review snapshot")
    return validated


def apply_vocabulary_decisions(reconciled_path, decisions_path, vocabulary_path, output_path):
    """Apply an exhaustive reviewed mapping artifact without changing its evidence."""
    output_path = Path(output_path)
    _refuse_existing([output_path])
    reconciled, reconciled_sha256 = _load_merged_snapshot(reconciled_path)
    decisions = load_json(decisions_path)
    vocabulary = _load_vocabulary(vocabulary_path)
    mappings = _validate_vocabulary_decisions(
        decisions,
        reconciled_path,
        reconciled_sha256,
        reconciled,
        vocabulary,
    )
    normalized = copy.deepcopy(reconciled)
    unmapped = []
    for record in normalized["records"]:
        clock_name = record["clock_name"]
        for field in (*ARRAY_FIELDS, *CONTROLLED_SCALAR_FIELDS):
            mapping = mappings[field]
            evidence = record["fields"][field]
            if clock_name in mapping["overrides"]:
                evidence["value"] = list(mapping["overrides"][clock_name])
                continue
            proposed = evidence["value"] if field in ARRAY_FIELDS else [evidence["value"]]
            targets = []
            for source_value in proposed:
                direct_mappings = int(source_value in mapping["canonical"]) + int(source_value in mapping["aliases"])
                if direct_mappings != 1:
                    unmapped.append(
                        f"{clock_name}.{field}={source_value!r}: expected exactly one mapping, found {direct_mappings}"
                    )
                    continue
                targets.append(mapping["aliases"].get(source_value, source_value))
            if field in ARRAY_FIELDS and len(targets) != len(set(targets)):
                raise ValueError(f"{clock_name}.{field}: mappings produce duplicate canonical values")
            if field in ARRAY_FIELDS:
                evidence["value"] = targets
            elif targets:
                evidence["value"] = targets[0]
    if unmapped:
        raise ValueError("observed controlled value has no mapping: " + "; ".join(unmapped))
    _load_merged_value(normalized)
    _write_json_new_atomic(output_path, normalized)
    return normalized


def _json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _ledger_bytes(records):
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=False, separators=(",", ":")) + "\n" for record in records
    ).encode("utf-8")


def _report_text(records, current):
    statuses = Counter(evidence["status"] for record in records for evidence in record["fields"].values())
    changes = Counter()
    access_issues = []
    for record in records:
        name = record["clock_name"]
        for field, evidence in record["fields"].items():
            if current[name].get(field) != evidence["value"]:
                changes[field] += 1
        access_issues.extend(f"- {name}: {issue}" for issue in record["access_issues"])
    lines = ["# Clock Metadata Audit Materialization", "", "## Evidence status counts", ""]
    lines.extend(f"- {status}: {statuses[status]}" for status in sorted(EVIDENCE_STATUSES))
    lines.extend(["", "## Access issues", ""])
    lines.extend(access_issues or ["- None"])
    lines.extend(["", "## Field change counts", ""])
    lines.extend(f"- {field}: {changes[field]}" for field in AUDITED_FIELDS)
    return "\n".join(lines) + "\n"


def _write_bytes_atomic(path, content):
    path = Path(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
        try:
            os.link(temporary_name, path)
        except FileExistsError as error:
            raise ValueError(f"output path already exists: {path}") from error
        os.unlink(temporary_name)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def _write_json_new_atomic(path, value):
    _write_bytes_atomic(path, _json_bytes(value))


def _fsync_directory(parent):
    descriptor = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _acquire_materialization_lock(parent):
    lock_path = parent / ".clock-metadata-materialize.lock"
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError(f"materialization is locked by {lock_path}") from error
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        os.fsync(descriptor)
    except Exception:
        os.close(descriptor)
        with suppress(FileNotFoundError):
            lock_path.unlink()
        raise
    os.close(descriptor)
    return lock_path


def _stage_materialization_bytes(parent, target, content, index):
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".clock-metadata-stage-{index}-",
        dir=parent,
    )
    path = Path(temporary_name)
    try:
        mode = target.stat().st_mode & 0o7777 if target.exists() else 0o644
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        with suppress(OSError):
            os.close(descriptor)
        with suppress(FileNotFoundError):
            path.unlink()
        raise
    return path


def _backup_materialization_target(parent, target, index):
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".clock-metadata-backup-{index}-",
        dir=parent,
    )
    os.close(descriptor)
    backup = Path(temporary_name)
    backup.unlink()
    try:
        os.link(target, backup)
    except Exception:
        with suppress(FileNotFoundError):
            backup.unlink()
        raise
    return backup


def _publish_replace(source, target):
    os.replace(source, target)


def _post_validate_materialization(targets, expected_bytes):
    if len(targets) != len(expected_bytes):
        raise ValueError("materialization target/content count mismatch")
    for target, expected in zip(targets, expected_bytes):
        if target.read_bytes() != expected:
            raise ValueError(f"post-publish validation failed for {target}")


def _cleanup_materialization_backup(backup):
    backup.unlink()


class _CommittedMaterializationError(ValueError):
    """A cleanup failure after all canonical targets crossed the commit boundary."""


def _transactional_install_materialization(parent, targets, contents):
    if len(targets) != len(contents):
        raise ValueError("materialization target/content count mismatch")
    stages = []
    backups = {}
    originally_missing = set()
    published = []
    rollback_failed_backups = set()
    committed = False
    try:
        for index, (target, content) in enumerate(zip(targets, contents), start=1):
            stages.append(_stage_materialization_bytes(parent, target, content, index))
        for index, target in enumerate(targets, start=1):
            if target.exists():
                backups[target] = _backup_materialization_target(parent, target, index)
            else:
                originally_missing.add(target)
        # Make recovery links durable before publishing the first canonical target.
        _fsync_directory(parent)
        for stage, target in zip(stages, targets):
            _publish_replace(stage, target)
            published.append(target)
        _fsync_directory(parent)
        _post_validate_materialization(targets, contents)
        committed = True

        cleanup_errors = []
        recovery_artifacts = []
        for backup in backups.values():
            try:
                _cleanup_materialization_backup(backup)
            except Exception as cleanup_error:
                cleanup_errors.append(f"{backup}: {cleanup_error}")
                recovery_artifacts.append(str(backup))
        try:
            _fsync_directory(parent)
        except Exception as cleanup_error:
            cleanup_errors.append(f"directory fsync: {cleanup_error}")
        if cleanup_errors:
            message = "materialization targets committed; post-commit cleanup failed: " + "; ".join(cleanup_errors)
            if recovery_artifacts:
                message += (
                    f"; recovery artifact(s) retained: {', '.join(recovery_artifacts)}; "
                    "remove manually after verifying canonical targets"
                )
            raise _CommittedMaterializationError(message)
    except Exception as primary_error:
        if committed:
            if isinstance(primary_error, _CommittedMaterializationError):
                raise
            raise _CommittedMaterializationError(
                f"materialization targets committed; post-commit cleanup failed: {primary_error}"
            ) from primary_error
        rollback_errors = []
        for target in reversed(published):
            try:
                if target in originally_missing:
                    with suppress(FileNotFoundError):
                        target.unlink()
                else:
                    os.replace(backups[target], target)
            except Exception as rollback_error:
                rollback_errors.append(f"{target}: {rollback_error}")
                if target in backups:
                    rollback_failed_backups.add(backups[target])
        try:
            _fsync_directory(parent)
        except Exception as rollback_error:
            rollback_errors.append(f"directory fsync: {rollback_error}")
        message = f"materialization transaction failed: {primary_error}"
        if rollback_errors:
            message += f"; rollback failure: {'; '.join(rollback_errors)}"
        raise ValueError(message) from primary_error
    finally:
        for stage in stages:
            with suppress(FileNotFoundError):
                stage.unlink()
        if not committed:
            for backup in backups.values():
                if backup not in rollback_failed_backups:
                    with suppress(FileNotFoundError):
                        backup.unlink()


def _remove_materialization_lock(lock_path):
    lock_path.unlink()


def _fsync_materialization_lock_removal(parent):
    _fsync_directory(parent)


def materialize(
    normalized_path,
    current_path,
    vocabulary_path,
    registry_path,
    ledger_path,
    report_path,
):
    """Preflight and materialize registry/ledger/report, with report written last."""
    requested_targets = [Path(registry_path), Path(ledger_path), Path(report_path)]
    targets = [path.resolve(strict=False) for path in requested_targets]
    if len(set(targets)) != len(targets):
        raise ValueError("registry, ledger, and report output paths must resolve to three distinct paths")
    parents = {target.parent for target in targets}
    if len(parents) != 1:
        raise ValueError("registry, ledger, and report output paths must share the same parent directory")
    parent = parents.pop()
    if not parent.is_dir():
        raise ValueError(f"materialization output parent is not an existing directory: {parent}")
    lock_path = _acquire_materialization_lock(parent)
    transaction_error = None
    targets_committed = False
    result = None
    try:
        merged = _load_merged(normalized_path)
        vocabulary = _load_vocabulary(vocabulary_path)
        current = load_json(current_path)
        if type(current) is not dict:
            raise ValueError("current registry: expected an object")
        names = [record["clock_name"] for record in merged["records"]]
        if set(current) != set(names):
            raise ValueError(
                f"current clock set mismatch: missing={sorted(set(names) - set(current))}, "
                f"extra={sorted(set(current) - set(names))}"
            )
        unresolved = []
        registry = {}
        for record in merged["records"]:
            name = record["clock_name"]
            current_record = current[name]
            if type(current_record) is not dict:
                raise ValueError(f"{name}: current registry record must be an object")
            for field, evidence in record["fields"].items():
                if evidence["status"] == "unresolved":
                    unresolved.append(f"{name}.{field}")
            rebuilt = copy.deepcopy(current_record)
            for field in AUDITED_FIELDS:
                rebuilt[field] = copy.deepcopy(record["fields"][field]["value"])
            rebuilt["clock_name"] = name
            registry[name] = rebuilt
        if unresolved:
            raise ValueError(f"unresolved evidence: {', '.join(unresolved)}")
        registry = {name: registry[name] for name in sorted(registry)}
        validate_registry(registry, vocabulary)
        contents = [
            _json_bytes(registry),
            _ledger_bytes(merged["records"]),
            _report_text(merged["records"], current).encode("utf-8"),
        ]

        # All input reads, validation, and byte construction precede target mutation.
        _transactional_install_materialization(parent, targets, contents)
        targets_committed = True
        result = {
            "paper_count": merged["paper_count"],
            "clock_count": merged["clock_count"],
            "registry": str(targets[0]),
            "ledger": str(targets[1]),
            "report": str(targets[2]),
        }
    except _CommittedMaterializationError as error:
        transaction_error = error
        targets_committed = True
    except Exception as error:
        transaction_error = error
    try:
        _remove_materialization_lock(lock_path)
        _fsync_materialization_lock_removal(parent)
    except Exception as lock_error:
        stale_lock = lock_path.exists()
        if targets_committed:
            message = f"materialization targets committed; lock cleanup failure: {lock_error}"
            if stale_lock:
                message += f"; stale lock retained at {lock_path}; remove manually after verification"
            if transaction_error is not None:
                message = f"{transaction_error}; {message}"
            raise ValueError(message) from lock_error
        if transaction_error is not None:
            raise ValueError(f"{transaction_error}; lock cleanup failure: {lock_error}") from transaction_error
        raise ValueError(f"materialization lock cleanup failed: {lock_error}") from lock_error
    if transaction_error is not None:
        raise transaction_error
    return result


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
    merge = subparsers.add_parser("merge-shards")
    merge.add_argument("--manifest", required=True)
    merge.add_argument("--shards", required=True)
    merge.add_argument("--output", required=True)
    vocabulary = subparsers.add_parser("vocabulary-report")
    vocabulary.add_argument("--merged", required=True)
    vocabulary.add_argument("--vocabulary", required=True)
    vocabulary.add_argument("--output", required=True)
    normalize = subparsers.add_parser("normalize-merged")
    normalize.add_argument("--merged", required=True)
    normalize.add_argument("--vocabulary", required=True)
    normalize.add_argument("--output", required=True)
    apply_decisions = subparsers.add_parser("apply-vocabulary-decisions")
    apply_decisions.add_argument("--reconciled", required=True)
    apply_decisions.add_argument("--decisions", required=True)
    apply_decisions.add_argument("--vocabulary", required=True)
    apply_decisions.add_argument("--output", required=True)
    materialization = subparsers.add_parser("materialize")
    materialization.add_argument("--normalized", required=True)
    materialization.add_argument("--current", required=True)
    materialization.add_argument("--vocabulary", required=True)
    materialization.add_argument("--registry", required=True)
    materialization.add_argument("--ledger", required=True)
    materialization.add_argument("--report", required=True)
    return parser


def main(argv=None):
    try:
        arguments = _parser().parse_args(argv)
        if arguments.command == "build-manifest":
            manifest = build_manifest(arguments.registry, arguments.output_dir, arguments.batches)
            print(
                f"built {len(manifest['batches'])} batches for "
                f"{manifest['paper_count']} papers and {manifest['clock_count']} clocks"
            )
        elif arguments.command == "validate-shard":
            summary = validate_shard(arguments.batch, arguments.shard)
            print(
                f"validated batch {summary['batch']}: {summary['paper_count']} papers, {summary['clock_count']} clocks"
            )
        elif arguments.command == "merge-shards":
            merged = merge_shards(arguments.manifest, arguments.shards, arguments.output)
            print(f"merged {merged['clock_count']} clocks across {merged['paper_count']} papers")
        elif arguments.command == "vocabulary-report":
            report = vocabulary_report(arguments.merged, arguments.vocabulary, arguments.output)
            print(f"reported vocabulary proposals for {len(report['fields'])} fields")
        elif arguments.command == "normalize-merged":
            merged = normalize_merged(arguments.merged, arguments.vocabulary, arguments.output)
            print(f"normalized {merged['clock_count']} clocks")
        elif arguments.command == "apply-vocabulary-decisions":
            normalized = apply_vocabulary_decisions(
                arguments.reconciled,
                arguments.decisions,
                arguments.vocabulary,
                arguments.output,
            )
            print(f"normalized {normalized['clock_count']} clocks from reviewed vocabulary decisions")
        else:
            summary = materialize(
                arguments.normalized,
                arguments.current,
                arguments.vocabulary,
                arguments.registry,
                arguments.ledger,
                arguments.report,
            )
            print(f"materialized {summary['clock_count']} clocks")
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
