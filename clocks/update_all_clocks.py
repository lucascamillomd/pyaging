#!/usr/bin/env python3
import argparse
import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path

import torch

RUNTIME_METADATA_FIELDS = (
    "version",
    "preprocess",
    "postprocess",
    "reference_values",
)
CURATED_REGISTRY_FIELDS = (
    "clock_name",
    "data_type",
    "species",
    "year",
    "approved_by_author",
    "citation",
    "doi",
    "notes",
    "research_only",
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
    "citations",
    "citations_date",
)
_REGISTRY_ARRAY_FIELDS = (
    "tissue",
    "predicts",
    "training_target",
    "unit",
    "platform",
)
_REGISTRY_INTEGER_FIELDS = ("year", "n_features", "citations")
_REGISTRY_STRING_FIELDS = (
    "clock_name",
    "data_type",
    "species",
    "approved_by_author",
    "doi",
    "notes",
    "model_type",
    "population",
    "journal",
    "last_author",
    "citations_date",
)


def merge_clock_metadata(generated_metadata, curated_metadata):
    """Merge generated runtime data into registry-backed curated metadata."""
    merged_metadata = {}

    for clock_name, curated_entry in curated_metadata.items():
        generated_entry = generated_metadata[clock_name]
        merged_entry = {key: value for key, value in curated_entry.items() if key not in RUNTIME_METADATA_FIELDS}
        merged_entry.update({key: generated_entry[key] for key in RUNTIME_METADATA_FIELDS if key in generated_entry})
        merged_metadata[clock_name] = merged_entry

    return merged_metadata


def _reject_non_finite(value):
    raise ValueError(f"non-finite JSON number {value!r} is not allowed")


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r} is not allowed")
        result[key] = value
    return result


def _validate_registry_entry(clock_name, entry):
    if not isinstance(entry, dict):
        raise ValueError(f"Curated metadata entry for '{clock_name}' must be a dictionary")
    entry_fields = list(entry)
    curated_count = len(CURATED_REGISTRY_FIELDS)
    if entry_fields[:curated_count] != list(CURATED_REGISTRY_FIELDS):
        raise ValueError(
            f"Curated metadata entry for '{clock_name}' must use canonical field "
            f"order {list(CURATED_REGISTRY_FIELDS)!r}"
        )
    optional_runtime_fields = entry_fields[curated_count:]
    if any(field not in RUNTIME_METADATA_FIELDS for field in optional_runtime_fields):
        raise ValueError(
            f"Curated metadata entry for '{clock_name}' contains unsupported fields after the canonical curated fields"
        )
    if entry["clock_name"] != clock_name:
        raise ValueError(f"Curated metadata entry clock_name for '{clock_name}' must match its key")

    for field in _REGISTRY_STRING_FIELDS:
        if not isinstance(entry[field], str) or not entry[field].strip():
            raise ValueError(f"{clock_name}.{field} must be a non-empty string")
    for field in _REGISTRY_INTEGER_FIELDS:
        if type(entry[field]) is not int:
            raise ValueError(f"{clock_name}.{field} must be an integer")
    for field in _REGISTRY_ARRAY_FIELDS:
        value = entry[field]
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(item, str) or not item.strip() for item in value)
            or len(value) != len(set(value))
        ):
            raise ValueError(f"{clock_name}.{field} must be a non-empty string array with unique values")

    citation = entry["citation"]
    if isinstance(citation, str):
        valid_citation = bool(citation.strip())
    elif isinstance(citation, list):
        valid_citation = (
            bool(citation)
            and all(isinstance(item, str) and item.strip() for item in citation)
            and len(citation) == len(set(citation))
        )
    else:
        valid_citation = False
    if not valid_citation:
        raise ValueError(f"{clock_name}.citation must be a non-empty string or string array")

    research_only = entry["research_only"]
    if research_only is not None and type(research_only) is not bool:
        raise ValueError(f"{clock_name}.research_only must be a boolean or null")

    for field in ("version", "preprocess", "postprocess"):
        if field in entry and (not isinstance(entry[field], str) or not entry[field].strip()):
            raise ValueError(f"{clock_name}.{field} must be a non-empty string")
    if "reference_values" in entry and type(entry["reference_values"]) is not bool:
        raise ValueError(f"{clock_name}.reference_values must be a boolean")


def load_curated_metadata(registry_path):
    """Load and validate the canonical curated JSON metadata registry."""
    registry_path = Path(registry_path)
    if not registry_path.is_file():
        raise FileNotFoundError(f"Curated metadata registry is required: {registry_path}")

    try:
        with registry_path.open(encoding="utf-8") as registry_file:
            metadata = json.load(
                registry_file,
                parse_constant=_reject_non_finite,
                object_pairs_hook=_reject_duplicate_keys,
            )
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid curated metadata JSON at column {error.colno}: {error.msg}") from error
    except ValueError as error:
        raise ValueError(f"Invalid curated metadata JSON: {error}") from error
    if not isinstance(metadata, dict):
        raise ValueError("Curated metadata must be a top-level dictionary")
    if list(metadata) != sorted(metadata):
        raise ValueError("Curated metadata registry clock keys must be in alphabetical order")

    for clock_name, entry in metadata.items():
        if not isinstance(clock_name, str) or not clock_name or clock_name != clock_name.lower():
            raise ValueError("Every curated metadata key must be a lowercase non-empty string")
        _validate_registry_entry(clock_name, entry)

    return metadata


def _generated_metadata_entry(clock):
    if not isinstance(clock.metadata, dict):
        raise ValueError("Clock metadata must be a dictionary")

    key = clock.metadata.get("clock_name")
    if not isinstance(key, str) or not key:
        raise ValueError("Clock metadata must contain a non-empty string clock_name")

    runtime_values = {
        "version": clock.version,
        "preprocess": clock.preprocess_name,
        "postprocess": clock.postprocess_name,
        "reference_values": (True if clock.reference_values is not None else None),
    }
    file_data = {field: runtime_values[field] for field in RUNTIME_METADATA_FIELDS if runtime_values[field] is not None}
    return key, file_data


def preflight_weight_files(weights_dir):
    """Load and validate every weight once before any rewrite."""
    if not weights_dir.is_dir():
        raise ValueError(f"A non-empty weights directory is required: {weights_dir}")

    weight_paths = sorted(weights_dir.glob("*.pt"))
    if not weight_paths:
        raise ValueError(f"A non-empty weights directory is required: {weights_dir}")

    loaded_weights = []
    clock_names = set()
    for weight_path in weight_paths:
        clock = torch.load(weight_path, weights_only=False)
        clock_name, _ = _generated_metadata_entry(clock)
        loaded_weights.append((weight_path, clock, clock_name))
        clock_names.add(clock_name)

    expected_clock_names = {weight_path.stem for weight_path in weight_paths}
    if len(clock_names) != len(weight_paths) or clock_names != expected_clock_names:
        raise ValueError(
            "Generated clock names do not match weight filenames: "
            f"expected {sorted(expected_clock_names)}, "
            f"generated {sorted(clock_names)}"
        )

    return loaded_weights, clock_names


def _prepare_weight_updates(version, loaded_weights):
    generated_metadata = {}
    updated_weights = []

    for weight_path, clock, preflight_name in loaded_weights:
        clock.version = version
        clock_name, runtime_metadata = _generated_metadata_entry(clock)
        if clock_name != preflight_name:
            raise ValueError(
                f"Generated clock names changed after preflight: expected {preflight_name!r}, generated {clock_name!r}"
            )
        if clock_name in generated_metadata:
            raise ValueError(f"Generated clock name {clock_name!r} is duplicated")
        generated_metadata[clock_name] = runtime_metadata
        updated_weights.append((weight_path, clock))

    return updated_weights, generated_metadata


def _fsync_directories(paths):
    parents = sorted({path.parent.resolve() for path in paths})
    for parent in parents:
        descriptor = os.open(str(parent), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _stage_torch_object(target, value, index):
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".clock-metadata-stage-{index}-",
        dir=str(target.parent),
    )
    os.close(descriptor)
    stage = Path(temporary_name)
    try:
        with stage.open("wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        mode = target.stat().st_mode & 0o7777 if target.exists() else 0o644
        os.chmod(stage, mode)
    except Exception:
        with suppress(FileNotFoundError):
            stage.unlink()
        raise
    return stage


def _backup_target(target, index):
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".clock-metadata-backup-{index}-",
        dir=str(target.parent),
    )
    os.close(descriptor)
    backup = Path(temporary_name)
    backup.unlink()
    try:
        os.link(str(target), str(backup))
    except Exception:
        with suppress(FileNotFoundError):
            backup.unlink()
        raise
    return backup


def _publish_replace(source, target):
    os.replace(str(source), str(target))


def _cleanup_backup(backup):
    backup.unlink()


class _CommittedRegenerationError(ValueError):
    """Cleanup failed after every canonical target crossed the commit boundary."""


def _transactional_publish(target_values):
    targets = [target for target, _value in target_values]
    if len(set(path.resolve() for path in targets)) != len(targets):
        raise ValueError("Regeneration transaction targets must be distinct")

    stages = []
    expected_bytes = {}
    backups = {}
    originally_missing = set()
    published = []
    rollback_failed_backups = set()
    committed = False
    try:
        for index, (target, value) in enumerate(target_values, start=1):
            stage = _stage_torch_object(target, value, index)
            stages.append(stage)
            expected_bytes[target] = stage.read_bytes()

        for index, target in enumerate(targets, start=1):
            if target.exists():
                backups[target] = _backup_target(target, index)
            else:
                originally_missing.add(target)
        _fsync_directories(targets)

        for stage, target in zip(stages, targets):
            _publish_replace(stage, target)
            published.append(target)
        _fsync_directories(targets)

        for target in targets:
            if target.read_bytes() != expected_bytes[target]:
                raise ValueError(f"Post-publish validation failed for {target}")
        committed = True

        cleanup_errors = []
        recovery_artifacts = []
        for backup in backups.values():
            try:
                _cleanup_backup(backup)
            except Exception as cleanup_error:
                cleanup_errors.append(f"{backup}: {cleanup_error}")
                recovery_artifacts.append(str(backup))
        try:
            _fsync_directories(targets)
        except Exception as cleanup_error:
            cleanup_errors.append(f"directory fsync: {cleanup_error}")
        if cleanup_errors:
            message = "regeneration targets committed; post-commit cleanup failed: " + "; ".join(cleanup_errors)
            if recovery_artifacts:
                message += "; recovery artifact(s) retained: " + ", ".join(recovery_artifacts)
            raise _CommittedRegenerationError(message)
    except Exception as primary_error:
        if committed:
            if isinstance(primary_error, _CommittedRegenerationError):
                raise
            raise _CommittedRegenerationError(
                f"regeneration targets committed; post-commit cleanup failed: {primary_error}"
            ) from primary_error

        rollback_errors = []
        for target in reversed(published):
            try:
                if target in originally_missing:
                    with suppress(FileNotFoundError):
                        target.unlink()
                else:
                    os.replace(str(backups[target]), str(target))
            except Exception as rollback_error:
                rollback_errors.append(f"{target}: {rollback_error}")
                if target in backups:
                    rollback_failed_backups.add(backups[target])
        try:
            _fsync_directories(targets)
        except Exception as rollback_error:
            rollback_errors.append(f"directory fsync: {rollback_error}")
        message = f"regeneration transaction failed: {primary_error}"
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


def regenerate_clock_metadata(
    version,
    weights_dir=Path("weights"),
    registry_path=Path("metadata/clock_metadata.json"),
    metadata_path=Path("metadata/all_clock_metadata.pt"),
):
    """Regenerate all weights and metadata as one registry-backed transaction."""
    weights_dir = Path(weights_dir)
    registry_path = Path(registry_path)
    metadata_path = Path(metadata_path)
    curated_dictionary = load_curated_metadata(registry_path)

    loaded_weights, validated_clock_names = preflight_weight_files(weights_dir)
    registry_clock_names = set(curated_dictionary)
    if validated_clock_names != registry_clock_names:
        raise ValueError(
            "Registry and weight clock names must match exactly: "
            f"registry {sorted(registry_clock_names)}, "
            f"weights {sorted(validated_clock_names)}"
        )

    updated_weights, generated_dictionary = _prepare_weight_updates(version, loaded_weights)
    generated_clock_names = set(generated_dictionary)
    if len(generated_dictionary) != len(loaded_weights) or generated_clock_names != validated_clock_names:
        raise ValueError(
            "Generated clock names changed after preflight: "
            f"expected {sorted(validated_clock_names)}, "
            f"generated {sorted(generated_clock_names)}"
        )

    combined_dictionary = merge_clock_metadata(generated_dictionary, curated_dictionary)
    if set(combined_dictionary) != registry_clock_names:
        raise ValueError("Merged aggregate clock names do not match registry")

    target_values = updated_weights + [(metadata_path, combined_dictionary)]
    _transactional_publish(target_values)
    return combined_dictionary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge PT files metadata.")
    parser.add_argument("version", type=str, help="Version number to be added to the metadata.")
    args = parser.parse_args()

    regenerate_clock_metadata(args.version)
    print("Metadata dictionary saved to 'metadata/all_clock_metadata.pt'.")
