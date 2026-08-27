#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import namedtuple
from contextlib import suppress
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_metadata import (  # noqa: E402
    RUNTIME_METADATA_FIELDS,
    _reject_duplicate_keys,
    _reject_non_finite,
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
_CLOCKS_DIR = Path(__file__).resolve().parent
_DEFAULT_WEIGHTS_DIR = _CLOCKS_DIR / "weights"
_DEFAULT_REGISTRY_PATH = _CLOCKS_DIR / "metadata" / "clock_metadata.json"
_DEFAULT_METADATA_PATH = _CLOCKS_DIR / "metadata" / "all_clock_metadata.pt"
_TargetRecord = namedtuple(
    "_TargetRecord",
    ("logical_path", "target_path", "logical_state", "target_state"),
)
_StagedTarget = namedtuple(
    "_StagedTarget",
    ("target_record", "stage_path", "sha256", "size"),
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


def _generated_clock_name(clock):
    """Return the clock_name recorded in a loaded clock's metadata."""
    if not isinstance(clock.metadata, dict):
        raise ValueError("Clock metadata must be a dictionary")

    key = clock.metadata.get("clock_name")
    if not isinstance(key, str) or not key:
        raise ValueError("Clock metadata must contain a non-empty string clock_name")
    return key


def _generated_metadata_entry(clock, version):
    """Build the runtime metadata entry for one clock at a pyaging release version.

    Parameters
    ----------
    clock : pyaging.models.pyagingModel
        Clock loaded from its built weight file.
    version : str
        The pyaging release version being stamped onto every clock.

    Returns
    -------
    tuple of (str, dict)
        The clock name and its non-null runtime metadata fields, in the
        canonical ``RUNTIME_METADATA_FIELDS`` order.
    """
    key = _generated_clock_name(clock)

    runtime_values = {
        "version": version,
        "preprocess": clock.preprocess_name,
        "postprocess": clock.postprocess_name,
        "reference_values": (True if clock.reference_values is not None else None),
    }
    file_data = {field: runtime_values[field] for field in RUNTIME_METADATA_FIELDS if runtime_values[field] is not None}
    return key, file_data


def _logical_path_state(path):
    try:
        stat_result = path.lstat()
    except FileNotFoundError:
        return None
    link_target = os.readlink(str(path)) if path.is_symlink() else None
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_mode,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        link_target,
    )


def _target_path_state(path):
    try:
        stat_result = path.stat()
    except FileNotFoundError:
        return None
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_mode,
        stat_result.st_size,
        stat_result.st_mtime_ns,
    )


def _resolve_target(logical_path, require_existing):
    if logical_path.is_symlink():
        try:
            target_path = logical_path.resolve(strict=True)
        except FileNotFoundError as error:
            raise ValueError(f"Refusing dangling symbolic link: {logical_path}") from error
    elif logical_path.exists():
        if require_existing and not logical_path.is_file():
            raise ValueError(f"Weight path must be a file: {logical_path}")
        target_path = logical_path.resolve(strict=True)
    elif require_existing:
        raise ValueError(f"Weight path does not exist: {logical_path}")
    else:
        target_path = logical_path.absolute()

    return _TargetRecord(
        logical_path=logical_path,
        target_path=target_path,
        logical_state=_logical_path_state(logical_path),
        target_state=_target_path_state(target_path),
    )


def preflight_weight_files(weights_dir, registry_clock_names):
    """Validate registry filenames and resolve backing targets without loading."""
    if not weights_dir.is_dir():
        raise ValueError(f"A non-empty weights directory is required: {weights_dir}")

    weight_paths = sorted(weights_dir.glob("*.pt"))
    if not weight_paths:
        raise ValueError(f"A non-empty weights directory is required: {weights_dir}")

    weight_clock_names = {weight_path.stem for weight_path in weight_paths}
    if weight_clock_names != registry_clock_names:
        raise ValueError(
            "Registry and weight clock names must match exactly: "
            f"registry {sorted(registry_clock_names)}, "
            f"weights {sorted(weight_clock_names)}"
        )

    records = [_resolve_target(weight_path, require_existing=True) for weight_path in weight_paths]
    resolved_targets = [record.target_path for record in records]
    if len(set(resolved_targets)) != len(resolved_targets):
        raise ValueError("Distinct weight paths must not resolve to the same backing target")
    return records


def _stream_sha256_and_size(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _fsync_directories(paths):
    parents = sorted({path.parent.resolve() for path in paths})
    for parent in parents:
        descriptor = os.open(str(parent), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _stage_torch_object(target, value, index):
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
        digest, size = _stream_sha256_and_size(stage)
    except Exception:
        with suppress(FileNotFoundError):
            stage.unlink()
        raise
    return stage, digest, size


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
    """Cleanup failed after a target crossed its commit boundary."""


def _publish_staged_target(staged, index):
    """Swap one staged file over its target, rolling back if the swap fails.

    A hard-link backup of the target is taken, ``os.replace`` makes the staged
    bytes visible in a single step, and the published bytes are re-hashed
    before the backup is dropped. Anything that fails up to and including that
    hash check restores the original bytes; anything that fails after it leaves
    the new bytes in place.

    Parameters
    ----------
    staged : _StagedTarget
        Staged replacement produced by ``_stage_torch_object``.
    index : int
        Position in the run, used only to name temporary files.

    Returns
    -------
    tuple of (list of str, list of str)
        Descriptions of post-commit cleanup failures, and the paths of any
        backups they left behind. Both are empty on a clean publish.

    Raises
    ------
    ValueError
        The target did not commit and holds its original bytes again.
    """
    record = staged.target_record
    target = record.target_path
    backup = None
    published = False
    committed = False
    retain_backup = False
    try:
        if _logical_path_state(record.logical_path) != record.logical_state:
            raise ValueError(f"Logical path changed during staging: {record.logical_path}")
        if _target_path_state(target) != record.target_state:
            raise ValueError(f"Backing target changed during staging: {target}")

        if record.target_state is not None:
            backup = _backup_target(target, index)
        _fsync_directories([target])

        _publish_replace(staged.stage_path, target)
        published = True
        _fsync_directories([target])

        digest, size = _stream_sha256_and_size(target)
        if digest != staged.sha256 or size != staged.size:
            raise ValueError(f"Post-publish validation failed for {target}")
        committed = True
    except Exception as primary_error:
        rollback_errors = []
        if published:
            try:
                if backup is None:
                    with suppress(FileNotFoundError):
                        target.unlink()
                else:
                    os.replace(str(backup), str(target))
                    backup = None
            except Exception as rollback_error:
                rollback_errors.append(f"{target}: {rollback_error}")
                retain_backup = backup is not None
        try:
            _fsync_directories([target])
        except Exception as rollback_error:
            rollback_errors.append(f"directory fsync: {rollback_error}")
        message = f"regeneration transaction failed: {primary_error}"
        if rollback_errors:
            message += f"; rollback failure: {'; '.join(rollback_errors)}"
        raise ValueError(message) from primary_error
    finally:
        with suppress(FileNotFoundError):
            staged.stage_path.unlink()
        if not committed and backup is not None and not retain_backup:
            with suppress(FileNotFoundError):
                backup.unlink()

    cleanup_errors = []
    recovery_artifacts = []
    if backup is not None:
        try:
            _cleanup_backup(backup)
        except Exception as cleanup_error:
            cleanup_errors.append(f"{backup}: {cleanup_error}")
            recovery_artifacts.append(str(backup))
    try:
        _fsync_directories([target])
    except Exception as cleanup_error:
        cleanup_errors.append(f"directory fsync: {cleanup_error}")
    return cleanup_errors, recovery_artifacts


def _stage_stamped_clock(record, curated_dictionary, version, index):
    """Load one clock, restamp its metadata, and stage the rewritten weight file.

    The loaded clock is released before returning, so the caller never holds
    two clocks at once.

    Parameters
    ----------
    record : _TargetRecord
        Preflight record for the weight file.
    curated_dictionary : dict
        Curated registry entries, keyed by clock name.
    version : str
        The pyaging release version to stamp.
    index : int
        Position in the run, used only to name temporary files.

    Returns
    -------
    tuple of (str, dict, _StagedTarget)
        The clock name, its runtime metadata fields, and the staged file.
    """
    clock = torch.load(record.logical_path, weights_only=False)
    try:
        clock_name = _generated_clock_name(clock)
        expected_name = record.logical_path.stem
        if clock_name != expected_name:
            raise ValueError(
                "Generated clock names do not match weight filenames: "
                f"expected {expected_name!r}, generated {clock_name!r}"
            )
        clock.version = version
        clock_name, runtime_metadata = _generated_metadata_entry(clock, version)
        clock.metadata = merge_clock_metadata(
            {clock_name: runtime_metadata},
            {clock_name: curated_dictionary[clock_name]},
        )[clock_name]
        stage_path, digest, size = _stage_torch_object(record.target_path, clock, index)
    finally:
        del clock
    return clock_name, runtime_metadata, _StagedTarget(record, stage_path, digest, size)


def _existing_runtime_metadata_entry(record):
    """Read one unchanged clock's runtime metadata without resaving it."""
    clock = torch.load(record.logical_path, weights_only=False)
    try:
        clock_name = _generated_clock_name(clock)
        expected_name = record.logical_path.stem
        if clock_name != expected_name:
            raise ValueError(
                "Generated clock names do not match weight filenames: "
                f"expected {expected_name!r}, generated {clock_name!r}"
            )
        version = getattr(clock, "version", None)
        metadata_version = clock.metadata.get("version")
        if not isinstance(version, str) or not version:
            raise ValueError(f"Unchanged clock '{clock_name}' has no embedded release version")
        if metadata_version != version:
            raise ValueError(
                f"Unchanged clock '{clock_name}' has inconsistent versions: "
                f"attribute {version!r}, metadata {metadata_version!r}"
            )
        return _generated_metadata_entry(clock, version)
    finally:
        del clock


def regenerate_clock_metadata(
    version,
    weights_dir=_DEFAULT_WEIGHTS_DIR,
    registry_path=_DEFAULT_REGISTRY_PATH,
    metadata_path=_DEFAULT_METADATA_PATH,
):
    """Restamp every clock and rebuild the aggregate metadata, one clock at a time.

    Each weight file is loaded, stamped, staged, published and released before
    the next one is opened, so peak disk and memory are bounded by the largest
    single clock rather than by the whole catalogue.

    Atomicity is per file, not per run: every ``.pt`` is either its old bytes
    or its complete new bytes, never a torn mixture. A failure part-way through
    therefore leaves the earlier clocks restamped, the rest untouched, and the
    aggregate file stale, because it is published last. Re-running the script
    is the recovery: restamping an already-restamped clock is a no-op.

    Parameters
    ----------
    version : str
        The pyaging release version stamped onto every clock and aggregate entry.
    weights_dir : pathlib.Path, optional
        Directory of built ``.pt`` weights. Defaults to ``clocks/weights``
        resolved relative to this script, not the working directory.
    registry_path : pathlib.Path, optional
        Curated JSON metadata registry.
    metadata_path : pathlib.Path, optional
        Aggregate ``.pt`` metadata file to write.

    Returns
    -------
    dict
        The merged aggregate metadata, keyed by clock name.
    """
    weights_dir = Path(weights_dir)
    registry_path = Path(registry_path)
    metadata_path = Path(metadata_path)
    curated_dictionary = load_curated_metadata(registry_path)
    registry_clock_names = set(curated_dictionary)
    weight_records = preflight_weight_files(weights_dir, registry_clock_names)
    metadata_record = _resolve_target(metadata_path, require_existing=False)
    all_target_paths = [record.target_path for record in weight_records] + [metadata_record.target_path]
    if len(set(all_target_paths)) != len(all_target_paths):
        raise ValueError("Distinct logical paths must not resolve to the same backing target")

    generated_dictionary = {}
    cleanup_errors = []
    recovery_artifacts = []
    for index, record in enumerate(weight_records, start=1):
        try:
            clock_name, runtime_metadata, staged = _stage_stamped_clock(record, curated_dictionary, version, index)
        except Exception as error:
            raise ValueError(f"regeneration staging failed: {error}") from error
        errors, artifacts = _publish_staged_target(staged, index)
        cleanup_errors += errors
        recovery_artifacts += artifacts
        generated_dictionary[clock_name] = runtime_metadata

    generated_clock_names = set(generated_dictionary)
    if len(generated_dictionary) != len(weight_records) or generated_clock_names != registry_clock_names:
        raise ValueError(
            "Generated clock names changed during staging: "
            f"expected {sorted(registry_clock_names)}, "
            f"generated {sorted(generated_clock_names)}"
        )

    combined_dictionary = merge_clock_metadata(generated_dictionary, curated_dictionary)
    if set(combined_dictionary) != registry_clock_names:
        raise ValueError("Merged aggregate clock names do not match registry")

    aggregate_index = len(weight_records) + 1
    try:
        aggregate_stage, aggregate_digest, aggregate_size = _stage_torch_object(
            metadata_record.target_path,
            combined_dictionary,
            aggregate_index,
        )
    except Exception as error:
        raise ValueError(f"regeneration staging failed: {error}") from error
    errors, artifacts = _publish_staged_target(
        _StagedTarget(metadata_record, aggregate_stage, aggregate_digest, aggregate_size),
        aggregate_index,
    )
    cleanup_errors += errors
    recovery_artifacts += artifacts

    if cleanup_errors:
        message = "regeneration targets committed; post-commit cleanup failed: " + "; ".join(cleanup_errors)
        if recovery_artifacts:
            message += "; recovery artifact(s) retained: " + ", ".join(recovery_artifacts)
        raise _CommittedRegenerationError(message)
    return combined_dictionary


def regenerate_selected_clock_metadata(
    version,
    clock_names,
    weights_dir=_DEFAULT_WEIGHTS_DIR,
    registry_path=_DEFAULT_REGISTRY_PATH,
    metadata_path=_DEFAULT_METADATA_PATH,
):
    """Restamp selected clocks and rebuild a mixed-version aggregate.

    Selected weights use the same staged serialization and registry merge as
    :func:`regenerate_clock_metadata`. Every other weight is loaded only long
    enough to validate and read its existing runtime metadata; it is never
    staged or resaved.
    """
    weights_dir = Path(weights_dir)
    registry_path = Path(registry_path)
    metadata_path = Path(metadata_path)
    selected_names = tuple(clock_names)
    if not selected_names:
        raise ValueError("At least one clock name is required for selected regeneration")
    if len(set(selected_names)) != len(selected_names):
        raise ValueError("Selected clock names must be unique")

    curated_dictionary = load_curated_metadata(registry_path)
    registry_clock_names = set(curated_dictionary)
    unknown_names = set(selected_names) - registry_clock_names
    if unknown_names:
        raise ValueError(f"Selected clock names are not in the registry: {sorted(unknown_names)}")

    weight_records = preflight_weight_files(weights_dir, registry_clock_names)
    records_by_name = {record.logical_path.stem: record for record in weight_records}
    selected_records = [records_by_name[name] for name in selected_names]
    metadata_record = _resolve_target(metadata_path, require_existing=False)
    target_paths = [record.target_path for record in weight_records] + [metadata_record.target_path]
    if len(set(target_paths)) != len(target_paths):
        raise ValueError("Distinct logical paths must not resolve to the same backing target")

    cleanup_errors = []
    recovery_artifacts = []
    for index, record in enumerate(selected_records, start=1):
        try:
            _, _, staged = _stage_stamped_clock(record, curated_dictionary, version, index)
        except Exception as error:
            raise ValueError(f"selected regeneration staging failed: {error}") from error
        errors, artifacts = _publish_staged_target(staged, index)
        cleanup_errors += errors
        recovery_artifacts += artifacts

    generated_dictionary = {}
    for record in weight_records:
        try:
            clock_name, runtime_metadata = _existing_runtime_metadata_entry(record)
        except Exception as error:
            raise ValueError(f"selected aggregate generation failed: {error}") from error
        generated_dictionary[clock_name] = runtime_metadata

    if set(generated_dictionary) != registry_clock_names:
        raise ValueError("Generated clock names changed while rebuilding selected aggregate metadata")
    combined_dictionary = merge_clock_metadata(generated_dictionary, curated_dictionary)

    aggregate_index = len(selected_records) + 1
    try:
        aggregate_stage, aggregate_digest, aggregate_size = _stage_torch_object(
            metadata_record.target_path,
            combined_dictionary,
            aggregate_index,
        )
    except Exception as error:
        raise ValueError(f"selected regeneration staging failed: {error}") from error
    errors, artifacts = _publish_staged_target(
        _StagedTarget(metadata_record, aggregate_stage, aggregate_digest, aggregate_size),
        aggregate_index,
    )
    cleanup_errors += errors
    recovery_artifacts += artifacts

    if cleanup_errors:
        message = "selected regeneration targets committed; post-commit cleanup failed: " + "; ".join(cleanup_errors)
        if recovery_artifacts:
            message += "; recovery artifact(s) retained: " + ", ".join(recovery_artifacts)
        raise _CommittedRegenerationError(message)
    return combined_dictionary


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Merge PT files metadata.")
    parser.add_argument("version", type=str, help="Version number to be added to the metadata.")
    parser.add_argument(
        "--clock",
        action="append",
        dest="clock_names",
        help="Restamp only this clock (repeatable) and preserve other embedded versions.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()

    if args.clock_names:
        regenerate_selected_clock_metadata(args.version, args.clock_names)
    else:
        regenerate_clock_metadata(args.version)
    print(f"Metadata dictionary saved to '{_DEFAULT_METADATA_PATH}'.")
