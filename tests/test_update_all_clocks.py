import gc
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from pyaging.models import pyagingModel

REGISTRY_FIELDS = (
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
    "version",
)


def _load_update_all_clocks_module():
    script_path = Path(__file__).parents[1] / "clocks" / "update_all_clocks.py"
    spec = importlib.util.spec_from_file_location("update_all_clocks", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clock(clock_name):
    return SimpleNamespace(
        metadata={
            "clock_name": clock_name,
            "citation": "Generated citation",
            "notes": "Generated notes",
            "tissue": ["stale controlled tissue"],
            "legacy_field": "must be removed",
            "version": "stale metadata version",
            "preprocess": "stale metadata preprocess",
        },
        postprocess_name=None,
        preprocess_name=None,
        reference_values=None,
        version="0.2.0",
    )


def _registry_entry(clock_name, **updates):
    entry = {
        "clock_name": clock_name,
        "data_type": "methylation",
        "species": "human",
        "year": 2020,
        "approved_by_author": "no",
        "citation": "Citation",
        "doi": "https://doi.org/10.1000/example",
        "notes": "Curated notes",
        "research_only": None,
        "tissue": ["blood"],
        "predicts": ["chronological age"],
        "training_target": ["chronological age"],
        "unit": ["years"],
        "model_type": "linear regression",
        "platform": ["Illumina HumanMethylation450 BeadChip"],
        "population": "adult",
        "journal": "Journal",
        "last_author": "Author",
        "n_features": 1,
        "citations": 0,
        "citations_date": "2026-07-18",
        "version": "stale registry version",
    }
    entry.update(updates)
    return entry


def _write_registry(path, registry):
    path.write_text(json.dumps(registry), encoding="utf-8")


def _write_inputs(tmp_path, clock_names=("clock",)):
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    for clock_name in clock_names:
        torch.save(_clock(clock_name), weights_dir / f"{clock_name}.pt")
    registry_path = tmp_path / "clock_metadata.json"
    _write_registry(
        registry_path,
        {clock_name: _registry_entry(clock_name) for clock_name in sorted(clock_names)},
    )
    metadata_path = tmp_path / "all_clock_metadata.pt"
    metadata_path.write_bytes(b"original aggregate")
    return weights_dir, registry_path, metadata_path


def _snapshot(paths):
    return {path: (path.read_bytes(), path.stat().st_mode & 0o7777) for path in paths}


def _assert_snapshot(snapshot):
    for path, (content, mode) in snapshot.items():
        assert path.read_bytes() == content
        assert path.stat().st_mode & 0o7777 == mode


def _transaction_residue(tmp_path):
    return sorted(
        path
        for path in tmp_path.rglob(".*")
        if ".clock-metadata-stage-" in path.name or ".clock-metadata-backup-" in path.name
    )


def _write_symlink_inputs(tmp_path, clock_names=("clock",)):
    backing_dir = tmp_path / "backing"
    backing_dir.mkdir()
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    logical_weights = []
    backing_weights = []
    for clock_name in clock_names:
        backing = backing_dir / f"{clock_name}.pt"
        torch.save(_clock(clock_name), backing)
        logical = weights_dir / f"{clock_name}.pt"
        logical.symlink_to(backing)
        backing_weights.append(backing)
        logical_weights.append(logical)

    registry_path = tmp_path / "clock_metadata.json"
    _write_registry(
        registry_path,
        {clock_name: _registry_entry(clock_name) for clock_name in sorted(clock_names)},
    )
    backing_metadata = backing_dir / "all_clock_metadata.pt"
    backing_metadata.write_bytes(b"original aggregate")
    metadata_path = tmp_path / "all_clock_metadata.pt"
    metadata_path.symlink_to(backing_metadata)
    return (
        weights_dir,
        registry_path,
        metadata_path,
        logical_weights,
        backing_weights,
        backing_metadata,
    )


def _symlink_identity(path):
    stat_result = path.lstat()
    return (
        os.readlink(path),
        stat_result.st_mode,
        stat_result.st_ino,
        stat_result.st_dev,
    )


def test_runtime_metadata_fields_are_explicitly_ordered():
    update_all_clocks = _load_update_all_clocks_module()

    assert update_all_clocks.RUNTIME_METADATA_FIELDS == (
        "version",
        "preprocess",
        "postprocess",
        "reference_values",
    )

    clock = _clock("clock")
    clock.preprocess_name = "runtime_preprocess"
    clock.postprocess_name = "runtime_postprocess"
    clock.reference_values = {"mean": 1.0}
    generated_orders = [list(update_all_clocks._generated_metadata_entry(clock, "0.3.0")[1]) for _ in range(10)]
    assert generated_orders == [list(update_all_clocks.RUNTIME_METADATA_FIELDS)] * 10


def test_runtime_metadata_serialization_order_is_hash_seed_independent():
    module_path = Path(__file__).parents[1] / "clocks" / "update_all_clocks.py"
    script = f"""
import importlib.util
import json
from types import SimpleNamespace
spec = importlib.util.spec_from_file_location("update_all_clocks", {str(module_path)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
clock = SimpleNamespace(
    metadata={{"clock_name": "clock"}},
    preprocess_name="pre",
    postprocess_name="post",
    reference_values={{"mean": 1.0}},
)
print(json.dumps(module._generated_metadata_entry(clock, "0.3.0")[1]))
"""
    outputs = []
    for seed in ("1", "8675309"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        outputs.append(
            subprocess.run(
                [sys.executable, "-c", script],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            ).stdout.strip()
        )

    assert outputs == ['{"version": "0.3.0", "preprocess": "pre", "postprocess": "post", "reference_values": true}'] * 2


def test_merge_clock_metadata_uses_registry_for_curated_fields_and_generated_runtime():
    update_all_clocks = _load_update_all_clocks_module()
    registry = {
        "clock": _registry_entry(
            "clock",
            citation="Curated citation",
            notes="Carefully curated notes",
            version="stale registry version",
        )
    }
    generated_metadata = {
        "clock": {
            "version": "0.3.0",
            "preprocess": "new_preprocess",
        }
    }

    merged_metadata = update_all_clocks.merge_clock_metadata(generated_metadata, registry)

    expected = _registry_entry(
        "clock",
        citation="Curated citation",
        notes="Carefully curated notes",
    )
    expected.pop("version")
    expected.update(version="0.3.0", preprocess="new_preprocess")
    assert merged_metadata == {"clock": expected}
    assert list(merged_metadata["clock"])[-2:] == ["version", "preprocess"]


def test_generated_metadata_entry_contains_only_runtime_fields():
    update_all_clocks = _load_update_all_clocks_module()
    clock = _clock("clock")
    clock.preprocess_name = "runtime_preprocess"
    clock.postprocess_name = "runtime_postprocess"
    clock.reference_values = {"mean": 1.0}

    clock_name, generated_entry = update_all_clocks._generated_metadata_entry(clock, "0.2.0")

    assert clock_name == "clock"
    assert generated_entry == {
        "version": "0.2.0",
        "preprocess": "runtime_preprocess",
        "postprocess": "runtime_postprocess",
        "reference_values": True,
    }


def test_load_curated_metadata_reads_canonical_utf8_json_registry(tmp_path):
    update_all_clocks = _load_update_all_clocks_module()
    registry_path = tmp_path / "clock_metadata.json"
    registry = {
        "alpha": _registry_entry("alpha", notes="Café clock"),
        "beta": _registry_entry("beta"),
    }
    _write_registry(registry_path, registry)

    assert update_all_clocks.load_curated_metadata(registry_path) == registry


def test_load_curated_metadata_accepts_registry_without_optional_runtime_fields(
    tmp_path,
):
    update_all_clocks = _load_update_all_clocks_module()
    registry_path = tmp_path / "clock_metadata.json"
    entry = _registry_entry("clock")
    entry.pop("version")
    registry = {"clock": entry}
    _write_registry(registry_path, registry)

    assert update_all_clocks.load_curated_metadata(registry_path) == registry


def test_regeneration_requires_registry_before_loading_weights(tmp_path, monkeypatch):
    update_all_clocks = _load_update_all_clocks_module()
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    (weights_dir / "clock.pt").touch()
    load = Mock()
    save = Mock()
    monkeypatch.setattr(update_all_clocks.torch, "load", load)
    monkeypatch.setattr(update_all_clocks.torch, "save", save)

    with pytest.raises(FileNotFoundError, match="Curated metadata registry is required"):
        update_all_clocks.regenerate_clock_metadata(
            "0.3.0",
            weights_dir=weights_dir,
            registry_path=tmp_path / "missing.json",
            metadata_path=tmp_path / "all_clock_metadata.pt",
        )

    load.assert_not_called()
    save.assert_not_called()


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('{"clock": {}, "clock": {}}', "duplicate key 'clock'"),
        ('{"clock": {"clock_name": NaN}}', "non-finite JSON number"),
        ('{"clock": {"clock_name": Infinity}}', "non-finite JSON number"),
        ('{"clock":', "Invalid curated metadata JSON"),
    ],
)
def test_load_curated_metadata_rejects_non_strict_json(tmp_path, text, message):
    update_all_clocks = _load_update_all_clocks_module()
    registry_path = tmp_path / "clock_metadata.json"
    registry_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        update_all_clocks.load_curated_metadata(registry_path)


@pytest.mark.parametrize(
    ("registry", "message"),
    [
        ([], "top-level dictionary"),
        (
            {
                "beta": _registry_entry("beta"),
                "alpha": _registry_entry("alpha"),
            },
            "clock keys must be in alphabetical order",
        ),
        (
            {"UpperClock": _registry_entry("UpperClock")},
            "lowercase non-empty string",
        ),
        ({"": _registry_entry("")}, "lowercase non-empty string"),
        ({"clock": []}, "entry for 'clock' must be a dictionary"),
        (
            {"clock": _registry_entry("different")},
            "entry clock_name for 'clock' must match its key",
        ),
        (
            {"clock": {key: value for key, value in _registry_entry("clock").items() if key != "notes"}},
            "canonical field order",
        ),
        (
            {"clock": dict(reversed(list(_registry_entry("clock").items())))},
            "canonical field order",
        ),
        (
            {"clock": _registry_entry("clock", year="2020")},
            "clock.year must be an integer",
        ),
        (
            {"clock": _registry_entry("clock", tissue="blood")},
            "clock.tissue must be a non-empty string array",
        ),
        (
            {"clock": _registry_entry("clock", research_only="no")},
            "clock.research_only must be a boolean or null",
        ),
    ],
)
def test_load_curated_metadata_rejects_invalid_registry(tmp_path, registry, message):
    update_all_clocks = _load_update_all_clocks_module()
    registry_path = tmp_path / "clock_metadata.json"
    _write_registry(registry_path, registry)

    with pytest.raises(ValueError, match=message):
        update_all_clocks.load_curated_metadata(registry_path)


@pytest.mark.parametrize("directory_state", ["missing", "empty"])
def test_regeneration_requires_nonempty_weights_directory(tmp_path, monkeypatch, directory_state):
    update_all_clocks = _load_update_all_clocks_module()
    weights_dir = tmp_path / "weights"
    if directory_state == "empty":
        weights_dir.mkdir()
    registry_path = tmp_path / "clock_metadata.json"
    _write_registry(registry_path, {"clock": _registry_entry("clock")})
    load = Mock()
    save = Mock()
    monkeypatch.setattr(update_all_clocks.torch, "load", load)
    monkeypatch.setattr(update_all_clocks.torch, "save", save)

    with pytest.raises(ValueError, match="non-empty weights directory"):
        update_all_clocks.regenerate_clock_metadata(
            "0.3.0",
            weights_dir=weights_dir,
            registry_path=registry_path,
            metadata_path=tmp_path / "all_clock_metadata.pt",
        )

    load.assert_not_called()
    save.assert_not_called()


def test_broken_later_weight_cleans_earlier_stage_without_publishing(tmp_path, monkeypatch):
    update_all_clocks = _load_update_all_clocks_module()
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    (weights_dir / "alpha.pt").touch()
    (weights_dir / "beta.pt").touch()
    registry_path = tmp_path / "clock_metadata.json"
    _write_registry(
        registry_path,
        {
            "alpha": _registry_entry("alpha"),
            "beta": _registry_entry("beta"),
        },
    )
    load = Mock(side_effect=[_clock("alpha"), OSError("broken weight")])
    monkeypatch.setattr(update_all_clocks.torch, "load", load)
    originals = _snapshot([weights_dir / "alpha.pt", weights_dir / "beta.pt"])

    with pytest.raises(ValueError, match="broken weight"):
        update_all_clocks.regenerate_clock_metadata(
            "0.3.0",
            weights_dir=weights_dir,
            registry_path=registry_path,
            metadata_path=tmp_path / "all_clock_metadata.pt",
        )

    assert load.call_count == 2
    _assert_snapshot(originals)
    assert _transaction_residue(tmp_path) == []


@pytest.mark.parametrize(
    ("registry_names", "weight_names"),
    [
        (("alpha", "registry_only"), ("alpha",)),
        (("alpha",), ("alpha", "weight_only")),
    ],
)
def test_registry_and_weight_sets_must_match_before_staging(tmp_path, monkeypatch, registry_names, weight_names):
    update_all_clocks = _load_update_all_clocks_module()
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    for clock_name in weight_names:
        torch.save(_clock(clock_name), weights_dir / f"{clock_name}.pt")
    registry_path = tmp_path / "clock_metadata.json"
    _write_registry(
        registry_path,
        {clock_name: _registry_entry(clock_name) for clock_name in sorted(registry_names)},
    )
    stage = Mock()
    monkeypatch.setattr(update_all_clocks, "_stage_torch_object", stage, raising=False)

    with pytest.raises(ValueError, match="Registry and weight clock names must match exactly"):
        update_all_clocks.regenerate_clock_metadata(
            "0.3.0",
            weights_dir=weights_dir,
            registry_path=registry_path,
            metadata_path=tmp_path / "all_clock_metadata.pt",
        )

    stage.assert_not_called()


def test_invalid_updated_clock_name_is_never_staged(tmp_path, monkeypatch):
    update_all_clocks = _load_update_all_clocks_module()
    weights_dir, registry_path, metadata_path = _write_inputs(tmp_path)
    clock_path = weights_dir / "clock.pt"
    original_load = update_all_clocks.torch.load
    loaded_clock = original_load(clock_path, weights_only=False)
    loaded_clock.metadata["clock_name"] = "different"
    torch.save(loaded_clock, clock_path)
    original = _snapshot([clock_path, metadata_path])
    stage = Mock()
    monkeypatch.setattr(update_all_clocks, "_stage_torch_object", stage, raising=False)

    with pytest.raises(ValueError, match="Generated clock names do not match weight filenames"):
        update_all_clocks.regenerate_clock_metadata(
            "0.3.0",
            weights_dir=weights_dir,
            registry_path=registry_path,
            metadata_path=metadata_path,
        )

    stage.assert_not_called()
    _assert_snapshot(original)


def test_later_stage_save_failure_preserves_all_original_bytes_and_modes(tmp_path, monkeypatch):
    update_all_clocks = _load_update_all_clocks_module()
    weights_dir, registry_path, metadata_path = _write_inputs(tmp_path, ("alpha", "beta"))
    targets = [weights_dir / "alpha.pt", weights_dir / "beta.pt", metadata_path]
    os.chmod(targets[0], 0o640)
    os.chmod(targets[1], 0o600)
    os.chmod(metadata_path, 0o644)
    original = _snapshot(targets)
    original_save = update_all_clocks.torch.save
    calls = 0

    def fail_second_save(value, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected later stage save")
        return original_save(value, destination)

    monkeypatch.setattr(update_all_clocks.torch, "save", fail_second_save)

    with pytest.raises(ValueError, match="injected later stage save"):
        update_all_clocks.regenerate_clock_metadata(
            "0.3.0",
            weights_dir=weights_dir,
            registry_path=registry_path,
            metadata_path=metadata_path,
        )

    _assert_snapshot(original)
    assert _transaction_residue(tmp_path) == []


@pytest.mark.parametrize("failure_point", [1, 2, 3])
def test_replace_failure_rolls_back_every_target_without_residue(tmp_path, monkeypatch, failure_point):
    update_all_clocks = _load_update_all_clocks_module()
    weights_dir, registry_path, metadata_path = _write_inputs(tmp_path, ("alpha", "beta"))
    targets = [weights_dir / "alpha.pt", weights_dir / "beta.pt", metadata_path]
    os.chmod(targets[0], 0o640)
    os.chmod(targets[1], 0o600)
    original = _snapshot(targets)
    original_publish = update_all_clocks._publish_replace
    calls = 0

    def fail_replace(source, target):
        nonlocal calls
        calls += 1
        if calls == failure_point:
            raise OSError(f"injected replace {failure_point}")
        original_publish(source, target)

    monkeypatch.setattr(update_all_clocks, "_publish_replace", fail_replace)

    with pytest.raises(ValueError, match=f"injected replace {failure_point}"):
        update_all_clocks.regenerate_clock_metadata(
            "0.3.0",
            weights_dir=weights_dir,
            registry_path=registry_path,
            metadata_path=metadata_path,
        )

    _assert_snapshot(original)
    assert _transaction_residue(tmp_path) == []


def test_backup_cleanup_failure_does_not_roll_back_committed_targets(tmp_path, monkeypatch):
    update_all_clocks = _load_update_all_clocks_module()
    weights_dir, registry_path, metadata_path = _write_inputs(tmp_path)
    clock_path = weights_dir / "clock.pt"
    original_cleanup = update_all_clocks._cleanup_backup
    calls = 0

    def fail_first_cleanup(backup):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected committed cleanup failure")
        original_cleanup(backup)

    monkeypatch.setattr(update_all_clocks, "_cleanup_backup", fail_first_cleanup)

    with pytest.raises(
        update_all_clocks._CommittedRegenerationError,
        match="targets committed.*recovery artifact",
    ) as error:
        update_all_clocks.regenerate_clock_metadata(
            "0.3.0",
            weights_dir=weights_dir,
            registry_path=registry_path,
            metadata_path=metadata_path,
        )

    assert torch.load(clock_path, weights_only=False).version == "0.3.0"
    aggregate = torch.load(metadata_path, weights_only=False)
    assert aggregate["clock"]["version"] == "0.3.0"
    residue = _transaction_residue(tmp_path)
    assert len(residue) == 1
    assert str(residue[0]) in str(error.value)


def test_success_preserves_weight_and_aggregate_symlink_entries(tmp_path):
    update_all_clocks = _load_update_all_clocks_module()
    (
        weights_dir,
        registry_path,
        metadata_path,
        logical_weights,
        backing_weights,
        backing_metadata,
    ) = _write_symlink_inputs(tmp_path)
    logical_paths = logical_weights + [metadata_path]
    identities = {path: _symlink_identity(path) for path in logical_paths}

    result = update_all_clocks.regenerate_clock_metadata(
        "0.3.0",
        weights_dir=weights_dir,
        registry_path=registry_path,
        metadata_path=metadata_path,
    )

    assert {path: _symlink_identity(path) for path in logical_paths} == identities
    assert torch.load(backing_weights[0], weights_only=False).version == "0.3.0"
    assert torch.load(backing_metadata, weights_only=False) == result
    assert _transaction_residue(tmp_path) == []


def test_replace_failure_restores_symlink_backing_targets_without_touching_links(tmp_path, monkeypatch):
    update_all_clocks = _load_update_all_clocks_module()
    (
        weights_dir,
        registry_path,
        metadata_path,
        logical_weights,
        backing_weights,
        backing_metadata,
    ) = _write_symlink_inputs(tmp_path, ("alpha", "beta"))
    logical_paths = logical_weights + [metadata_path]
    identities = {path: _symlink_identity(path) for path in logical_paths}
    backing_paths = backing_weights + [backing_metadata]
    os.chmod(backing_paths[0], 0o640)
    os.chmod(backing_paths[1], 0o600)
    originals = _snapshot(backing_paths)
    original_publish = update_all_clocks._publish_replace
    calls = 0

    def fail_second_replace(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected symlink backing replace")
        original_publish(source, target)

    monkeypatch.setattr(update_all_clocks, "_publish_replace", fail_second_replace)

    with pytest.raises(ValueError, match="injected symlink backing replace"):
        update_all_clocks.regenerate_clock_metadata(
            "0.3.0",
            weights_dir=weights_dir,
            registry_path=registry_path,
            metadata_path=metadata_path,
        )

    assert {path: _symlink_identity(path) for path in logical_paths} == identities
    _assert_snapshot(originals)
    assert _transaction_residue(tmp_path) == []


def test_rejects_dangling_and_aliased_weight_symlinks_before_loading(tmp_path, monkeypatch):
    update_all_clocks = _load_update_all_clocks_module()
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    backing = tmp_path / "backing.pt"
    torch.save(_clock("alpha"), backing)
    (weights_dir / "alpha.pt").symlink_to(backing)
    (weights_dir / "beta.pt").symlink_to(backing)
    registry_path = tmp_path / "clock_metadata.json"
    _write_registry(
        registry_path,
        {
            "alpha": _registry_entry("alpha"),
            "beta": _registry_entry("beta"),
        },
    )
    load = Mock()
    monkeypatch.setattr(update_all_clocks.torch, "load", load)

    with pytest.raises(ValueError, match="resolve to the same backing target"):
        update_all_clocks.regenerate_clock_metadata(
            "0.3.0",
            weights_dir=weights_dir,
            registry_path=registry_path,
            metadata_path=tmp_path / "metadata.pt",
        )
    load.assert_not_called()

    (weights_dir / "beta.pt").unlink()
    (weights_dir / "beta.pt").symlink_to(tmp_path / "missing.pt")
    with pytest.raises(ValueError, match="dangling symbolic link"):
        update_all_clocks.regenerate_clock_metadata(
            "0.3.0",
            weights_dir=weights_dir,
            registry_path=registry_path,
            metadata_path=tmp_path / "metadata.pt",
        )
    load.assert_not_called()


def test_regeneration_streams_one_clock_at_a_time_without_path_read_bytes(tmp_path, monkeypatch):
    update_all_clocks = _load_update_all_clocks_module()
    names = tuple(f"clock_{index:03d}" for index in range(24))
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    for name in names:
        (weights_dir / f"{name}.pt").write_bytes(b"original")
    registry_path = tmp_path / "clock_metadata.json"
    _write_registry(
        registry_path,
        {name: _registry_entry(name) for name in names},
    )
    metadata_path = tmp_path / "all_clock_metadata.pt"
    metadata_path.write_bytes(b"original aggregate")

    live = 0
    maximum_live = 0

    class TrackingClock:
        def __init__(self, clock_name):
            nonlocal live, maximum_live
            live += 1
            maximum_live = max(maximum_live, live)
            self.metadata = {"clock_name": clock_name}
            self.version = "old"
            self.preprocess_name = None
            self.postprocess_name = None
            self.reference_values = None

        def __del__(self):
            nonlocal live
            live -= 1

    def fake_load(path, weights_only=False):
        assert weights_only is False
        return TrackingClock(Path(path).stem)

    def fake_save(value, stream):
        if isinstance(value, TrackingClock):
            stream.write(f"clock:{value.metadata['clock_name']}:{value.version}".encode())
        else:
            stream.write(json.dumps(value).encode())

    monkeypatch.setattr(update_all_clocks.torch, "load", fake_load)
    monkeypatch.setattr(update_all_clocks.torch, "save", fake_save)
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda self: (_ for _ in ()).throw(AssertionError(f"Path.read_bytes is forbidden: {self}")),
    )

    result = update_all_clocks.regenerate_clock_metadata(
        "0.3.0",
        weights_dir=weights_dir,
        registry_path=registry_path,
        metadata_path=metadata_path,
    )
    gc.collect()

    assert len(result) == len(names)
    assert maximum_live == 1
    assert live == 0


def test_successful_regeneration_publishes_valid_registry_backed_transaction(tmp_path, monkeypatch):
    update_all_clocks = _load_update_all_clocks_module()
    weights_dir, registry_path, metadata_path = _write_inputs(tmp_path)
    clock_path = weights_dir / "clock.pt"
    old_mode = clock_path.stat().st_mode & 0o7777
    original_load = update_all_clocks.torch.load
    load = Mock(side_effect=original_load)
    monkeypatch.setattr(update_all_clocks.torch, "load", load)

    result = update_all_clocks.regenerate_clock_metadata(
        "0.3.0",
        weights_dir=weights_dir,
        registry_path=registry_path,
        metadata_path=metadata_path,
    )

    updated_clock = torch.load(clock_path, weights_only=False)
    assert updated_clock.version == "0.3.0"
    assert updated_clock.metadata == result["clock"]
    assert updated_clock.metadata["tissue"] == ["blood"]
    assert "legacy_field" not in updated_clock.metadata
    assert set(updated_clock.metadata) == set(result["clock"])
    assert clock_path.stat().st_mode & 0o7777 == old_mode
    assert torch.load(metadata_path, weights_only=False) == result
    assert result["clock"]["notes"] == "Curated notes"
    assert result["clock"]["version"] == "0.3.0"
    assert load.call_count == 3
    assert load.call_args_list[0].args[0] == clock_path
    assert load.call_args_list[1].args[0] == clock_path
    assert load.call_args_list[2].args[0] == metadata_path
    assert _transaction_residue(tmp_path) == []


class _ConcreteClock(pyagingModel):
    """Minimal concrete ``pyagingModel`` standing in for a built clock.

    The other fixtures in this module fake clocks with ``SimpleNamespace``, so
    they cannot detect the script reading attributes the real model class no
    longer defines. This one inherits the real ``__init__``.
    """

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


def _write_real_model_inputs(tmp_path, clock_names):
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    for clock_name in clock_names:
        clock = _ConcreteClock()
        clock.metadata["clock_name"] = clock_name
        torch.save(clock, weights_dir / f"{clock_name}.pt")
    registry_path = tmp_path / "clock_metadata.json"
    _write_registry(
        registry_path,
        {clock_name: _registry_entry(clock_name) for clock_name in sorted(clock_names)},
    )
    return weights_dir, registry_path, tmp_path / "all_clock_metadata.pt"


def test_regeneration_of_real_model_clocks_covers_every_registry_entry(tmp_path):
    update_all_clocks = _load_update_all_clocks_module()
    clock_names = ("clocka", "clockb")
    weights_dir, registry_path, metadata_path = _write_real_model_inputs(tmp_path, clock_names)

    result = update_all_clocks.regenerate_clock_metadata(
        "0.5.0",
        weights_dir=weights_dir,
        registry_path=registry_path,
        metadata_path=metadata_path,
    )

    assert set(result) == set(clock_names)
    assert torch.load(metadata_path, weights_only=False) == result
    for clock_name in clock_names:
        assert result[clock_name]["version"] == "0.5.0"
        assert torch.load(weights_dir / f"{clock_name}.pt", weights_only=False).metadata == result[clock_name]


def test_script_only_reads_runtime_attributes_the_model_class_defines():
    update_all_clocks = _load_update_all_clocks_module()
    clock = _ConcreteClock()
    clock.metadata["clock_name"] = "clock"

    _, runtime_metadata = update_all_clocks._generated_metadata_entry(clock, "0.5.0")

    assert runtime_metadata["version"] == "0.5.0"
    assert set(runtime_metadata).issubset(update_all_clocks.RUNTIME_METADATA_FIELDS)
