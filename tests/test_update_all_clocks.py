import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


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
            "version": "stale metadata version",
            "preprocess": "stale metadata preprocess",
        },
        postprocess_name=None,
        preprocess_name=None,
        reference_values=None,
        version="0.2.0",
    )


def _write_registry(path, registry):
    path.write_text(json.dumps(registry), encoding="utf-8")


def test_merge_clock_metadata_uses_registry_for_curated_fields_and_generated_runtime():
    update_all_clocks = _load_update_all_clocks_module()
    registry = {
        "clock": {
            "clock_name": "clock",
            "citation": "Curated citation",
            "notes": "Carefully curated notes",
            "version": "stale registry version",
            "preprocess": "stale registry preprocess",
            "postprocess": "stale registry postprocess",
            "reference_values": True,
        }
    }
    generated_metadata = {
        "clock": {
            "clock_name": "clock",
            "citation": "Generated citation",
            "notes": "Generated notes",
            "version": "0.3.0",
            "preprocess": "new_preprocess",
        }
    }

    merged_metadata = update_all_clocks.merge_clock_metadata(generated_metadata, registry)

    assert merged_metadata == {
        "clock": {
            "clock_name": "clock",
            "citation": "Curated citation",
            "notes": "Carefully curated notes",
            "version": "0.3.0",
            "preprocess": "new_preprocess",
        }
    }


def test_generated_metadata_entry_contains_only_runtime_fields():
    update_all_clocks = _load_update_all_clocks_module()
    clock = _clock("clock")
    clock.preprocess_name = "runtime_preprocess"
    clock.postprocess_name = "runtime_postprocess"
    clock.reference_values = {"mean": 1.0}

    clock_name, generated_entry = update_all_clocks._generated_metadata_entry(clock)

    assert clock_name == "clock"
    assert generated_entry == {
        "version": "0.2.0",
        "preprocess": "runtime_preprocess",
        "postprocess": "runtime_postprocess",
        "reference_values": True,
    }


def test_load_curated_metadata_reads_utf8_json_registry(tmp_path):
    update_all_clocks = _load_update_all_clocks_module()
    registry_path = tmp_path / "clock_metadata.json"
    registry = {"clock": {"clock_name": "clock", "notes": "Café clock"}}
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


def test_load_curated_metadata_rejects_invalid_json_cleanly(tmp_path):
    update_all_clocks = _load_update_all_clocks_module()
    registry_path = tmp_path / "clock_metadata.json"
    registry_path.write_text('{"clock":', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid curated metadata JSON"):
        update_all_clocks.load_curated_metadata(registry_path)


@pytest.mark.parametrize(
    ("invalid_metadata", "message"),
    [
        ([], "top-level dictionary"),
        ({"UpperClock": {"clock_name": "UpperClock"}}, "lowercase non-empty string"),
        ({"": {"clock_name": ""}}, "lowercase non-empty string"),
        ({"clock": []}, "entry for 'clock' must be a dictionary"),
        (
            {"clock": {"clock_name": "different"}},
            "entry clock_name for 'clock' must match its key",
        ),
    ],
)
def test_load_curated_metadata_rejects_invalid_registry(tmp_path, invalid_metadata, message):
    update_all_clocks = _load_update_all_clocks_module()
    registry_path = tmp_path / "clock_metadata.json"
    _write_registry(registry_path, invalid_metadata)

    with pytest.raises(ValueError, match=message):
        update_all_clocks.load_curated_metadata(registry_path)


def test_broken_weight_preflight_prevents_all_weight_and_aggregate_saves(tmp_path, monkeypatch):
    update_all_clocks = _load_update_all_clocks_module()
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    (weights_dir / "a_good.pt").touch()
    (weights_dir / "b_broken.pt").touch()
    registry_path = tmp_path / "clock_metadata.json"
    _write_registry(
        registry_path,
        {
            "a_good": {"clock_name": "a_good"},
            "b_broken": {"clock_name": "b_broken"},
        },
    )
    load = Mock(side_effect=[_clock("a_good"), OSError("broken weight")])
    save = Mock()
    monkeypatch.setattr(update_all_clocks.torch, "load", load)
    monkeypatch.setattr(update_all_clocks.torch, "save", save)

    with pytest.raises(OSError, match="broken weight"):
        update_all_clocks.regenerate_clock_metadata(
            "0.3.0",
            weights_dir=weights_dir,
            registry_path=registry_path,
            metadata_path=tmp_path / "all_clock_metadata.pt",
        )

    assert load.call_count == 2
    save.assert_not_called()


@pytest.mark.parametrize("directory_state", ["missing", "empty"])
def test_regeneration_requires_nonempty_weights_directory(tmp_path, monkeypatch, directory_state):
    update_all_clocks = _load_update_all_clocks_module()
    weights_dir = tmp_path / "weights"
    if directory_state == "empty":
        weights_dir.mkdir()
    registry_path = tmp_path / "clock_metadata.json"
    _write_registry(registry_path, {"clock": {"clock_name": "clock"}})
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


@pytest.mark.parametrize(
    ("registry", "weight_name", "message"),
    [
        (
            {
                "clock": {"clock_name": "clock"},
                "registry_only": {"clock_name": "registry_only"},
            },
            "clock",
            "Registry and weight clock names must match exactly",
        ),
        (
            {"clock": {"clock_name": "clock"}},
            "weight_only",
            "Registry and weight clock names must match exactly",
        ),
    ],
)
def test_registry_and_weight_sets_must_match_before_any_write(tmp_path, monkeypatch, registry, weight_name, message):
    update_all_clocks = _load_update_all_clocks_module()
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    (weights_dir / f"{weight_name}.pt").touch()
    registry_path = tmp_path / "clock_metadata.json"
    _write_registry(registry_path, registry)
    load = Mock(return_value=_clock(weight_name))
    save = Mock()
    monkeypatch.setattr(update_all_clocks.torch, "load", load)
    monkeypatch.setattr(update_all_clocks.torch, "save", save)

    with pytest.raises(ValueError, match=message):
        update_all_clocks.regenerate_clock_metadata(
            "0.3.0",
            weights_dir=weights_dir,
            registry_path=registry_path,
            metadata_path=tmp_path / "all_clock_metadata.pt",
        )

    load.assert_called_once()
    save.assert_not_called()


def test_update_failure_never_writes_aggregate(tmp_path, monkeypatch):
    update_all_clocks = _load_update_all_clocks_module()
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    weight_path = weights_dir / "clock.pt"
    weight_path.touch()
    registry_path = tmp_path / "clock_metadata.json"
    _write_registry(registry_path, {"clock": {"clock_name": "clock", "notes": "Curated notes"}})
    load = Mock(side_effect=[_clock("clock"), _clock("clock")])
    save = Mock(side_effect=OSError("disk full"))
    monkeypatch.setattr(update_all_clocks.torch, "load", load)
    monkeypatch.setattr(update_all_clocks.torch, "save", save)

    with pytest.raises(OSError, match="disk full"):
        update_all_clocks.regenerate_clock_metadata(
            "0.3.0",
            weights_dir=weights_dir,
            registry_path=registry_path,
            metadata_path=tmp_path / "all_clock_metadata.pt",
        )

    save.assert_called_once()
    assert save.call_args.args[1] == weight_path


def test_post_preflight_merge_failure_never_writes_aggregate(tmp_path, monkeypatch):
    update_all_clocks = _load_update_all_clocks_module()
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    weight_path = weights_dir / "clock.pt"
    weight_path.touch()
    registry_path = tmp_path / "clock_metadata.json"
    _write_registry(registry_path, {"clock": {"clock_name": "clock"}})
    load = Mock(side_effect=[_clock("clock"), _clock("different")])
    save = Mock()
    monkeypatch.setattr(update_all_clocks.torch, "load", load)
    monkeypatch.setattr(update_all_clocks.torch, "save", save)

    with pytest.raises(ValueError, match="Generated clock names changed after preflight"):
        update_all_clocks.regenerate_clock_metadata(
            "0.3.0",
            weights_dir=weights_dir,
            registry_path=registry_path,
            metadata_path=tmp_path / "all_clock_metadata.pt",
        )

    save.assert_called_once()
    assert save.call_args.args[1] == weight_path


def test_successful_regeneration_saves_registry_backed_aggregate_last(tmp_path, monkeypatch):
    update_all_clocks = _load_update_all_clocks_module()
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    weight_path = weights_dir / "clock.pt"
    weight_path.touch()
    registry_path = tmp_path / "clock_metadata.json"
    _write_registry(
        registry_path,
        {
            "clock": {
                "clock_name": "clock",
                "citation": "Curated citation",
                "notes": "Curated notes",
                "version": "stale registry version",
                "postprocess": "stale registry postprocess",
            }
        },
    )
    preflight_clock = _clock("clock")
    updated_clock = _clock("clock")
    updated_clock.preprocess_name = "runtime_preprocess"
    load = Mock(side_effect=[preflight_clock, updated_clock])
    save = Mock()
    monkeypatch.setattr(update_all_clocks.torch, "load", load)
    monkeypatch.setattr(update_all_clocks.torch, "save", save)
    metadata_path = tmp_path / "all_clock_metadata.pt"

    result = update_all_clocks.regenerate_clock_metadata(
        "0.3.0",
        weights_dir=weights_dir,
        registry_path=registry_path,
        metadata_path=metadata_path,
    )

    assert result == {
        "clock": {
            "clock_name": "clock",
            "citation": "Curated citation",
            "notes": "Curated notes",
            "version": "0.3.0",
            "preprocess": "runtime_preprocess",
        }
    }
    assert save.call_count == 2
    assert save.call_args_list[0].args[1] == weight_path
    assert save.call_args_list[1].args == (result, metadata_path)
