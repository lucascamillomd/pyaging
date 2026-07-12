import importlib.util
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
        metadata={"clock_name": clock_name, "notes": "Generated notes"},
        postprocess_name=None,
        preprocess_name=None,
        reference_values=None,
        version="0.2.0",
    )


def test_merge_clock_metadata_preserves_curated_fields_and_refreshes_runtime_fields():
    update_all_clocks = _load_update_all_clocks_module()
    existing_metadata = {
        "curated_clock": {
            "clock_name": "curated_clock",
            "citation": "Curated citation",
            "notes": "Carefully curated notes",
            "version": "0.2.0",
            "preprocess": "old_preprocess",
            "postprocess": "stale_postprocess",
            "reference_values": True,
        },
        "removed_clock": {
            "clock_name": "removed_clock",
            "notes": "This clock is no longer generated",
        },
    }
    generated_metadata = {
        "curated_clock": {
            "clock_name": "curated_clock",
            "citation": "Uncurated generated citation",
            "notes": "Generated notes",
            "version": "0.3.0",
            "preprocess": "new_preprocess",
        },
        "new_clock": {
            "clock_name": "new_clock",
            "notes": "Generated notes for a new clock",
            "version": "0.3.0",
            "postprocess": "new_postprocess",
        },
    }

    merged_metadata = update_all_clocks.merge_clock_metadata(generated_metadata, existing_metadata)

    assert merged_metadata == {
        "curated_clock": {
            "clock_name": "curated_clock",
            "citation": "Curated citation",
            "notes": "Carefully curated notes",
            "version": "0.3.0",
            "preprocess": "new_preprocess",
        },
        "new_clock": generated_metadata["new_clock"],
    }


def test_regeneration_requires_existing_curated_metadata_before_loading_weights(tmp_path, monkeypatch):
    update_all_clocks = _load_update_all_clocks_module()
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    (weights_dir / "clock.pt").touch()
    load = Mock()
    save = Mock()
    monkeypatch.setattr(update_all_clocks.torch, "load", load)
    monkeypatch.setattr(update_all_clocks.torch, "save", save)

    with pytest.raises(FileNotFoundError, match="Existing curated metadata aggregate is required"):
        update_all_clocks.regenerate_clock_metadata(
            "0.3.0",
            weights_dir=weights_dir,
            metadata_path=tmp_path / "missing.pt",
        )

    load.assert_not_called()
    save.assert_not_called()


@pytest.mark.parametrize(
    ("invalid_metadata", "message"),
    [
        ([], "top-level dictionary"),
        ({"UpperClock": {"clock_name": "UpperClock"}}, "lowercase string"),
        ({"clock": []}, "entry for 'clock' must be a dictionary"),
        (
            {"clock": {"clock_name": "different"}},
            "entry clock_name for 'clock' must match its key",
        ),
    ],
)
def test_regeneration_rejects_invalid_curated_metadata_before_loading_weights(
    tmp_path, monkeypatch, invalid_metadata, message
):
    update_all_clocks = _load_update_all_clocks_module()
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    (weights_dir / "clock.pt").touch()
    metadata_path = tmp_path / "all_clock_metadata.pt"
    metadata_path.touch()
    load = Mock(return_value=invalid_metadata)
    save = Mock()
    monkeypatch.setattr(update_all_clocks.torch, "load", load)
    monkeypatch.setattr(update_all_clocks.torch, "save", save)

    with pytest.raises(ValueError, match=message):
        update_all_clocks.regenerate_clock_metadata("0.3.0", weights_dir=weights_dir, metadata_path=metadata_path)

    load.assert_called_once_with(metadata_path, weights_only=False)
    save.assert_not_called()


def test_broken_weight_preflight_prevents_all_weight_and_aggregate_saves(tmp_path, monkeypatch):
    update_all_clocks = _load_update_all_clocks_module()
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    (weights_dir / "a_good.pt").touch()
    (weights_dir / "b_broken.pt").touch()
    metadata_path = tmp_path / "all_clock_metadata.pt"
    metadata_path.touch()
    load = Mock(
        side_effect=[
            {
                "a_good": {"clock_name": "a_good"},
                "b_broken": {"clock_name": "b_broken"},
            },
            _clock("a_good"),
            OSError("broken weight"),
        ]
    )
    save = Mock()
    monkeypatch.setattr(update_all_clocks.torch, "load", load)
    monkeypatch.setattr(update_all_clocks.torch, "save", save)

    with pytest.raises(OSError, match="broken weight"):
        update_all_clocks.regenerate_clock_metadata("0.3.0", weights_dir=weights_dir, metadata_path=metadata_path)

    assert load.call_count == 3
    save.assert_not_called()


@pytest.mark.parametrize("directory_state", ["missing", "empty"])
def test_regeneration_requires_nonempty_weights_directory(tmp_path, monkeypatch, directory_state):
    update_all_clocks = _load_update_all_clocks_module()
    weights_dir = tmp_path / "weights"
    if directory_state == "empty":
        weights_dir.mkdir()
    metadata_path = tmp_path / "all_clock_metadata.pt"
    metadata_path.touch()
    load = Mock(return_value={"clock": {"clock_name": "clock"}})
    save = Mock()
    monkeypatch.setattr(update_all_clocks.torch, "load", load)
    monkeypatch.setattr(update_all_clocks.torch, "save", save)

    with pytest.raises(ValueError, match="non-empty weights directory"):
        update_all_clocks.regenerate_clock_metadata("0.3.0", weights_dir=weights_dir, metadata_path=metadata_path)

    load.assert_called_once_with(metadata_path, weights_only=False)
    save.assert_not_called()


def test_update_failure_never_writes_aggregate(tmp_path, monkeypatch):
    update_all_clocks = _load_update_all_clocks_module()
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    weight_path = weights_dir / "clock.pt"
    weight_path.touch()
    metadata_path = tmp_path / "all_clock_metadata.pt"
    metadata_path.touch()
    load = Mock(
        side_effect=[
            {"clock": {"clock_name": "clock", "notes": "Curated notes"}},
            _clock("clock"),
            _clock("clock"),
        ]
    )
    save = Mock(side_effect=OSError("disk full"))
    monkeypatch.setattr(update_all_clocks.torch, "load", load)
    monkeypatch.setattr(update_all_clocks.torch, "save", save)

    with pytest.raises(OSError, match="disk full"):
        update_all_clocks.regenerate_clock_metadata("0.3.0", weights_dir=weights_dir, metadata_path=metadata_path)

    assert save.call_count == 1
    assert save.call_args.args[1] == weight_path


def test_generated_clock_set_must_match_weight_filenames_before_aggregate_save(tmp_path, monkeypatch):
    update_all_clocks = _load_update_all_clocks_module()
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    weight_path = weights_dir / "clock.pt"
    weight_path.touch()
    metadata_path = tmp_path / "all_clock_metadata.pt"
    metadata_path.touch()
    load = Mock(
        side_effect=[
            {"clock": {"clock_name": "clock"}},
            _clock("different"),
            _clock("different"),
        ]
    )
    save = Mock()
    monkeypatch.setattr(update_all_clocks.torch, "load", load)
    monkeypatch.setattr(update_all_clocks.torch, "save", save)

    with pytest.raises(ValueError, match="Generated clock names do not match weight filenames"):
        update_all_clocks.regenerate_clock_metadata("0.3.0", weights_dir=weights_dir, metadata_path=metadata_path)

    save.assert_not_called()
    assert load.call_count == 2
