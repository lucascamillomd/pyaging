import sys
from pathlib import Path

import pytest
import torch

CLOCKS_DIR = Path(__file__).resolve().parents[1] / "clocks"
sys.path.insert(0, str(CLOCKS_DIR))

import patch_clocks_v050  # noqa: E402
from patch_clocks_v050 import main, patch_weight_file, rename_features  # noqa: E402


def _save_fake_clock(path, features):
    model = torch.nn.Module()
    model.features = features
    model.metadata = {"clock_name": path.stem, "data_type": "DNA methylation"}
    torch.save(model, path)
    return path


def test_rename_features_maps_legacy_names():
    assert rename_features(["cg001", "Female", "Age", "GrimAge"]) == [
        "cg001",
        "female",
        "age",
        "GrimAge",
    ]


def test_rename_features_leaves_already_correct_names_alone():
    assert rename_features(["female", "age"]) == ["female", "age"]


def test_rename_features_rejects_a_collision():
    with pytest.raises(ValueError, match="already present"):
        rename_features(["Female", "female"])


def test_patch_weight_file_renames_and_reports_change(tmp_path):
    model = torch.nn.Module()
    model.features = ["cg001", "Female", "Age"]
    model.metadata = {"clock_name": "fake", "data_type": "DNA methylation"}
    path = tmp_path / "fake.pt"
    torch.save(model, path)

    assert patch_weight_file(path) is True
    assert torch.load(path, weights_only=False).features == ["cg001", "female", "age"]


def test_patch_weight_file_is_idempotent(tmp_path):
    model = torch.nn.Module()
    model.features = ["female", "age"]
    model.metadata = {"clock_name": "fake", "data_type": "clinical biomarkers"}
    path = tmp_path / "fake.pt"
    torch.save(model, path)

    patch_weight_file(path)
    assert patch_weight_file(path) is False


def test_rename_features_rejects_a_repeated_legacy_name():
    with pytest.raises(ValueError, match="already present"):
        rename_features(["Female", "Female"])


def test_patch_weight_file_rejects_a_model_without_features(tmp_path):
    model = torch.nn.Module()
    model.metadata = {"clock_name": "fake", "data_type": "DNA methylation"}
    path = tmp_path / "fake.pt"
    torch.save(model, path)

    with pytest.raises(ValueError, match="no 'features' list"):
        patch_weight_file(path)


def test_patch_weight_file_leaves_no_temporary_files_behind(tmp_path):
    path = _save_fake_clock(tmp_path / "fake.pt", ["cg001", "Female", "Age"])

    assert patch_weight_file(path) is True
    assert [p.name for p in tmp_path.iterdir()] == ["fake.pt"]


def test_patch_weight_file_keeps_the_original_when_the_save_fails(tmp_path, monkeypatch):
    path = _save_fake_clock(tmp_path / "fake.pt", ["cg001", "Female", "Age"])
    original_bytes = path.read_bytes()

    def failing_save(obj, target, *args, **kwargs):
        Path(target).write_bytes(b"truncated")  # a real partial write, not a clean no-op
        raise OSError("disk full")

    monkeypatch.setattr(patch_clocks_v050.torch, "save", failing_save)
    with pytest.raises(OSError, match="disk full"):
        patch_weight_file(path)

    assert path.read_bytes() == original_bytes
    assert [p.name for p in tmp_path.iterdir()] == ["fake.pt"]


def test_main_reports_scanned_and_patched_counts(tmp_path, capsys, monkeypatch):
    _save_fake_clock(tmp_path / "stale.pt", ["cg001", "Female", "Age"])
    _save_fake_clock(tmp_path / "current.pt", ["cg001", "female", "age"])
    monkeypatch.setattr(sys, "argv", ["patch_clocks_v050.py", str(tmp_path)])

    assert main() == 0
    assert capsys.readouterr().out.strip() == "scanned 2 clocks, patched 1: stale.pt"


def test_main_reports_none_when_every_clock_is_already_current(tmp_path, capsys, monkeypatch):
    _save_fake_clock(tmp_path / "current.pt", ["cg001", "female", "age"])
    monkeypatch.setattr(sys, "argv", ["patch_clocks_v050.py", str(tmp_path)])

    assert main() == 0
    assert capsys.readouterr().out.strip() == "scanned 1 clocks, patched 0: none"


def test_main_fails_on_a_directory_with_no_clocks(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["patch_clocks_v050.py", str(tmp_path)])

    assert main() == 1
    assert "no .pt files found" in capsys.readouterr().err
