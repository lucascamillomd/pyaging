import sys
from pathlib import Path

import pytest
import torch

CLOCKS_DIR = Path(__file__).resolve().parents[1] / "clocks"
sys.path.insert(0, str(CLOCKS_DIR))

from patch_clocks_v050 import patch_weight_file, rename_features  # noqa: E402


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
