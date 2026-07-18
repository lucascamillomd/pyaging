import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.online


def test_public_hf_repository_serves_small_runtime_assets(tmp_path):
    hf_home = tmp_path / "hf-home"
    env = os.environ.copy()
    for name in (
        "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
        "ACTIONS_ID_TOKEN_REQUEST_URL",
        "HF_ENDPOINT",
        "HF_OIDC_RESOURCE",
        "HF_STORED_TOKENS_PATH",
        "HF_TOKEN",
        "HF_TOKEN_PATH",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGINGFACE_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "PYTHONOPTIMIZE",
    ):
        env.pop(name, None)
    env.update(
        {
            "HF_HOME": str(hf_home),
            "HF_HUB_CACHE": str(hf_home / "hub"),
            "HUGGINGFACE_HUB_CACHE": str(hf_home / "hub"),
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HF_HUB_OFFLINE": "0",
            "HF_XET_CACHE": str(hf_home / "xet"),
        }
    )

    code = """
import os

import torch

from pyaging.utils._hf import download_hf_file

data_dir = os.environ["PYAGING_SMOKE_DATA_DIR"]
metadata_path = download_hf_file("all_clock_metadata.pt", data_dir)
clock_path = download_hf_file("horvath2013.pt", data_dir)
example_path = download_hf_file("blood_chemistry_example.pkl", data_dir)

assert os.path.basename(metadata_path) == "all_clock_metadata.pt"
assert os.path.basename(clock_path) == "horvath2013.pt"
assert os.path.basename(example_path) == "blood_chemistry_example.pkl"
assert not os.path.exists(data_dir)
assert all(os.path.commonpath([path, os.environ["HF_HUB_CACHE"]]) == os.environ["HF_HUB_CACHE"]
           for path in (metadata_path, clock_path, example_path))
assert "horvath2013" in torch.load(metadata_path, weights_only=False)
assert torch.load(clock_path, weights_only=False).metadata["clock_name"].lower() == "horvath2013"
"""
    env["PYAGING_SMOKE_DATA_DIR"] = str(tmp_path / "data")
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
