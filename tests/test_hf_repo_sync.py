"""Sidecar selection for the per-clock Hugging Face repo sync.

Only the file-selection logic is covered here; the upload itself needs the Hub.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("hf_repo_sync", ROOT / "clocks" / "hf_repo_sync.py")
hf_repo_sync = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hf_repo_sync)


def test_sidecar_assets_are_prefix_scoped_to_one_clock(tmp_path):
    for name in (
        "tage_gene_mapping.csv.gz",
        "tage_extra_lookup.csv.gz",
        "tagemortality_gene_mapping.csv.gz",
        "tage.pt",
        "tage_notes.txt",
    ):
        tmp_path.joinpath(name).write_bytes(b"")

    assert [path.name for path in hf_repo_sync._sidecar_assets("tage", tmp_path)] == [
        "tage_extra_lookup.csv.gz",
        "tage_gene_mapping.csv.gz",
    ]
    assert [path.name for path in hf_repo_sync._sidecar_assets("tagemortality", tmp_path)] == [
        "tagemortality_gene_mapping.csv.gz"
    ]
    assert hf_repo_sync._sidecar_assets("horvath2013", tmp_path) == []


@pytest.mark.skipif(
    not (ROOT / "clocks" / "weights" / "tage_gene_mapping.csv.gz").exists(),
    reason="mapping asset not built locally (clocks/weights/ is gitignored)",
)
def test_tage_gene_mapping_is_shipped_as_a_tage_sidecar():
    """The preprocessing downloads this asset from ``pyaging/tage``, so the sync must upload it."""
    assert [path.name for path in hf_repo_sync._sidecar_assets("tage")] == ["tage_gene_mapping.csv.gz"]
