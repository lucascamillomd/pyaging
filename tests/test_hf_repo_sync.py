"""Local validation and sidecar selection for per-clock Hugging Face sync."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("hf_repo_sync", ROOT / "clocks" / "hf_repo_sync.py")
hf_repo_sync = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hf_repo_sync)


@pytest.mark.parametrize("create_weights_directory", [False, True], ids=["missing", "empty"])
@pytest.mark.parametrize("arguments", [[], ["--tag", "v0.5.3", "--tag-only"]], ids=["sync", "tag-only"])
def test_sync_cli_rejects_empty_catalog_before_contacting_hub(
    tmp_path, monkeypatch, capsys, create_weights_directory, arguments
):
    weights = tmp_path / "weights"
    if create_weights_directory:
        weights.mkdir()
    metadata = tmp_path / "clock_metadata.json"
    metadata.write_text(json.dumps({"horvath2013": {"clock_name": "horvath2013"}}))
    monkeypatch.setattr(hf_repo_sync, "WEIGHTS_DIR", weights)
    monkeypatch.setattr(hf_repo_sync, "METADATA_FILE", metadata)
    monkeypatch.setattr("sys.argv", ["hf_repo_sync.py", *arguments])
    hub_clients = []

    def create_hub_client():
        hub_clients.append(object())
        return hub_clients[-1]

    monkeypatch.setattr(hf_repo_sync, "HfApi", create_hub_client)

    with pytest.raises(SystemExit) as error:
        hf_repo_sync.main()

    assert error.value.code == 2
    assert "non-empty weights directory" in capsys.readouterr().err
    assert hub_clients == []


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
