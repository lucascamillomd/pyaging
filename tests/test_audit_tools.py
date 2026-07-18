import copy
import json
from pathlib import Path

import pytest

from clocks.metadata._audit_tools import (
    assign_families,
    build_manifest,
    normalize_doi,
    validate_shard,
)
from clocks.metadata.validate_metadata import AUDITED_FIELDS


def registry_record(clock_name, doi):
    return {
        "clock_name": clock_name,
        "doi": doi,
        "data_type": "methylation",
        "species": "Homo sapiens",
        "year": 2020,
        "citation": "Example citation",
        "notes": "",
        "tissue": ["blood"],
        "predicts": ["age"],
        "training_target": ["age"],
        "unit": ["years"],
        "model_type": "Elastic net",
        "platform": ["Illumina 450K"],
        "population": "adults",
        "journal": "Example Journal",
        "last_author": "Example Author",
        "n_features": 10,
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("10.1000/example", "https://doi.org/10.1000/example"),
        (" https://doi.org/10.1000/example ", "https://doi.org/10.1000/example"),
        (
            "https://doi.org/10.18632%2Faging.101414",
            "https://doi.org/10.18632%2Faging.101414",
        ),
    ],
)
def test_normalize_doi(value, expected):
    assert normalize_doi(value) == expected


@pytest.mark.parametrize(
    "value",
    [None, True, 10, "", "   ", "not-a-doi", "http://doi.org/10.1/x", "https://example.com/10.1/x"],
)
def test_normalize_doi_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="DOI"):
        normalize_doi(value)


def test_assign_families_is_deterministic_greedy_and_never_splits_doi():
    registry = {
        "f": registry_record("f", "10.1/d"),
        "e": registry_record("e", "10.1/c"),
        "d": registry_record("d", "10.1/b"),
        "c": registry_record("c", "10.1/b"),
        "b": registry_record("b", "10.1/a"),
        "a": registry_record("a", "10.1/a"),
    }

    assignments = assign_families(registry, 3)

    assert assignments == [
        {
            "batch": "01",
            "clock_count": 2,
            "paper_count": 1,
            "families": [{"doi": "https://doi.org/10.1/a", "clock_names": ["a", "b"]}],
        },
        {
            "batch": "02",
            "clock_count": 2,
            "paper_count": 1,
            "families": [{"doi": "https://doi.org/10.1/b", "clock_names": ["c", "d"]}],
        },
        {
            "batch": "03",
            "clock_count": 2,
            "paper_count": 2,
            "families": [
                {"doi": "https://doi.org/10.1/c", "clock_names": ["e"]},
                {"doi": "https://doi.org/10.1/d", "clock_names": ["f"]},
            ],
        },
    ]
    assert {family["doi"] for batch in assignments for family in batch["families"]} == {
        "https://doi.org/10.1/a",
        "https://doi.org/10.1/b",
        "https://doi.org/10.1/c",
        "https://doi.org/10.1/d",
    }


@pytest.mark.parametrize("batch_count", [True, False, 0, -1, 1.0, "2"])
def test_assign_families_rejects_invalid_batch_count(batch_count):
    with pytest.raises(ValueError, match="batch_count"):
        assign_families({}, batch_count)


def test_build_manifest_writes_exact_deterministic_shapes(tmp_path):
    registry = {
        "alpha": registry_record("alpha", "10.1/shared"),
        "beta": registry_record("beta", "https://doi.org/10.1/shared"),
        "gamma": registry_record("gamma", "10.1/solo"),
    }
    registry_path = tmp_path / "registry.json"
    output_dir = tmp_path / "out"
    write_json(registry_path, registry)

    manifest = build_manifest(registry_path, output_dir, batch_count=2)

    expected_manifest = {
        "schema_version": 1,
        "paper_count": 2,
        "clock_count": 3,
        "batches": [
            {"batch": "01", "paper_count": 1, "clock_count": 2, "path": "batch-01.json"},
            {"batch": "02", "paper_count": 1, "clock_count": 1, "path": "batch-02.json"},
        ],
    }
    assert manifest == expected_manifest
    assert json.loads((output_dir / "manifest.json").read_text()) == expected_manifest
    batch = json.loads((output_dir / "batch-01.json").read_text())
    assert batch == {
        "schema_version": 1,
        "batch": "01",
        "paper_count": 1,
        "clock_count": 2,
        "papers": [
            {
                "doi": "https://doi.org/10.1/shared",
                "clock_names": ["alpha", "beta"],
                "current_metadata": {
                    "alpha": registry["alpha"],
                    "beta": registry["beta"],
                },
                "audited_fields": list(AUDITED_FIELDS),
            }
        ],
    }
    for path in output_dir.iterdir():
        assert path.read_bytes().endswith(b"\n")
    before = {path.name: path.read_bytes() for path in output_dir.iterdir()}
    build_manifest(registry_path, output_dir, batch_count=2)
    assert {path.name: path.read_bytes() for path in output_dir.iterdir()} == before


def test_build_manifest_validates_before_writing_any_output(tmp_path):
    registry_path = tmp_path / "registry.json"
    output_dir = tmp_path / "out"
    write_json(registry_path, {"alpha": {"clock_name": "alpha", "doi": "invalid"}})

    with pytest.raises(ValueError, match="alpha.*DOI"):
        build_manifest(registry_path, output_dir)

    assert not output_dir.exists()


def make_batch_and_shard(tmp_path):
    alpha = registry_record("alpha", "https://doi.org/10.1/a")
    beta = registry_record("beta", "https://doi.org/10.1/b")
    batch = {
        "schema_version": 1,
        "batch": "01",
        "paper_count": 2,
        "clock_count": 2,
        "papers": [
            {
                "doi": alpha["doi"],
                "clock_names": ["alpha"],
                "current_metadata": {"alpha": alpha},
                "audited_fields": list(AUDITED_FIELDS),
            },
            {
                "doi": beta["doi"],
                "clock_names": ["beta"],
                "current_metadata": {"beta": beta},
                "audited_fields": list(AUDITED_FIELDS),
            },
        ],
    }

    def record(name, current):
        return {
            "clock_name": name,
            "doi": current["doi"],
            "reviewer": "paper-audit-01",
            "sources": [
                {
                    "id": "paper",
                    "type": "paper",
                    "url": current["doi"],
                    "accessed": "2026-07-18",
                }
            ],
            "fields": {
                field: {
                    "value": copy.deepcopy(current[field]),
                    "source_text": "unresolved",
                    "source_id": "paper",
                    "locator": "pending source audit",
                    "status": "unresolved",
                    "note": "",
                }
                for field in AUDITED_FIELDS
            },
            "access_issues": [],
        }

    shard = {
        "schema_version": 1,
        "batch": "01",
        "reviewer": "paper-audit-01",
        "records": [record("alpha", alpha), record("beta", beta)],
    }
    batch_path = tmp_path / "batch-01.json"
    shard_path = tmp_path / "shard-01.json"
    write_json(batch_path, batch)
    write_json(shard_path, shard)
    return batch_path, shard_path, shard


def test_validate_shard_accepts_unresolved_and_resolved_evidence(tmp_path):
    batch_path, shard_path, shard = make_batch_and_shard(tmp_path)
    evidence = shard["records"][0]["fields"]["year"]
    evidence.update(
        {
            "source_text": "Published in 2020",
            "locator": "page 1",
            "status": "paper-confirmed",
        }
    )
    write_json(shard_path, shard)

    summary = validate_shard(batch_path, shard_path)

    assert summary == {"batch": "01", "paper_count": 2, "clock_count": 2}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda s: s["records"].pop(), "clock set mismatch"),
        (
            lambda s: s["records"].append({**copy.deepcopy(s["records"][1]), "clock_name": "extra"}),
            "clock set mismatch",
        ),
        (lambda s: s.__setitem__("reviewer", "someone-else"), "reviewer"),
        (lambda s: s.__setitem__("batch", "02"), "batch"),
        (lambda s: s["records"].reverse(), "alphabetical"),
        (lambda s: s["records"][0]["fields"].pop("year"), "fields"),
        (
            lambda s: s["records"][0]["fields"].__setitem__("extra", copy.deepcopy(s["records"][0]["fields"]["year"])),
            "fields",
        ),
        (lambda s: s["records"][0]["fields"]["year"].__setitem__("value", "2020"), "value"),
        (
            lambda s: s["records"][0]["fields"]["year"].__setitem__("status", "paper-confirmed"),
            "resolved evidence",
        ),
        (
            lambda s: s["records"][0]["fields"]["year"].update(
                {
                    "status": "code-confirmed",
                    "source_text": "2020",
                    "locator": "line 1",
                }
            ),
            "requires source type",
        ),
        (lambda s: s["records"][0].__setitem__("doi", "https://doi.org/10.1/wrong"), "DOI"),
    ],
)
def test_validate_shard_rejects_representative_invalid_cases(tmp_path, mutation, message):
    batch_path, shard_path, shard = make_batch_and_shard(tmp_path)
    mutation(shard)
    write_json(shard_path, shard)

    with pytest.raises(ValueError, match=message):
        validate_shard(batch_path, shard_path)
