import copy
import json
from pathlib import Path

import pytest

from clocks.metadata._audit_tools import (
    assign_families,
    build_manifest,
    main,
    normalize_doi,
    validate_shard,
)
from clocks.metadata.validate_metadata import AUDITED_FIELDS, validate_audited_value

ROOT = Path(__file__).resolve().parents[1]


def test_audit_report_has_exact_fixed_scaffold():
    expected = """# Clock Metadata Source Audit

## Scope
173 clocks across 71 DOI families.

## Controlled-vocabulary decisions

## Access issues

## Source contradictions and adjudications

## Changed-value summary

## Validation

## Hugging Face publication
"""
    assert (ROOT / "clocks/metadata/audit_report.md").read_text(encoding="utf-8") == expected


def registry_record(clock_name, doi):
    return {
        "clock_name": clock_name,
        "doi": doi if doi.casefold().startswith("https://doi.org/") else f"https://doi.org/{doi}",
        "data_type": "methylation",
        "species": "Homo sapiens",
        "year": 2020,
        "citation": "Example citation",
        "notes": "Example note",
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
            "https://doi.org/10.18632/aging.101414",
        ),
        ("https://DOI.org/10.1234%2Fencoded", "https://doi.org/10.1234/encoded"),
    ],
)
def test_normalize_doi(value, expected):
    assert normalize_doi(value) == expected


def test_normalize_doi_is_lowercase_idempotent_and_merges_case_collisions():
    normalized = normalize_doi("HTTPS://DOI.ORG/10.1234/AbC")
    assert normalized == "https://doi.org/10.1234/abc"
    assert normalize_doi(normalized) == normalized
    assignments = assign_families(
        {
            "a": registry_record("a", "10.1234/ABC"),
            "b": registry_record("b", "https://doi.org/10.1234/abc"),
        },
        1,
    )
    assert assignments[0]["families"] == [{"doi": "https://doi.org/10.1234/abc", "clock_names": ["a", "b"]}]


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        10,
        "",
        "   ",
        "not-a-doi",
        "foo/bar",
        "10.x/foo",
        "doi:10.1234/x",
        "http://doi.org/10.1234/x",
        "https://example.com/10.1234/x",
        "https://doi.org/foo%2Fbar",
        "https://doi.org/%2F",
        "https://doi.org/10.1234/",
        "https://doi.org/10.1234/x?download=1",
        "https://doi.org/10.1234/x#section",
        "https://doi.org/10.1234%2Fx%3Fdownload",
        "10.1234/white space",
        "10.123/x",
        "10.1234567890/x",
        "https://doi.org/10.1234%252Fdouble",
    ],
)
def test_normalize_doi_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="DOI"):
        normalize_doi(value)


def test_assign_families_is_deterministic_greedy_and_never_splits_doi():
    registry = {
        "f": registry_record("f", "10.1000/d"),
        "e": registry_record("e", "10.1000/c"),
        "d": registry_record("d", "10.1000/b"),
        "c": registry_record("c", "10.1000/b"),
        "b": registry_record("b", "10.1000/a"),
        "a": registry_record("a", "10.1000/a"),
    }

    assignments = assign_families(registry, 3)

    assert assignments == [
        {
            "batch": "01",
            "clock_count": 2,
            "paper_count": 1,
            "families": [{"doi": "https://doi.org/10.1000/a", "clock_names": ["a", "b"]}],
        },
        {
            "batch": "02",
            "clock_count": 2,
            "paper_count": 1,
            "families": [{"doi": "https://doi.org/10.1000/b", "clock_names": ["c", "d"]}],
        },
        {
            "batch": "03",
            "clock_count": 2,
            "paper_count": 2,
            "families": [
                {"doi": "https://doi.org/10.1000/c", "clock_names": ["e"]},
                {"doi": "https://doi.org/10.1000/d", "clock_names": ["f"]},
            ],
        },
    ]
    assert {family["doi"] for batch in assignments for family in batch["families"]} == {
        "https://doi.org/10.1000/a",
        "https://doi.org/10.1000/b",
        "https://doi.org/10.1000/c",
        "https://doi.org/10.1000/d",
    }


@pytest.mark.parametrize("batch_count", [True, False, 0, -1, 1.0, "2"])
def test_assign_families_rejects_invalid_batch_count(batch_count):
    with pytest.raises(ValueError, match="batch_count"):
        assign_families({}, batch_count)


def test_build_manifest_writes_exact_deterministic_shapes(tmp_path):
    registry = {
        "alpha": registry_record("alpha", "10.1000/shared"),
        "beta": registry_record("beta", "https://doi.org/10.1000/shared"),
        "gamma": registry_record("gamma", "10.1000/solo"),
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
                "doi": "https://doi.org/10.1000/shared",
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


def test_build_manifest_refuses_nonempty_output_without_changing_bytes(tmp_path):
    registry_path = tmp_path / "registry.json"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    existing = output_dir / "keep.bin"
    existing.write_bytes(b"\x00do not replace\xff")
    nested = output_dir / "keep-dir"
    nested.mkdir()
    write_json(registry_path, {"alpha": registry_record("alpha", "10.1234/a")})

    with pytest.raises(ValueError, match="output.*empty"):
        build_manifest(registry_path, output_dir, batch_count=1)

    assert existing.read_bytes() == b"\x00do not replace\xff"
    assert list(nested.iterdir()) == []
    assert sorted(path.name for path in output_dir.iterdir()) == ["keep-dir", "keep.bin"]


def test_build_manifest_accepts_existing_empty_output_directory(tmp_path):
    registry_path = tmp_path / "registry.json"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    write_json(registry_path, {"alpha": registry_record("alpha", "10.1234/a")})

    manifest = build_manifest(registry_path, output_dir, batch_count=1)

    assert manifest["clock_count"] == 1
    assert sorted(path.name for path in output_dir.iterdir()) == ["batch-01.json", "manifest.json"]


def test_build_manifest_validates_before_writing_any_output(tmp_path):
    registry_path = tmp_path / "registry.json"
    output_dir = tmp_path / "out"
    write_json(registry_path, {"alpha": {"clock_name": "alpha", "doi": "invalid"}})

    with pytest.raises(ValueError, match="alpha.*DOI"):
        build_manifest(registry_path, output_dir)

    assert not output_dir.exists()


def make_batch_and_shard(tmp_path):
    alpha = registry_record("alpha", "https://doi.org/10.1000/a")
    beta = registry_record("beta", "https://doi.org/10.1000/b")
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


def test_validate_shard_allows_citation_shape_transitions(tmp_path):
    batch_path, shard_path, shard = make_batch_and_shard(tmp_path)
    batch = json.loads(batch_path.read_text())
    batch["papers"][1]["current_metadata"]["beta"]["citation"] = ["Old citation"]
    shard["records"][0]["fields"]["citation"]["value"] = ["New citation", "Second citation"]
    shard["records"][1]["fields"]["citation"]["value"] = "Replacement citation"
    write_json(batch_path, batch)
    write_json(shard_path, shard)

    validate_shard(batch_path, shard_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tissue", []),
        ("tissue", ["blood", "blood"]),
        ("tissue", [""]),
        ("tissue", [{}]),
        ("data_type", ""),
        ("journal", "  "),
        ("year", True),
        ("year", 2020.0),
        ("n_features", False),
        ("citation", ""),
        ("citation", []),
        ("citation", ["same", "same"]),
        ("citation", ["valid", ""]),
        ("citation", {"text": "invalid"}),
        ("doi", "https://doi.org/not-a-doi"),
        ("notes", None),
        ("notes", ""),
    ],
)
def test_validate_audited_value_rejects_invalid_shapes(field, value):
    with pytest.raises(ValueError, match=f"example.{field}"):
        validate_audited_value(field, value, f"example.{field}")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tissue", ["blood", "saliva"]),
        ("data_type", "methylation"),
        ("journal", "Journal"),
        ("year", 2020),
        ("n_features", 1),
        ("citation", "One citation"),
        ("citation", ["One citation", "Two citations"]),
        ("doi", "https://doi.org/10.1234/valid"),
        ("notes", "Canonical note"),
    ],
)
def test_validate_audited_value_accepts_canonical_shapes(field, value):
    validate_audited_value(field, value, f"example.{field}")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tissue", []),
        ("tissue", ["blood", "blood"]),
        ("data_type", " "),
        ("year", True),
        ("citation", []),
        ("doi", "bad DOI"),
        ("notes", None),
    ],
)
def test_validate_shard_uses_shared_audited_value_contract(tmp_path, field, value):
    batch_path, shard_path, shard = make_batch_and_shard(tmp_path)
    shard["records"][0]["fields"][field]["value"] = value
    write_json(shard_path, shard)

    with pytest.raises(ValueError, match=f"alpha.{field}.value"):
        validate_shard(batch_path, shard_path)


def test_validate_shard_rejects_paper_with_no_assigned_clocks(tmp_path):
    batch_path, shard_path, shard = make_batch_and_shard(tmp_path)
    batch = json.loads(batch_path.read_text())
    batch["papers"] = [
        {
            "doi": "https://doi.org/10.1234/empty",
            "clock_names": [],
            "current_metadata": {},
            "audited_fields": list(AUDITED_FIELDS),
        }
    ]
    batch["paper_count"] = 1
    batch["clock_count"] = 0
    shard["records"] = []
    write_json(batch_path, batch)
    write_json(shard_path, shard)

    with pytest.raises(ValueError, match="clock_names.*nonempty"):
        validate_shard(batch_path, shard_path)


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
        (lambda s: s["records"][0].__setitem__("doi", "https://doi.org/10.1000/wrong"), "DOI"),
    ],
)
def test_validate_shard_rejects_representative_invalid_cases(tmp_path, mutation, message):
    batch_path, shard_path, shard = make_batch_and_shard(tmp_path)
    mutation(shard)
    write_json(shard_path, shard)

    with pytest.raises(ValueError, match=message):
        validate_shard(batch_path, shard_path)


def test_cli_reports_expected_errors_without_traceback(tmp_path, capsys):
    registry_path = tmp_path / "invalid.json"
    write_json(registry_path, {"alpha": {"doi": "invalid"}})

    exit_code = main(
        [
            "build-manifest",
            "--registry",
            str(registry_path),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert "Traceback" not in captured.err


def test_real_registry_manifest_is_complete_unique_and_byte_deterministic(tmp_path):
    registry_path = ROOT / "clocks/metadata/clock_metadata.json"
    output_one = tmp_path / "one"
    output_two = tmp_path / "two"

    first = build_manifest(registry_path, output_one, batch_count=12)
    second = build_manifest(registry_path, output_two, batch_count=12)

    assert first == second
    assert first["paper_count"] == 71
    assert first["clock_count"] == 173
    assert len(first["batches"]) == 12
    dois = []
    clocks = []
    for descriptor in first["batches"]:
        assert descriptor["paper_count"] > 0
        assert descriptor["clock_count"] > 0
        batch = json.loads((output_one / descriptor["path"]).read_text())
        dois.extend(paper["doi"] for paper in batch["papers"])
        clocks.extend(clock for paper in batch["papers"] for clock in paper["clock_names"])
    assert len(dois) == len(set(dois)) == 71
    assert len(clocks) == len(set(clocks)) == 173
    assert sorted(path.name for path in output_one.iterdir()) == sorted(path.name for path in output_two.iterdir())
    for path in output_one.iterdir():
        assert path.read_bytes() == (output_two / path.name).read_bytes()
