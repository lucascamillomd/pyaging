import ast
import copy
import hashlib
import json
import os
import weakref
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import clocks.metadata._audit_tools as audit_tools
from clocks.metadata._audit_tools import (
    apply_vocabulary_decisions,
    assign_families,
    build_manifest,
    collapse_source_text,
    discover_metadata_cell,
    fingerprint_weights,
    main,
    materialize,
    merge_shards,
    migrate_dry_run,
    model_fingerprint,
    normalize_doi,
    normalize_merged,
    normalize_runtime_value,
    render_metadata_lines,
    tensor_digest,
    validate_shard,
    vocabulary_report,
)
from clocks.metadata.validate_metadata import (
    ADMIN_FIELDS,
    ARRAY_FIELDS,
    AUDITED_FIELDS,
    CONTROLLED_SCALAR_FIELDS,
    validate_audited_value,
)

ROOT = Path(__file__).resolve().parents[1]


class TinyFingerprintModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        self.features = ["b", "a"]
        self.base_model_features = ("x", "y")
        self.reference_values = np.array([1.0, 2.0])
        self.preprocess_name = None
        self.preprocess_dependencies = {"z": pd.Index(["a", "b"])}
        self.postprocess_name = "identity"
        self.postprocess_dependencies = pd.Series([3, 4])
        self.version = "v1"
        self.metadata = {"clock_name": "tiny", "obsolete": "remove"}


def test_audit_tools_do_not_use_python_310_zip_strict_keyword():
    tree = ast.parse(Path(audit_tools.__file__).read_text(encoding="utf-8"))
    incompatible_calls = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "zip"
        and any(keyword.arg == "strict" for keyword in node.keywords)
    ]
    assert incompatible_calls == []


def test_audit_report_preserves_fixed_scaffold():
    report = (ROOT / "clocks/metadata/audit_report.md").read_text(encoding="utf-8")
    assert report.startswith("# Clock Metadata Source Audit\n")
    for heading in (
        "## Scope",
        "## Controlled-vocabulary decisions",
        "## Access issues",
        "## Source contradictions and adjudications",
        "## Changed-value summary",
        "## Validation",
        "## Hugging Face publication",
    ):
        assert heading in report
    assert "173 clocks across 71 DOI families." in report


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


def complete_registry_record(clock_name="tiny"):
    return {
        "clock_name": clock_name,
        "data_type": "DNA methylation",
        "species": "Homo sapiens",
        "year": 2020,
        "approved_by_author": "⌛",
        "citation": "A citation",
        "doi": "https://doi.org/10.1000/example",
        "notes": "A note",
        "research_only": None,
        "tissue": ["whole blood"],
        "predicts": ["chronological age"],
        "training_target": ["chronological age"],
        "unit": ["years"],
        "model_type": "elastic net regression",
        "platform": ["Illumina 450K"],
        "population": "adults",
        "journal": "Journal",
        "last_author": "Author",
        "n_features": 2,
        "citations": 4,
        "citations_date": "2026-07-18",
    }


def complete_ledger_record(clock_name="tiny"):
    record = complete_registry_record(clock_name)
    return {
        "clock_name": clock_name,
        "doi": record["doi"],
        "reviewer": "paper-audit-01",
        "sources": [
            {
                "id": "paper",
                "type": "paper",
                "url": record["doi"],
                "accessed": "2026-07-18",
            }
        ],
        "fields": {
            field: {
                "value": record[field],
                "source_text": f"Exact {field}\n wording\tfrom paper.",
                "source_id": "paper",
                "locator": "Methods",
                "status": "paper-confirmed",
                "note": "",
            }
            for field in AUDITED_FIELDS
        },
        "access_issues": [],
    }


def notebook_with_metadata(clock_name="tiny", duplicate=False):
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            f'model.metadata["clock_name"] = "{clock_name}"\n',
            'model.metadata["data_type"] = "old"\n',
        ],
    }
    cells = [copy.deepcopy(cell)]
    if duplicate:
        cells.append(copy.deepcopy(cell))
    return {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def test_collapse_source_text_makes_control_characters_one_readable_line():
    assert collapse_source_text("  exact\npaper\r\nwording\twith\x00 controls  ") == (
        "exact paper wording with controls"
    )


def test_discover_metadata_cell_finds_assignment_semantically():
    notebook = notebook_with_metadata()
    notebook["cells"].insert(
        0,
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["%load_ext autoreload\n"],
        },
    )
    notebook["cells"][1]["source"][0] = "model.metadata [ 'clock_name' ] = 'tiny'\n"

    assert discover_metadata_cell(notebook, "tiny.ipynb") == 1


@pytest.mark.parametrize(
    ("notebook", "message"),
    [
        ({"cells": []}, "zero metadata cells"),
        (notebook_with_metadata(duplicate=True), "multiple metadata cells"),
    ],
)
def test_discover_metadata_cell_rejects_zero_or_multiple_cells(notebook, message):
    with pytest.raises(ValueError, match=message):
        discover_metadata_cell(notebook, "tiny.ipynb")


def test_render_metadata_lines_renders_every_field_and_controlled_comments():
    record = complete_registry_record()
    evidence = complete_ledger_record()["fields"]

    lines = render_metadata_lines(record, evidence)

    assert len(lines) == len(record)
    assert lines[0] == 'model.metadata["clock_name"] = "tiny"\n'
    assert (
        'model.metadata["tissue"] = ["whole blood"]'
        "  # Paper: Exact tissue wording from paper.\n"
    ) in lines
    assert (
        'model.metadata["model_type"] = "elastic net regression"'
        "  # Paper: Exact model_type wording from paper.\n"
    ) in lines
    assert 'model.metadata["research_only"] = None\n' in lines
    for field in ARRAY_FIELDS:
        line = next(item for item in lines if f'["{field}"]' in item)
        assert ast.literal_eval(line.split(" = ", 1)[1].split("  #", 1)[0]) == record[field]
    assert all("\r" not in line and "\x00" not in line for line in lines)


def test_render_metadata_lines_rejects_missing_fields_and_unresolved_evidence():
    record = complete_registry_record()
    del record["unit"]
    with pytest.raises(ValueError, match="missing curated fields"):
        render_metadata_lines(record, complete_ledger_record()["fields"])

    record = complete_registry_record()
    evidence = complete_ledger_record()["fields"]
    evidence["tissue"]["status"] = "unresolved"
    with pytest.raises(ValueError, match="tiny.tissue.*resolved"):
        render_metadata_lines(record, evidence)


def test_migrate_dry_run_reports_changes_and_preserves_all_input_artifacts(tmp_path):
    registry = {"tiny": complete_registry_record()}
    registry_path = tmp_path / "registry.json"
    write_json(registry_path, registry)
    (tmp_path / "controlled_vocabulary.json").write_bytes(
        (ROOT / "clocks/metadata/controlled_vocabulary.json").read_bytes()
    )
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_text(
        json.dumps(complete_ledger_record(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    notebooks = tmp_path / "notebooks"
    notebooks.mkdir()
    notebook_path = notebooks / "tiny.ipynb"
    write_json(notebook_path, notebook_with_metadata())
    weights = tmp_path / "weights"
    weights.mkdir()
    model = TinyFingerprintModel()
    model.metadata["clock_name"] = "tiny"
    weight_path = weights / "tiny.pt"
    torch.save(model, weight_path)
    aggregate_path = tmp_path / "all_clock_metadata.pt"
    torch.save({"tiny": copy.deepcopy(model.metadata)}, aggregate_path)
    baseline_path = tmp_path / "baseline.json"
    write_json(baseline_path, {"tiny": model_fingerprint(model)})
    report_path = tmp_path / "proposed_changes.json"
    before = {
        path: (path.read_bytes(), path.lstat(), os.readlink(path) if path.is_symlink() else None)
        for path in (notebook_path, weight_path, aggregate_path)
    }

    report = migrate_dry_run(
        registry_path,
        ledger_path,
        notebooks,
        weights,
        aggregate_path,
        baseline_path,
        report_path,
        expected_count=1,
    )

    assert report["schema_version"] == 1
    assert report["clock_count"] == 1
    assert report["dry_run"] is True
    assert report["fingerprint_verification"] == {"clock_count": 1, "matches": True}
    assert list(report["clocks"]) == ["tiny"]
    change = report["clocks"]["tiny"]
    assert set(change["curated_fields"]) == set(registry["tiny"])
    assert change["curated_fields"]["data_type"] == {
        "old": "old",
        "new": "DNA methylation",
        "changed": True,
    }
    assert change["weight_metadata"]["additions"]["training_target"] == [
        "chronological age"
    ]
    assert change["weight_metadata"]["removals"] == {"obsolete": "remove"}
    assert change["aggregate_metadata"]["additions"]
    assert len(change["notebook_lines"]) == len(registry["tiny"])
    assert change["unchanged_fields"]
    assert report_path.is_file()
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    for path, (contents, stat_result, link_target) in before.items():
        assert path.read_bytes() == contents
        assert path.lstat() == stat_result
        assert (os.readlink(path) if path.is_symlink() else None) == link_target


def test_migrate_dry_run_excludes_template_and_rejects_clock_set_mismatches(tmp_path):
    registry = {"tiny": complete_registry_record()}
    registry_path = tmp_path / "registry.json"
    write_json(registry_path, registry)
    (tmp_path / "controlled_vocabulary.json").write_bytes(
        (ROOT / "clocks/metadata/controlled_vocabulary.json").read_bytes()
    )
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_text(
        json.dumps(complete_ledger_record()) + "\n", encoding="utf-8"
    )
    notebooks = tmp_path / "notebooks"
    notebooks.mkdir()
    write_json(notebooks / "template.ipynb", notebook_with_metadata("tiny"))
    weights = tmp_path / "weights"
    weights.mkdir()

    with pytest.raises(ValueError, match="notebook clock set mismatch"):
        migrate_dry_run(
            registry_path,
            ledger_path,
            notebooks,
            weights,
            tmp_path / "aggregate.pt",
            tmp_path / "baseline.json",
            tmp_path / "report.json",
            expected_count=1,
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (True, True),
        (7, 7),
        (2.5, 2.5),
        ("x", "x"),
        ([1, "x"], [1, "x"]),
        ((1, "x"), [1, "x"]),
        ({"z": 1, "a": 2}, {"a": 2, "z": 1}),
        (np.array([[1, 2], [3, 4]]), [[1, 2], [3, 4]]),
        (np.int64(4), 4),
        (np.float32(1.5), 1.5),
        (pd.Index(["x", "y"]), ["x", "y"]),
        (pd.Series([1, 2]), [1, 2]),
        (torch.tensor([1, 2]), [1, 2]),
    ],
)
def test_normalize_runtime_value_supports_safe_runtime_types(value, expected):
    assert normalize_runtime_value(value) == expected


def test_normalize_runtime_value_intentionally_treats_tuples_as_json_lists():
    assert normalize_runtime_value(("x", 1)) == normalize_runtime_value(["x", 1])
    assert "tuple" in normalize_runtime_value.__doc__.casefold()


def test_fingerprint_normalization_encodes_distinct_nonfinite_float_sentinels():
    assert audit_tools._normalize_fingerprint_value(float("nan")) == {
        "__pyaging_runtime_type__": "nonfinite_float",
        "value": "nan",
    }
    assert audit_tools._normalize_fingerprint_value(float("inf")) == {
        "__pyaging_runtime_type__": "nonfinite_float",
        "value": "+inf",
    }
    assert audit_tools._normalize_fingerprint_value(np.float32("-inf")) == {
        "__pyaging_runtime_type__": "nonfinite_float",
        "value": "-inf",
    }


def test_fingerprint_normalization_is_deterministic_for_pasta_like_nan_list():
    values = [float("nan"), np.float64("nan"), {"z": float("nan")}]

    assert audit_tools._normalize_fingerprint_value(values) == [
        {"__pyaging_runtime_type__": "nonfinite_float", "value": "nan"},
        {"__pyaging_runtime_type__": "nonfinite_float", "value": "nan"},
        {
            "__pyaging_runtime_type__": "dict",
            "items": [
                [
                    "z",
                    {
                        "__pyaging_runtime_type__": "nonfinite_float",
                        "value": "nan",
                    },
                ]
            ],
        },
    ]
    with pytest.raises(ValueError, match="non-finite"):
        normalize_runtime_value(values)


def test_fingerprint_nonfinite_tag_cannot_collide_with_a_legitimate_runtime_dict():
    nonfinite = audit_tools._normalize_fingerprint_value(float("nan"))
    legitimate = audit_tools._normalize_fingerprint_value(
        {"__pyaging_runtime_type__": "nonfinite_float", "value": "nan"}
    )

    assert legitimate == {
        "__pyaging_runtime_type__": "dict",
        "items": [
            ["__pyaging_runtime_type__", "nonfinite_float"],
            ["value", "nan"],
        ],
    }
    assert legitimate != nonfinite


def test_migration_fingerprint_comparison_is_type_strict():
    assert audit_tools._same_fingerprint({"features": [1]}, {"features": [1]})
    assert not audit_tools._same_fingerprint({"features": [1]}, {"features": [True]})


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        np.float64("-inf"),
        np.array([1.0, np.nan]),
        np.array([{"unsafe": "pickle-capable"}], dtype=object),
        torch.tensor([float("inf")]),
        {1: "non-string key"},
        object(),
        complex(1, 2),
    ],
)
def test_normalize_runtime_value_rejects_nonfinite_and_unknown_values(value):
    with pytest.raises(ValueError):
        normalize_runtime_value(value)


def test_tensor_digest_is_deterministic_and_records_logical_tensor_identity():
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    view = source.t().t()

    digest = tensor_digest(view)

    assert digest == {
        "dtype": "torch.float32",
        "shape": [2, 2],
        "sha256": hashlib.sha256(source.detach().numpy().tobytes()).hexdigest(),
    }


def test_tensor_digest_hashes_bfloat16_raw_bytes_without_numpy_dtype_conversion():
    tensor = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)

    digest = tensor_digest(tensor)

    assert digest == {
        "dtype": "torch.bfloat16",
        "shape": [2],
        "sha256": hashlib.sha256(
            tensor.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
    }


def test_tensor_digest_hashes_zero_dimensional_multibyte_tensors():
    tensor = torch.tensor(42, dtype=torch.int64)
    expected_bytes = tensor.reshape(1).view(torch.uint8).numpy().tobytes()

    assert tensor_digest(tensor) == {
        "dtype": "torch.int64",
        "shape": [],
        "sha256": hashlib.sha256(expected_bytes).hexdigest(),
    }


def test_model_fingerprint_is_deterministic_and_excludes_metadata():
    model = TinyFingerprintModel()
    first = model_fingerprint(model)
    model.metadata["changed"] = "metadata is not prediction state"
    second = model_fingerprint(model)

    assert first == second
    assert list(first["state_dict"]) == sorted(first["state_dict"])
    assert first["class"].endswith(".TinyFingerprintModel")
    assert first["features"] == ["b", "a"]
    assert set(first) == {
        "class",
        "state_dict",
        "features",
        "base_model_features",
        "reference_values",
        "preprocess_name",
        "preprocess_dependencies",
        "postprocess_name",
        "postprocess_dependencies",
        "version",
    }
    with torch.no_grad():
        model.linear.weight.add_(1)
    assert model_fingerprint(model) != first


def test_model_fingerprint_preserves_pasta_like_nonfinite_reference_values():
    model = TinyFingerprintModel()
    model.reference_values = [float("nan")] * 3

    assert model_fingerprint(model)["reference_values"] == [
        {"__pyaging_runtime_type__": "nonfinite_float", "value": "nan"},
        {"__pyaging_runtime_type__": "nonfinite_float", "value": "nan"},
        {"__pyaging_runtime_type__": "nonfinite_float", "value": "nan"},
    ]


def test_fingerprint_weights_writes_exact_alphabetical_immutable_entries(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    for name in ("zeta", "alpha"):
        model = TinyFingerprintModel()
        model.metadata["clock_name"] = name
        torch.save(model, weights / f"{name}.pt")
    output = tmp_path / "fingerprints.json"

    result = fingerprint_weights(weights, output, expected_count=2)

    assert list(result) == ["alpha", "zeta"]
    assert json.loads(output.read_text()) == result
    with pytest.raises(ValueError, match="already exists"):
        fingerprint_weights(weights, output, expected_count=2)


def test_fingerprint_weights_rejects_filename_clock_name_mismatch_without_output(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    model = TinyFingerprintModel()
    model.metadata["clock_name"] = "duplicate-logical-name"
    torch.save(model, weights / "physical-name.pt")
    output = tmp_path / "fingerprints.json"

    with pytest.raises(ValueError, match="clock_name mismatch"):
        fingerprint_weights(weights, output, expected_count=1)

    assert not output.exists()


def test_fingerprint_weights_reports_clock_and_runtime_field_on_unsafe_value(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    model = TinyFingerprintModel()
    model.metadata["clock_name"] = "tiny"
    model.reference_values = object()
    torch.save(model, weights / "tiny.pt")
    output = tmp_path / "fingerprints.json"

    with pytest.raises(ValueError, match=r"tiny\.reference_values: unsupported"):
        fingerprint_weights(weights, output, expected_count=1)

    assert not output.exists()


def test_fingerprint_weights_preserves_symlink_lstat_and_only_keeps_one_model_live(
    tmp_path, monkeypatch
):
    weights = tmp_path / "weights"
    backing = tmp_path / "backing"
    weights.mkdir()
    backing.mkdir()
    for name in ("alpha", "beta", "gamma"):
        target = backing / f"{name}.pt"
        target.write_bytes(b"placeholder")
        (weights / f"{name}.pt").symlink_to(target)
    links_before = {
        path.name: (path.lstat().st_mode, path.lstat().st_size, path.lstat().st_mtime_ns, os.readlink(path))
        for path in weights.glob("*.pt")
    }
    live = []
    peak_live = 0

    def fake_load(path, **_kwargs):
        nonlocal peak_live
        live[:] = [reference for reference in live if reference() is not None]
        peak_live = max(peak_live, len(live))
        model = TinyFingerprintModel()
        model.metadata["clock_name"] = Path(path).stem
        live.append(weakref.ref(model))
        peak_live = max(peak_live, sum(reference() is not None for reference in live))
        return model

    monkeypatch.setattr(audit_tools.torch, "load", fake_load)
    output = tmp_path / "fingerprints.json"

    fingerprint_weights(weights, output, expected_count=3)

    links_after = {
        path.name: (path.lstat().st_mode, path.lstat().st_size, path.lstat().st_mtime_ns, os.readlink(path))
        for path in weights.glob("*.pt")
    }
    assert links_after == links_before
    assert peak_live == 1


def test_verify_fingerprints_compares_exact_sets_and_values_read_only(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    model = TinyFingerprintModel()
    model.metadata["clock_name"] = "tiny"
    torch.save(model, weights / "tiny.pt")
    baseline = tmp_path / "baseline.json"
    fingerprint_weights(weights, baseline, expected_count=1)
    before = baseline.read_bytes()

    result = audit_tools.verify_fingerprints(weights, baseline, expected_count=1)

    assert result == {"clock_count": 1, "matches": True}
    assert baseline.read_bytes() == before

    changed = json.loads(before)
    changed["tiny"]["version"] = "changed"
    mismatch = tmp_path / "mismatch.json"
    write_json(mismatch, changed)
    with pytest.raises(ValueError, match="fingerprint mismatch.*tiny"):
        audit_tools.verify_fingerprints(weights, mismatch, expected_count=1)

    changed["extra"] = changed.pop("tiny")
    set_mismatch = tmp_path / "set-mismatch.json"
    write_json(set_mismatch, changed)
    with pytest.raises(ValueError, match="clock set mismatch"):
        audit_tools.verify_fingerprints(weights, set_mismatch, expected_count=1)


@pytest.mark.parametrize("different_json_scalar", [True, 1.0])
def test_verify_fingerprints_distinguishes_int_from_bool_and_float(
    tmp_path, different_json_scalar
):
    weights = tmp_path / "weights"
    weights.mkdir()
    model = TinyFingerprintModel()
    model.metadata["clock_name"] = "tiny"
    model.version = 1
    torch.save(model, weights / "tiny.pt")
    current = model_fingerprint(model)
    current["version"] = different_json_scalar
    baseline = tmp_path / "baseline.json"
    write_json(baseline, {"tiny": current})

    with pytest.raises(ValueError, match="fingerprint mismatch.*tiny"):
        audit_tools.verify_fingerprints(weights, baseline, expected_count=1)


def test_cli_fingerprint_and_verify_fingerprints(tmp_path, capsys):
    weights = tmp_path / "weights"
    weights.mkdir()
    model = TinyFingerprintModel()
    model.metadata["clock_name"] = "tiny"
    torch.save(model, weights / "tiny.pt")
    baseline = tmp_path / "baseline.json"

    assert main(
        [
            "fingerprint",
            "--weights",
            str(weights),
            "--output",
            str(baseline),
            "--expected-count",
            "1",
        ]
    ) == 0
    assert "fingerprinted 1 clocks" in capsys.readouterr().out
    assert main(
        [
            "verify-fingerprints",
            "--weights",
            str(weights),
            "--baseline",
            str(baseline),
            "--expected-count",
            "1",
        ]
    ) == 0
    assert "verified 1 clock fingerprints" in capsys.readouterr().out


def populate_decision_snapshots(decisions, reconciled):
    coverage = {}
    observed_counts = {}
    canonical_counts = {}
    alias_counts = {}
    override_counts = {}
    for field, descriptor in decisions["decisions"].items():
        canonical = set(descriptor["canonical_values"])
        aliases = descriptor["aliases"]
        overrides = descriptor["per_clock_overrides"]
        observations = {}
        for record in reconciled["records"]:
            value = record["fields"][field]["value"]
            values = value if field in ARRAY_FIELDS else [value]
            for source_value in values:
                observations.setdefault(source_value, set()).add(record["clock_name"])
        proof = []
        for source_value in sorted(observations):
            direct_clocks = []
            override_clocks = []
            targets = set()
            for clock_name in sorted(observations[source_value]):
                if clock_name in overrides:
                    override_clocks.append(clock_name)
                    targets.update(overrides[clock_name]["canonical_values"])
                else:
                    direct_clocks.append(clock_name)
                    targets.add(aliases.get(source_value, source_value))
            proof.append(
                {
                    "source_value": source_value,
                    "clock_count": len(observations[source_value]),
                    "direct_clock_count": len(direct_clocks),
                    "override_clock_count": len(override_clocks),
                    "mapping": (
                        "per-clock override"
                        if override_clocks
                        else "canonical"
                        if source_value in canonical
                        else "alias"
                    ),
                    "canonical_values": sorted(targets),
                }
            )
        coverage[field] = {
            "observed_value_count": len(observations),
            "covered_value_count": len(observations),
            "unmapped_values": [],
            "canonical_value_count": len(canonical),
            "proof": proof,
        }
        observed_counts[field] = len(observations)
        canonical_counts[field] = len(canonical)
        alias_counts[field] = len(aliases)
        override_counts[field] = len(overrides)
    decisions["coverage"] = coverage
    decisions["counts"] = {
        "fields": len(decisions["decisions"]),
        "observed_values": observed_counts,
        "canonical_values": canonical_counts,
        "aliases": alias_counts,
        "per_clock_overrides": override_counts,
        "unmapped_values": 0,
    }


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


def test_validate_shard_accepts_author_confirmed_evidence_from_author_communication(tmp_path):
    batch_path, shard_path, shard = make_batch_and_shard(tmp_path)
    record = shard["records"][0]
    record["reviewer"] = shard["reviewer"] = "paper-audit-01"
    record["sources"][0].update(
        {
            "type": "author communication",
            "url": "https://github.com/lcamillo/CpGPT",
            "accessed": "2026-07-18",
        }
    )
    record["fields"]["year"].update(
        {
            "status": "author-confirmed",
            "source_text": "Direct author clarification",
            "locator": "Direct author clarification in Codex task, 2026-07-18",
        }
    )
    write_json(shard_path, shard)

    assert validate_shard(batch_path, shard_path) == {
        "batch": "01",
        "paper_count": 2,
        "clock_count": 2,
    }


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


def reconciliation_fixture(tmp_path):
    batch_path, shard_path, shard = make_batch_and_shard(tmp_path)
    batch = json.loads(batch_path.read_text())
    manifest = {
        "schema_version": 1,
        "paper_count": 2,
        "clock_count": 2,
        "batches": [{"batch": "01", "paper_count": 2, "clock_count": 2, "path": "batch-01.json"}],
    }
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    vocabulary = {
        "schema_version": 1,
        "array_fields": list(ARRAY_FIELDS),
        "fields": {},
    }
    for field in (*ARRAY_FIELDS, *CONTROLLED_SCALAR_FIELDS):
        values = []
        for record in shard["records"]:
            value = record["fields"][field]["value"]
            values.extend(value if field in ARRAY_FIELDS else [value])
        vocabulary["fields"][field] = {
            "description": f"Allowed {field}",
            "values": sorted(set(values)),
            "aliases": {},
        }
    vocabulary_path = tmp_path / "vocabulary.json"
    write_json(vocabulary_path, vocabulary)
    return manifest_path, batch_path, shard_path, shard, vocabulary_path, vocabulary


def vocabulary_decision_fixture(tmp_path):
    (
        manifest_path,
        _batch_path,
        _shard_path,
        _shard,
        vocabulary_path,
        vocabulary,
    ) = reconciliation_fixture(tmp_path)
    reconciled_path = tmp_path / "reconciled.json"
    reconciled = merge_shards(manifest_path, tmp_path, reconciled_path)
    decisions = {
        "schema_version": 1,
        "source": {
            "reconciled": str(reconciled_path),
            "sha256": hashlib.sha256(reconciled_path.read_bytes()).hexdigest(),
            "paper_count": reconciled["paper_count"],
            "clock_count": reconciled["clock_count"],
        },
        "rationale": ["Test decision artifact."],
        "decisions": {
            field: {
                "canonical_values": descriptor["values"],
                "aliases": descriptor["aliases"],
                "per_clock_overrides": {},
            }
            for field, descriptor in vocabulary["fields"].items()
        },
        "ambiguities": [],
        "coverage": {},
        "counts": {},
    }
    populate_decision_snapshots(decisions, reconciled)
    decisions_path = tmp_path / "decisions.json"
    write_json(decisions_path, decisions)
    return reconciled_path, decisions_path, vocabulary_path, vocabulary, decisions


def test_merge_shards_happy_path_preserves_exact_records_and_refuses_existing_output(tmp_path):
    manifest_path, _batch, _shard_path, shard, _vocabulary_path, _vocabulary = reconciliation_fixture(tmp_path)
    output = tmp_path / "merged.json"

    merged = merge_shards(manifest_path, tmp_path, output)

    assert merged == {
        "schema_version": 1,
        "paper_count": 2,
        "clock_count": 2,
        "records": shard["records"],
    }
    assert json.loads(output.read_text()) == merged
    original = output.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        merge_shards(manifest_path, tmp_path, output)
    assert output.read_bytes() == original


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda root, manifest: (root / "shard-01.json").unlink(), "missing shard"),
        (lambda root, manifest: write_json(root / "shard-99.json", {}), "extra shard"),
        (lambda root, manifest: write_json(root / "batch-99.json", {}), "extra batch"),
        (
            lambda root, manifest: manifest["batches"][0].__setitem__("clock_count", 1),
            "clock_count",
        ),
        (
            lambda root, manifest: manifest.__setitem__("clock_count", 3),
            "clock_count",
        ),
    ],
)
def test_merge_shards_rejects_missing_extra_and_bad_coverage(tmp_path, mutation, message):
    manifest_path, _batch, _shard_path, _shard, _vocabulary_path, _vocabulary = reconciliation_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    mutation(tmp_path, manifest)
    write_json(manifest_path, manifest)
    output = tmp_path / "merged.json"

    with pytest.raises(ValueError, match=message):
        merge_shards(manifest_path, tmp_path, output)

    assert not output.exists()


def test_merge_shards_rejects_duplicate_clock_across_batches(tmp_path):
    manifest_path, _batch, _shard_path, shard, _vocabulary_path, _vocabulary = reconciliation_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    batch = json.loads((tmp_path / "batch-01.json").read_text())
    beta_paper = batch["papers"].pop()
    batch["paper_count"] = batch["clock_count"] = 1
    shard["records"].pop()
    write_json(tmp_path / "batch-01.json", batch)
    write_json(tmp_path / "shard-01.json", shard)

    duplicate_batch = copy.deepcopy(batch)
    duplicate_batch["batch"] = "02"
    duplicate_batch["papers"][0]["current_metadata"]["alpha"]["doi"] = beta_paper["doi"]
    duplicate_batch["papers"][0]["doi"] = beta_paper["doi"]
    duplicate_shard = copy.deepcopy(shard)
    duplicate_shard["batch"] = "02"
    duplicate_shard["reviewer"] = "paper-audit-02"
    duplicate_shard["records"][0]["reviewer"] = "paper-audit-02"
    duplicate_shard["records"][0]["doi"] = beta_paper["doi"]
    write_json(tmp_path / "batch-02.json", duplicate_batch)
    write_json(tmp_path / "shard-02.json", duplicate_shard)
    manifest["paper_count"] = manifest["clock_count"] = 2
    manifest["batches"] = [
        {"batch": "01", "paper_count": 1, "clock_count": 1, "path": "batch-01.json"},
        {"batch": "02", "paper_count": 1, "clock_count": 1, "path": "batch-02.json"},
    ]
    write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="duplicate clock"):
        merge_shards(manifest_path, tmp_path, tmp_path / "merged.json")


def test_merge_shards_rejects_doi_split_across_batches(tmp_path):
    manifest_path, _batch, _shard_path, shard, _vocabulary_path, _vocabulary = reconciliation_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    batch = json.loads((tmp_path / "batch-01.json").read_text())
    second_paper = batch["papers"].pop()
    second_record = shard["records"].pop()
    batch["paper_count"] = batch["clock_count"] = 1
    shard["records"][0] = shard["records"][0]
    write_json(tmp_path / "batch-01.json", batch)
    write_json(tmp_path / "shard-01.json", shard)

    second_paper["doi"] = batch["papers"][0]["doi"]
    second_paper["current_metadata"]["beta"]["doi"] = second_paper["doi"]
    second_record["doi"] = second_paper["doi"]
    second_record["reviewer"] = "paper-audit-02"
    second_batch = {
        "schema_version": 1,
        "batch": "02",
        "paper_count": 1,
        "clock_count": 1,
        "papers": [second_paper],
    }
    second_shard = {
        "schema_version": 1,
        "batch": "02",
        "reviewer": "paper-audit-02",
        "records": [second_record],
    }
    write_json(tmp_path / "batch-02.json", second_batch)
    write_json(tmp_path / "shard-02.json", second_shard)
    manifest["batches"] = [
        {"batch": "01", "paper_count": 1, "clock_count": 1, "path": "batch-01.json"},
        {"batch": "02", "paper_count": 1, "clock_count": 1, "path": "batch-02.json"},
    ]
    write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="DOI split"):
        merge_shards(manifest_path, tmp_path, tmp_path / "merged.json")


def test_merge_shards_loads_each_batch_and_shard_exactly_once(tmp_path, monkeypatch):
    manifest_path, batch_path, shard_path, shard, _vocabulary_path, _vocabulary = reconciliation_fixture(tmp_path)
    original_load = audit_tools.load_json
    load_counts = {}

    def counting_load(path):
        resolved = Path(path).resolve()
        load_counts[resolved] = load_counts.get(resolved, 0) + 1
        value = original_load(path)
        if resolved == batch_path.resolve():
            replacement = json.loads(batch_path.read_text())
            replacement["papers"][0]["current_metadata"]["alpha"]["notes"] = "replacement after read"
            write_json(batch_path, replacement)
        if resolved == shard_path.resolve():
            replacement = copy.deepcopy(shard)
            replacement["records"][0]["fields"]["notes"]["value"] = "replacement after read"
            write_json(shard_path, replacement)
        return value

    monkeypatch.setattr(audit_tools, "load_json", counting_load)
    merged = merge_shards(manifest_path, tmp_path, tmp_path / "merged.json")

    assert load_counts[batch_path.resolve()] == 1
    assert load_counts[shard_path.resolve()] == 1
    assert merged["records"][0]["fields"]["notes"]["value"] == "Example note"


def test_vocabulary_report_classifies_values_and_groups_only_conservative_variants(tmp_path):
    manifest_path, _batch, _shard_path, shard, vocabulary_path, vocabulary = reconciliation_fixture(tmp_path)
    merged_path = tmp_path / "merged.json"
    merge_shards(manifest_path, tmp_path, merged_path)
    merged = json.loads(merged_path.read_text())
    vocabulary["fields"]["tissue"]["aliases"] = {"Blood": "blood"}
    write_json(vocabulary_path, vocabulary)
    merged["records"][0]["fields"]["tissue"]["value"] = ["blood"]
    merged["records"][1]["fields"]["tissue"]["value"] = [
        "Blood",
        "cell-lines",
        "blood plasma",
    ]
    merged["records"][1]["fields"]["tissue"]["status"] = "paper-confirmed"
    merged["records"][1]["fields"]["tissue"]["source_text"] = "Blood; cell lines; blood plasma"
    merged["records"][1]["fields"]["tissue"]["locator"] = "page 1"
    write_json(merged_path, merged)

    report = vocabulary_report(merged_path, vocabulary_path, tmp_path / "vocabulary-report.json")
    tissue = report["fields"]["tissue"]

    assert tissue["exact_known"][0]["value"] == "blood"
    assert tissue["alias_known"][0]["value"] == "Blood"
    assert [item["value"] for item in tissue["unknown_values"]] == ["blood plasma", "cell-lines"]
    groups = [[value["value"] for value in group["values"]] for group in tissue["candidate_groups"]]
    assert ["Blood", "blood"] in groups
    assert all("blood plasma" not in group or "blood" not in group for group in groups)
    assert tissue["alias_known"][0]["clock_names"] == ["beta"]
    assert tissue["alias_known"][0]["statuses"] == ["paper-confirmed"]
    assert tissue["alias_known"][0]["source_terms"] == ["Blood; cell lines; blood plasma"]


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("blood cell", "blood cells"),
        ("brain tissue", "brain tissues"),
        ("methylation assay", "methylation assays"),
        ("450K array", "450K arrays"),
        ("10 probe", "10 probes"),
        ("clock CpG", "clock CpGs"),
        ("training sample", "training samples"),
        ("healthy adult", "healthy adults"),
        ("histone mark", "histone marks"),
        ("risk score", "risk scores"),
        ("CD4 + cell", "cd4+ cells"),
        ("blood‐cell", "blood-cell"),
        ("Elastic net regression", "elastic-net regression"),
        ("tissue-specific", "tissue specific"),
        ("β-cell assay", "β cell assays"),
    ],
)
def test_candidate_key_groups_only_explicit_domain_safe_plurals(left, right):
    assert audit_tools._candidate_key(left) == audit_tools._candidate_key(right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("axe", "axes"),
        ("CD4+", "CD4-"),
        ("CD4+ cells", "CD4- cells"),
        ("blood/cells", "blood-cells"),
        ("risk (score)", "risk score"),
        ("5% sample", "5 sample"),
        ("cell+cell", "cell-cell"),
        ("H3K27-ac", "H3K27 ac"),
        ("age-", "age"),
        ("-adult", "adult"),
    ],
)
def test_candidate_key_preserves_meaningful_punctuation_and_unsafe_plurals(left, right):
    assert audit_tools._candidate_key(left) != audit_tools._candidate_key(right)


def test_normalize_merged_applies_explicit_aliases_only_and_preserves_evidence(tmp_path):
    manifest_path, _batch, _shard_path, _shard, vocabulary_path, vocabulary = reconciliation_fixture(tmp_path)
    merged_path = tmp_path / "merged.json"
    merge_shards(manifest_path, tmp_path, merged_path)
    merged = json.loads(merged_path.read_text())
    vocabulary["fields"]["tissue"]["aliases"] = {"Blood": "blood"}
    write_json(vocabulary_path, vocabulary)
    evidence = merged["records"][0]["fields"]["tissue"]
    evidence["value"] = ["Blood"]
    before = copy.deepcopy(evidence)
    write_json(merged_path, merged)

    normalized = normalize_merged(merged_path, vocabulary_path, tmp_path / "normalized.json")
    after = normalized["records"][0]["fields"]["tissue"]

    assert after["value"] == ["blood"]
    assert {key: after[key] for key in after if key != "value"} == {
        key: before[key] for key in before if key != "value"
    }


def test_normalize_merged_aggregates_unknowns_and_rejects_alias_duplicates(tmp_path):
    manifest_path, _batch, _shard_path, _shard, vocabulary_path, vocabulary = reconciliation_fixture(tmp_path)
    merged_path = tmp_path / "merged.json"
    merge_shards(manifest_path, tmp_path, merged_path)
    merged = json.loads(merged_path.read_text())
    merged["records"][0]["fields"]["tissue"]["value"] = ["mystery"]
    merged["records"][1]["fields"]["model_type"]["value"] = "Mystery model"
    write_json(merged_path, merged)
    output = tmp_path / "normalized.json"

    with pytest.raises(ValueError) as error:
        normalize_merged(merged_path, vocabulary_path, output)
    message = str(error.value)
    assert "model_type='Mystery model': beta" in message
    assert "tissue='mystery': alpha" in message
    assert not output.exists()

    vocabulary["fields"]["tissue"]["aliases"] = {"Blood": "blood"}
    write_json(vocabulary_path, vocabulary)
    merged["records"][0]["fields"]["tissue"]["value"] = ["blood", "Blood"]
    merged["records"][1]["fields"]["model_type"]["value"] = "Elastic net"
    write_json(merged_path, merged)
    with pytest.raises(ValueError, match="duplicate.*alias"):
        normalize_merged(merged_path, vocabulary_path, output)


def test_apply_vocabulary_decisions_uses_aliases_and_multivalued_overrides_without_changing_evidence(
    tmp_path,
):
    reconciled_path, decisions_path, vocabulary_path, vocabulary, decisions = vocabulary_decision_fixture(tmp_path)
    reconciled = json.loads(reconciled_path.read_text())
    alpha_tissue = reconciled["records"][0]["fields"]["tissue"]
    beta_tissue = reconciled["records"][1]["fields"]["tissue"]
    alpha_tissue["value"] = ["Blood"]
    beta_tissue["value"] = ["blood and cultured cells"]
    write_json(reconciled_path, reconciled)
    decisions["source"]["sha256"] = hashlib.sha256(reconciled_path.read_bytes()).hexdigest()

    canonical = ["blood", "cultured human cells"]
    vocabulary["fields"]["tissue"]["values"] = canonical
    vocabulary["fields"]["tissue"]["aliases"] = {"Blood": "blood"}
    write_json(vocabulary_path, vocabulary)
    tissue_decision = decisions["decisions"]["tissue"]
    tissue_decision["canonical_values"] = canonical
    tissue_decision["aliases"] = {"Blood": "blood"}
    tissue_decision["per_clock_overrides"] = {
        "beta": {
            "expected_value": ["blood and cultured cells"],
            "canonical_values": ["blood", "cultured human cells"],
        },
    }
    populate_decision_snapshots(decisions, reconciled)
    write_json(decisions_path, decisions)

    before = copy.deepcopy(reconciled)
    output = tmp_path / "normalized.json"
    normalized = apply_vocabulary_decisions(
        reconciled_path,
        decisions_path,
        vocabulary_path,
        output,
    )

    assert normalized["records"][0]["fields"]["tissue"]["value"] == ["blood"]
    assert normalized["records"][1]["fields"]["tissue"]["value"] == [
        "blood",
        "cultured human cells",
    ]
    for before_record, after_record in zip(before["records"], normalized["records"]):
        assert before_record["clock_name"] == after_record["clock_name"]
        assert before_record["reviewer"] == after_record["reviewer"]
        assert before_record["sources"] == after_record["sources"]
        assert before_record["access_issues"] == after_record["access_issues"]
        for field, before_evidence in before_record["fields"].items():
            after_evidence = after_record["fields"][field]
            if field not in (*ARRAY_FIELDS, *CONTROLLED_SCALAR_FIELDS):
                assert after_evidence == before_evidence
            else:
                assert {key: value for key, value in after_evidence.items() if key != "value"} == {
                    key: value for key, value in before_evidence.items() if key != "value"
                }
    assert output.read_text() == json.dumps(normalized, indent=2, ensure_ascii=False) + "\n"
    with pytest.raises(ValueError, match="already exists"):
        apply_vocabulary_decisions(
            reconciled_path,
            decisions_path,
            vocabulary_path,
            output,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda reconciled, decisions, vocabulary: decisions["source"].update({"reconciled": "/tmp/different.json"}),
            "source.*reconciled",
        ),
        (
            lambda reconciled, decisions, vocabulary: decisions["source"].update({"clock_count": 99}),
            "source.*clock_count",
        ),
        (
            lambda reconciled, decisions, vocabulary: reconciled["records"][0]["fields"]["tissue"].update(
                {"value": ["unmapped tissue"]}
            ),
            "no mapping.*unmapped tissue",
        ),
        (
            lambda reconciled, decisions, vocabulary: reconciled["records"][0]["fields"]["population"].update(
                {"value": "unmapped population"}
            ),
            "no mapping.*unmapped population",
        ),
        (
            lambda reconciled, decisions, vocabulary: decisions["ambiguities"].append(
                {"field": "platform", "clocks": ["alpha"], "reason": "Still under review"}
            ),
            "unresolved ambiguities",
        ),
        (
            lambda reconciled, decisions, vocabulary: decisions["decisions"]["tissue"]["aliases"].update(
                {"blood": "blood"}
            ),
            "collide|exactly one mapping",
        ),
        (
            lambda reconciled, decisions, vocabulary: decisions["decisions"]["tissue"]["per_clock_overrides"].update(
                {
                    "missing-clock": {
                        "expected_value": ["blood"],
                        "canonical_values": ["blood"],
                    }
                }
            ),
            "unknown clock",
        ),
        (
            lambda reconciled, decisions, vocabulary: decisions["decisions"]["population"][
                "per_clock_overrides"
            ].update(
                {
                    "alpha": {
                        "expected_value": "adults",
                        "canonical_values": ["adults", "children"],
                    }
                }
            ),
            "scalar field",
        ),
        (
            lambda reconciled, decisions, vocabulary: decisions["decisions"]["tissue"]["per_clock_overrides"].update(
                {
                    "alpha": {
                        "expected_value": ["blood"],
                        "canonical_values": ["blood", "blood"],
                    }
                }
            ),
            "unique",
        ),
        (
            lambda reconciled, decisions, vocabulary: vocabulary["fields"]["tissue"].update({"values": ["different"]}),
            "vocabulary.*does not match",
        ),
    ],
)
def test_apply_vocabulary_decisions_rejects_invalid_or_incomplete_decisions(
    tmp_path,
    mutate,
    message,
):
    reconciled_path, decisions_path, vocabulary_path, vocabulary, decisions = vocabulary_decision_fixture(tmp_path)
    reconciled = json.loads(reconciled_path.read_text())
    mutate(reconciled, decisions, vocabulary)
    write_json(reconciled_path, reconciled)
    decisions["source"]["sha256"] = hashlib.sha256(reconciled_path.read_bytes()).hexdigest()
    write_json(decisions_path, decisions)
    write_json(vocabulary_path, vocabulary)
    output = tmp_path / "normalized.json"

    with pytest.raises(ValueError, match=message):
        apply_vocabulary_decisions(
            reconciled_path,
            decisions_path,
            vocabulary_path,
            output,
        )
    assert not output.exists()


def test_apply_vocabulary_decisions_rejects_same_path_and_count_source_mutation(tmp_path):
    reconciled_path, decisions_path, vocabulary_path, _vocabulary, _decisions = vocabulary_decision_fixture(tmp_path)
    reconciled = json.loads(reconciled_path.read_text())
    reconciled["records"][0]["fields"]["tissue"]["value"] = ["skin"]
    write_json(reconciled_path, reconciled)

    with pytest.raises(ValueError, match=r"source\.sha256.*does not match"):
        apply_vocabulary_decisions(
            reconciled_path,
            decisions_path,
            vocabulary_path,
            tmp_path / "normalized.json",
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda decisions: decisions["counts"].update({"fields": 8}), r"counts.*does not match"),
        (
            lambda decisions: decisions["coverage"]["tissue"]["proof"][0].update({"clock_count": 999}),
            r"coverage.*(does not match|must equal)",
        ),
        (lambda decisions: decisions.update({"rationale": []}), r"rationale.*nonempty"),
        (
            lambda decisions: decisions.update({"rationale": ["Repeated.", "Repeated."]}),
            r"rationale.*unique",
        ),
        (lambda decisions: decisions.update({"rationale": [3]}), r"rationale.*strings"),
    ],
)
def test_apply_vocabulary_decisions_rejects_tampered_review_snapshots(
    tmp_path,
    mutate,
    message,
):
    reconciled_path, decisions_path, vocabulary_path, _vocabulary, decisions = vocabulary_decision_fixture(tmp_path)
    mutate(decisions)
    write_json(decisions_path, decisions)
    output = tmp_path / "normalized.json"

    with pytest.raises(ValueError, match=message):
        apply_vocabulary_decisions(
            reconciled_path,
            decisions_path,
            vocabulary_path,
            output,
        )
    assert not output.exists()


def test_apply_vocabulary_decisions_rejects_bool_for_proof_count_one(tmp_path):
    reconciled_path, decisions_path, vocabulary_path, vocabulary, decisions = vocabulary_decision_fixture(tmp_path)
    reconciled = json.loads(reconciled_path.read_text())
    reconciled["records"][1]["fields"]["tissue"]["value"] = ["skin"]
    write_json(reconciled_path, reconciled)
    decisions["source"]["sha256"] = hashlib.sha256(reconciled_path.read_bytes()).hexdigest()
    decisions["decisions"]["tissue"]["canonical_values"] = ["blood", "skin"]
    vocabulary["fields"]["tissue"]["values"] = ["blood", "skin"]
    write_json(vocabulary_path, vocabulary)
    populate_decision_snapshots(decisions, reconciled)
    proof = next(item for item in decisions["coverage"]["tissue"]["proof"] if item["source_value"] == "skin")
    assert proof["clock_count"] == 1
    proof["clock_count"] = True
    write_json(decisions_path, decisions)

    with pytest.raises(ValueError, match=r"coverage.*clock_count.*integer"):
        apply_vocabulary_decisions(
            reconciled_path,
            decisions_path,
            vocabulary_path,
            tmp_path / "normalized.json",
        )


def test_apply_vocabulary_decisions_rejects_bool_for_override_count_one(tmp_path):
    reconciled_path, decisions_path, vocabulary_path, _vocabulary, decisions = vocabulary_decision_fixture(tmp_path)
    reconciled = json.loads(reconciled_path.read_text())
    current = reconciled["records"][0]["fields"]["tissue"]["value"]
    decisions["decisions"]["tissue"]["per_clock_overrides"] = {
        "alpha": {
            "expected_value": current,
            "canonical_values": current,
        }
    }
    populate_decision_snapshots(decisions, reconciled)
    assert decisions["counts"]["per_clock_overrides"]["tissue"] == 1
    decisions["counts"]["per_clock_overrides"]["tissue"] = True
    write_json(decisions_path, decisions)

    with pytest.raises(
        ValueError,
        match=r"counts.*per_clock_overrides.*tissue.*integer",
    ):
        apply_vocabulary_decisions(
            reconciled_path,
            decisions_path,
            vocabulary_path,
            tmp_path / "normalized.json",
        )


def test_apply_vocabulary_decisions_requires_override_to_match_reviewed_current_value(
    tmp_path,
):
    reconciled_path, decisions_path, vocabulary_path, _vocabulary, decisions = vocabulary_decision_fixture(tmp_path)
    reconciled = json.loads(reconciled_path.read_text())
    decisions["decisions"]["tissue"]["per_clock_overrides"] = {
        "alpha": {
            "expected_value": ["skin"],
            "canonical_values": ["blood"],
        }
    }
    populate_decision_snapshots(decisions, reconciled)
    write_json(decisions_path, decisions)

    with pytest.raises(ValueError, match=r"expected_value.*current reviewed value"):
        apply_vocabulary_decisions(
            reconciled_path,
            decisions_path,
            vocabulary_path,
            tmp_path / "normalized.json",
        )


def test_apply_vocabulary_decisions_cli_is_deterministic_and_refuses_overwrite(tmp_path, capsys):
    reconciled_path, decisions_path, vocabulary_path, _vocabulary, _decisions = vocabulary_decision_fixture(tmp_path)
    output = tmp_path / "normalized.json"

    arguments = [
        "apply-vocabulary-decisions",
        "--reconciled",
        str(reconciled_path),
        "--decisions",
        str(decisions_path),
        "--vocabulary",
        str(vocabulary_path),
        "--output",
        str(output),
    ]
    assert main(arguments) == 0
    first_bytes = output.read_bytes()
    assert "normalized 2 clocks" in capsys.readouterr().out
    assert main(arguments) == 2
    assert output.read_bytes() == first_bytes
    assert "already exists" in capsys.readouterr().err


def test_merged_record_requires_evidence_doi_to_exactly_match_record_doi(tmp_path):
    manifest_path, _batch, _shard_path, _shard, vocabulary_path, _vocabulary = reconciliation_fixture(tmp_path)
    merged_path = tmp_path / "merged.json"
    merge_shards(manifest_path, tmp_path, merged_path)
    merged = json.loads(merged_path.read_text())
    merged["records"][0]["fields"]["doi"]["value"] = "https://doi.org/10.1000/different"
    write_json(merged_path, merged)

    with pytest.raises(ValueError, match=r"alpha\.doi.*record DOI"):
        normalize_merged(merged_path, vocabulary_path, tmp_path / "normalized.json")


def _resolve_all(merged):
    for record in merged["records"]:
        for evidence in record["fields"].values():
            evidence.update(
                {
                    "source_text": f"Confirmed {evidence['value']}",
                    "locator": "page 1",
                    "status": "paper-confirmed",
                }
            )
    return merged


def materialization_fixture(tmp_path, *, current_is_registry=False):
    manifest_path, _batch, _shard_path, _shard, vocabulary_path, _vocabulary = reconciliation_fixture(tmp_path)
    merged_path = tmp_path / "merged.json"
    merge_shards(manifest_path, tmp_path, merged_path)
    merged = _resolve_all(json.loads(merged_path.read_text()))
    normalized_path = tmp_path / "normalized.json"
    write_json(normalized_path, merged)
    current = {}
    for record in merged["records"]:
        name = record["clock_name"]
        current[name] = {field: copy.deepcopy(evidence["value"]) for field, evidence in record["fields"].items()}
        current[name].update(
            {
                "clock_name": name,
                "approved_by_author": "pending",
                "research_only": None,
                "citations": 0,
                "citations_date": "2026-07-18",
                "runtime_extra": {"keep": True},
            }
        )
    current["alpha"]["notes"] = "stale"
    current_path = tmp_path / ("registry.json" if current_is_registry else "current.json")
    write_json(current_path, current)
    targets = [
        tmp_path / "registry.json",
        tmp_path / "ledger.jsonl",
        tmp_path / "report.md",
    ]
    if not current_is_registry:
        targets[0].write_bytes(b"old registry bytes\n")
    targets[1].write_bytes(audit_tools._ledger_bytes(merged["records"]))
    targets[2].write_text("# Existing Clock Metadata Audit Report\n", encoding="utf-8")
    for index, target in enumerate(targets):
        target.chmod(0o640 + index)
    return normalized_path, current_path, vocabulary_path, targets, merged, current


def test_materialize_replaces_existing_targets_with_current_equal_to_registry(tmp_path, monkeypatch):
    normalized_path, current_path, vocabulary_path, targets, merged, current = materialization_fixture(
        tmp_path, current_is_registry=True
    )
    publishes = []
    original_publish = audit_tools._publish_replace

    def recording_publish(source, target):
        publishes.append(Path(target).name)
        original_publish(source, target)

    monkeypatch.setattr(audit_tools, "_publish_replace", recording_publish)

    result = materialize(
        normalized_path,
        current_path,
        vocabulary_path,
        *targets,
    )

    registry = json.loads(targets[0].read_text())
    assert registry["alpha"]["notes"] == merged["records"][0]["fields"]["notes"]["value"]
    assert registry["alpha"]["runtime_extra"] == {"keep": True}
    assert {key: registry["alpha"][key] for key in ADMIN_FIELDS} == {key: current["alpha"][key] for key in ADMIN_FIELDS}
    assert [json.loads(line) for line in targets[1].read_text().splitlines()] == merged["records"]
    assert result["clock_count"] == 2
    report = targets[2].read_text()
    assert "## Evidence status counts" in report
    assert "## Access issues" in report
    assert "## Field change counts" in report
    assert publishes == ["registry.json", "ledger.jsonl", "report.md"]
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".clock-metadata-")]


def test_materialized_report_preserves_scaffold_and_records_author_adjudication(tmp_path):
    normalized_path, current_path, vocabulary_path, targets, merged, _current = materialization_fixture(tmp_path)
    record = merged["records"][0]
    record["sources"].append(
        {
            "id": "author-clarification",
            "type": "author communication",
            "url": "https://github.com/lcamillo/CpGPT",
            "accessed": "2026-07-18",
        }
    )
    record["fields"]["tissue"].update(
        {
            "source_id": "author-clarification",
            "source_text": "The model was trained in blood with the 450K array in FHS.",
            "locator": "Direct author clarification in Codex task, 2026-07-18",
            "status": "author-confirmed",
            "note": "The manuscript anonymizes the cohort.",
        }
    )
    write_json(normalized_path, merged)

    materialize(normalized_path, current_path, vocabulary_path, *targets)

    report = targets[2].read_text()
    for heading in (
        "## Scope",
        "## Controlled-vocabulary decisions",
        "## Access issues",
        "## Source contradictions and adjudications",
        "## Changed-value summary",
        "## Validation",
        "## Hugging Face publication",
    ):
        assert heading in report
    assert "author-confirmed" in report
    assert "Direct author clarification in Codex task, 2026-07-18" in report
    assert "manuscript anonymizes the cohort" in report


def test_materialized_report_labels_stale_unresolved_access_issue_as_superseded(tmp_path):
    normalized_path, current_path, vocabulary_path, targets, merged, _current = materialization_fixture(tmp_path)
    merged["records"][0]["access_issues"] = [
        "The training target remains unresolved in the assigned paper."
    ]
    write_json(normalized_path, merged)

    materialize(normalized_path, current_path, vocabulary_path, *targets)

    report = targets[2].read_text()
    assert (
        "- alpha: First-pass source limitation "
        "(retained for provenance; final metadata evidence resolved): "
        "The training target remains unresolved in the assigned paper."
    ) in report
    assert "\n- alpha: The training target remains unresolved" not in report
    assert "## Source contradictions and adjudications" in report


@pytest.mark.parametrize("failure_point", [1, 2, 3, "post-validate"])
def test_materialize_rolls_back_every_target_on_publish_or_validation_failure(tmp_path, monkeypatch, failure_point):
    normalized_path, current_path, vocabulary_path, targets, _merged, _current = materialization_fixture(tmp_path)
    original_bytes = {target: target.read_bytes() for target in targets}
    original_modes = {target: target.stat().st_mode for target in targets}
    original_publish = audit_tools._publish_replace
    publish_count = 0

    def failing_publish(source, target):
        nonlocal publish_count
        publish_count += 1
        if publish_count == failure_point:
            raise OSError(f"injected replace {failure_point}")
        original_publish(source, target)

    if failure_point == "post-validate":
        monkeypatch.setattr(
            audit_tools,
            "_post_validate_materialization",
            lambda *_args: (_ for _ in ()).throw(ValueError("injected post-validate")),
        )
    else:
        monkeypatch.setattr(audit_tools, "_publish_replace", failing_publish)

    with pytest.raises(ValueError, match="materialization transaction failed"):
        materialize(normalized_path, current_path, vocabulary_path, *targets)

    assert {target: target.read_bytes() for target in targets} == original_bytes
    assert {target: target.stat().st_mode for target in targets} == original_modes
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".clock-metadata-")]


def _assert_materialized_targets_are_new_and_consistent(targets, merged):
    registry = json.loads(targets[0].read_text())
    assert registry["alpha"]["notes"] == merged["records"][0]["fields"]["notes"]["value"]
    assert [json.loads(line) for line in targets[1].read_text().splitlines()] == merged["records"]
    assert targets[2].read_text().startswith("# Clock Metadata Source Audit\n")


@pytest.mark.parametrize("failure_boundary", [1, 2, 3])
def test_materialize_backup_cleanup_failure_keeps_fully_committed_targets(tmp_path, monkeypatch, failure_boundary):
    normalized_path, current_path, vocabulary_path, targets, merged, _current = materialization_fixture(tmp_path)
    original_cleanup = audit_tools._cleanup_materialization_backup
    cleanup_count = 0

    def failing_cleanup(backup):
        nonlocal cleanup_count
        cleanup_count += 1
        if cleanup_count == failure_boundary:
            raise OSError(f"injected backup cleanup {failure_boundary}")
        original_cleanup(backup)

    monkeypatch.setattr(audit_tools, "_cleanup_materialization_backup", failing_cleanup)

    with pytest.raises(ValueError) as error:
        materialize(normalized_path, current_path, vocabulary_path, *targets)

    assert "targets committed" in str(error.value)
    assert "recovery artifact" in str(error.value)
    _assert_materialized_targets_are_new_and_consistent(targets, merged)
    leftovers = [path for path in tmp_path.iterdir() if path.name.startswith(".clock-metadata-backup-")]
    assert len(leftovers) == 1
    assert not (tmp_path / ".clock-metadata-materialize.lock").exists()


@pytest.mark.parametrize("failure_kind", ["unlink", "fsync"])
def test_materialize_lock_cleanup_failure_never_rolls_back_committed_targets(tmp_path, monkeypatch, failure_kind):
    normalized_path, current_path, vocabulary_path, targets, merged, _current = materialization_fixture(tmp_path)
    lock = tmp_path / ".clock-metadata-materialize.lock"
    if failure_kind == "unlink":
        monkeypatch.setattr(
            audit_tools,
            "_remove_materialization_lock",
            lambda _path: (_ for _ in ()).throw(OSError("injected lock unlink")),
        )
    else:
        monkeypatch.setattr(
            audit_tools,
            "_fsync_materialization_lock_removal",
            lambda _parent: (_ for _ in ()).throw(OSError("injected lock fsync")),
        )

    with pytest.raises(ValueError) as error:
        materialize(normalized_path, current_path, vocabulary_path, *targets)

    assert "targets committed" in str(error.value)
    assert "lock cleanup failure" in str(error.value)
    _assert_materialized_targets_are_new_and_consistent(targets, merged)
    assert lock.exists() is (failure_kind == "unlink")
    if failure_kind == "unlink":
        assert "remove manually" in str(error.value)


def test_materialize_rolls_back_new_targets_by_removing_them(tmp_path, monkeypatch):
    normalized_path, current_path, vocabulary_path, targets, _merged, _current = materialization_fixture(tmp_path)
    targets[1].unlink()
    targets[2].unlink()
    original_registry = targets[0].read_bytes()
    original_publish = audit_tools._publish_replace
    publish_count = 0

    def fail_third(source, target):
        nonlocal publish_count
        publish_count += 1
        if publish_count == 3:
            raise OSError("injected third replace")
        original_publish(source, target)

    monkeypatch.setattr(audit_tools, "_publish_replace", fail_third)

    with pytest.raises(ValueError, match="materialization transaction failed"):
        materialize(normalized_path, current_path, vocabulary_path, *targets)

    assert targets[0].read_bytes() == original_registry
    assert not targets[1].exists()
    assert not targets[2].exists()
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".clock-metadata-")]


def test_materialize_reports_both_publication_and_rollback_failure(tmp_path, monkeypatch):
    normalized_path, current_path, vocabulary_path, targets, _merged, _current = materialization_fixture(tmp_path)
    original_publish = audit_tools._publish_replace
    original_replace = audit_tools.os.replace
    publish_count = 0

    def fail_second(source, target):
        nonlocal publish_count
        publish_count += 1
        if publish_count == 2:
            raise OSError("injected publication failure")
        original_publish(source, target)

    def fail_backup_restore(source, target):
        if ".clock-metadata-backup-" in str(source):
            raise OSError("injected rollback failure")
        original_replace(source, target)

    monkeypatch.setattr(audit_tools, "_publish_replace", fail_second)
    monkeypatch.setattr(audit_tools.os, "replace", fail_backup_restore)

    with pytest.raises(ValueError) as error:
        materialize(normalized_path, current_path, vocabulary_path, *targets)

    assert "injected publication failure" in str(error.value)
    assert "injected rollback failure" in str(error.value)


def test_materialize_fails_cleanly_when_transaction_lock_exists(tmp_path):
    normalized_path, current_path, vocabulary_path, targets, _merged, _current = materialization_fixture(tmp_path)
    originals = {target: target.read_bytes() for target in targets}
    lock = tmp_path / ".clock-metadata-materialize.lock"
    lock.write_text("other process\n")

    with pytest.raises(ValueError, match="locked"):
        materialize(normalized_path, current_path, vocabulary_path, *targets)

    assert {target: target.read_bytes() for target in targets} == originals
    assert lock.read_text() == "other process\n"
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".clock-metadata-") and path != lock]


def test_materialize_rejects_different_output_parents_before_reading_inputs(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    targets = [tmp_path / "registry.json", other / "ledger.jsonl", tmp_path / "report.md"]

    with pytest.raises(ValueError, match="same parent"):
        materialize(
            tmp_path / "missing-normalized.json",
            tmp_path / "missing-current.json",
            tmp_path / "missing-vocabulary.json",
            *targets,
        )

    assert not any(path.exists() for path in targets)


def test_materialize_rejects_evidence_doi_mismatch_before_mutating_existing_targets(tmp_path):
    normalized_path, current_path, vocabulary_path, targets, _merged, _current = materialization_fixture(tmp_path)
    normalized = json.loads(normalized_path.read_text())
    normalized["records"][0]["fields"]["doi"]["value"] = "https://doi.org/10.1000/different"
    write_json(normalized_path, normalized)
    originals = {target: target.read_bytes() for target in targets}

    with pytest.raises(ValueError, match=r"alpha\.doi.*record DOI"):
        materialize(normalized_path, current_path, vocabulary_path, *targets)

    assert {target: target.read_bytes() for target in targets} == originals
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".clock-metadata-")]


def test_materialize_cli_replaces_existing_canonical_targets(tmp_path, capsys):
    normalized_path, current_path, vocabulary_path, targets, _merged, _current = materialization_fixture(
        tmp_path, current_is_registry=True
    )

    exit_code = main(
        [
            "materialize",
            "--normalized",
            str(normalized_path),
            "--current",
            str(current_path),
            "--vocabulary",
            str(vocabulary_path),
            "--registry",
            str(targets[0]),
            "--ledger",
            str(targets[1]),
            "--report",
            str(targets[2]),
        ]
    )

    assert exit_code == 0
    assert "materialized 2 clocks" in capsys.readouterr().out
    assert json.loads(targets[0].read_text())["alpha"]["notes"] == "Example note"


def test_materialize_blocks_unresolved_and_preflight_errors_write_nothing(tmp_path):
    manifest_path, _batch, _shard_path, _shard, vocabulary_path, _vocabulary = reconciliation_fixture(tmp_path)
    normalized_path = tmp_path / "normalized.json"
    merge_shards(manifest_path, tmp_path, normalized_path)
    current = {
        record["clock_name"]: {
            **{field: copy.deepcopy(evidence["value"]) for field, evidence in record["fields"].items()},
            "clock_name": record["clock_name"],
            "approved_by_author": "pending",
            "research_only": None,
            "citations": 0,
            "citations_date": "2026-07-18",
        }
        for record in json.loads(normalized_path.read_text())["records"]
    }
    current_path = tmp_path / "current.json"
    write_json(current_path, current)
    outputs = [tmp_path / name for name in ("registry.json", "ledger.jsonl", "report.md")]

    with pytest.raises(ValueError, match="unresolved"):
        materialize(normalized_path, current_path, vocabulary_path, *outputs)

    assert not any(path.exists() for path in outputs)


@pytest.mark.parametrize(
    "targets",
    [
        ("same.json", "same.json", "report.md"),
        ("registry.json", "same.json", "same.json"),
        ("same.json", "ledger.jsonl", "same.json"),
        ("same.json", "same.json", "same.json"),
    ],
)
def test_materialize_rejects_pairwise_and_all_same_output_paths_before_reading_inputs(tmp_path, targets):
    output_paths = [tmp_path / target for target in targets]

    with pytest.raises(ValueError, match="distinct"):
        materialize(
            tmp_path / "missing-normalized.json",
            tmp_path / "missing-current.json",
            tmp_path / "missing-vocabulary.json",
            *output_paths,
        )

    assert not any(path.exists() for path in set(output_paths))


def test_materialize_rejects_output_paths_that_resolve_to_same_location(tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    targets = [tmp_path / "same.json", alias / "same.json", tmp_path / "report.md"]

    with pytest.raises(ValueError, match="distinct"):
        materialize(
            tmp_path / "missing-normalized.json",
            tmp_path / "missing-current.json",
            tmp_path / "missing-vocabulary.json",
            *targets,
        )

    assert not any(path.exists() for path in targets)


def test_reconciliation_cli_wiring(tmp_path, capsys):
    manifest_path, _batch, _shard_path, _shard, vocabulary_path, _vocabulary = reconciliation_fixture(tmp_path)
    merged = tmp_path / "merged.json"
    assert (
        main(["merge-shards", "--manifest", str(manifest_path), "--shards", str(tmp_path), "--output", str(merged)])
        == 0
    )
    report = tmp_path / "vocab.json"
    assert (
        main(
            [
                "vocabulary-report",
                "--merged",
                str(merged),
                "--vocabulary",
                str(vocabulary_path),
                "--output",
                str(report),
            ]
        )
        == 0
    )
    assert "merged 2 clocks" in capsys.readouterr().out
