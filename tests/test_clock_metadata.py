import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from clocks.metadata.validate_metadata import (
    ARRAY_FIELDS,
    AUDITED_FIELDS,
    _effective_model_feature_count,
    load_json,
    load_ledger,
    validate_artifact_consistency,
    validate_evidence,
    validate_registry,
    validate_vocabulary,
)

ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = ROOT / "clocks" / "metadata"
REGISTRY_PATH = METADATA_DIR / "clock_metadata.json"
VOCABULARY_PATH = METADATA_DIR / "controlled_vocabulary.json"
LEDGER_PATH = METADATA_DIR / "evidence_ledger.jsonl"


@pytest.fixture(scope="module")
def registry():
    return load_json(REGISTRY_PATH)


@pytest.fixture(scope="module")
def vocabulary():
    return load_json(VOCABULARY_PATH)


@pytest.fixture(scope="module")
def ledger():
    return load_ledger(LEDGER_PATH)


def test_registry_has_every_implementation_notebook(registry):
    notebook_names = {
        path.stem for path in (ROOT / "clocks" / "notebooks").glob("*.ipynb") if path.name != "template.ipynb"
    }
    assert len(registry) == 173
    assert set(registry) == notebook_names


def test_registry_uses_controlled_arrays(registry, vocabulary):
    validate_registry(registry, vocabulary)
    for clock_name, record in registry.items():
        for field in ARRAY_FIELDS:
            assert isinstance(record[field], list), f"{clock_name}.{field}"
            assert record[field], f"{clock_name}.{field}"


def test_evidence_is_complete_and_resolved(registry, ledger):
    validate_evidence(registry, ledger)
    for clock_name, record in ledger.items():
        for field in AUDITED_FIELDS:
            assert record["fields"][field]["status"] != "unresolved", f"{clock_name}.{field}"


def test_reedbmi_access_issues_do_not_contradict_corrected_feature_count(ledger):
    assert not any(
        "134 CpGs" in issue for issue in ledger["reedbmi"]["access_issues"]
    )


def _write_consistent_artifact_fixture(tmp_path, registry, clock_name="tiny"):
    root = tmp_path
    metadata_dir = root / "clocks" / "metadata"
    notebooks_dir = root / "clocks" / "notebooks"
    weights_dir = root / "clocks" / "weights"
    metadata_dir.mkdir(parents=True)
    notebooks_dir.mkdir()
    weights_dir.mkdir()
    record = copy.deepcopy(next(iter(registry.values())))
    for runtime_field in ("version", "preprocess", "postprocess", "reference_values"):
        record.pop(runtime_field, None)
    record["clock_name"] = clock_name
    record["n_features"] = 2
    (metadata_dir / "clock_metadata.json").write_text(
        json.dumps({clock_name: record}, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (metadata_dir / "controlled_vocabulary.json").write_bytes(VOCABULARY_PATH.read_bytes())
    evidence_fields = {
        field: {
            "value": record[field],
            "source_text": "exact\nsource\twording",
            "source_id": "paper",
            "locator": "Methods",
            "status": "paper-confirmed",
            "note": "",
        }
        for field in AUDITED_FIELDS
    }
    ledger_record = {
        "clock_name": clock_name,
        "doi": record["doi"],
        "reviewer": "test-reviewer",
        "sources": [
            {
                "id": "paper",
                "type": "paper",
                "url": record["doi"],
                "accessed": "2026-07-18",
            }
        ],
        "fields": evidence_fields,
        "access_issues": [],
    }
    (metadata_dir / "evidence_ledger.jsonl").write_text(
        json.dumps(ledger_record, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    controlled = ARRAY_FIELDS + ("data_type", "species", "model_type", "population")
    lines = []
    for field, value in record.items():
        literal = repr(value)
        comment = "  # Paper: exact source wording" if field in controlled else ""
        lines.append(f'model.metadata["{field}"] = {literal}{comment}\n')
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": lines,
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (notebooks_dir / f"{clock_name}.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    model = SimpleNamespace(
        metadata=copy.deepcopy(record),
        features=["a", "b"],
        version="1",
        preprocess_name="none",
        postprocess_name="none",
        reference_values=None,
    )
    torch.save(model, weights_dir / f"{clock_name}.pt")
    aggregate_record = copy.deepcopy(record)
    aggregate_record.update(
        {"version": "1", "preprocess": "none", "postprocess": "none"}
    )
    torch.save({clock_name: aggregate_record}, metadata_dir / "all_clock_metadata.pt")
    return root


def test_validate_artifact_consistency_checks_notebooks_weights_and_aggregate(tmp_path, registry):
    root = _write_consistent_artifact_fixture(tmp_path, registry)

    assert validate_artifact_consistency(root) == {"clock_count": 1}


def test_validate_artifact_consistency_verifies_registry_runtime_fields_separately(tmp_path, registry):
    root = _write_consistent_artifact_fixture(tmp_path, registry)
    registry_path = root / "clocks" / "metadata" / "clock_metadata.json"
    local_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    local_registry["tiny"]["version"] = "1"
    registry_path.write_text(json.dumps(local_registry) + "\n", encoding="utf-8")
    weight_path = root / "clocks" / "weights" / "tiny.pt"
    model = torch.load(weight_path, weights_only=False, map_location="cpu")
    model.metadata["version"] = "1"
    torch.save(model, weight_path)

    assert validate_artifact_consistency(root) == {"clock_count": 1}


def test_validate_artifact_consistency_requires_same_line_paper_comments(tmp_path, registry):
    root = _write_consistent_artifact_fixture(tmp_path, registry)
    notebook_path = root / "clocks" / "notebooks" / "tiny.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"][0]["source"] = [
        line.replace("  # Paper: exact source wording", "")
        if '["tissue"]' in line
        else line
        for line in notebook["cells"][0]["source"]
    ]
    notebook_path.write_text(json.dumps(notebook) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"tiny\.tissue.*same-line.*# Paper:"):
        validate_artifact_consistency(root)


def test_validate_artifact_consistency_requires_exact_collapsed_ledger_comment(tmp_path, registry):
    root = _write_consistent_artifact_fixture(tmp_path, registry)
    notebook_path = root / "clocks" / "notebooks" / "tiny.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"][0]["source"] = [
        line.replace(
            "# Paper: exact source wording",
            "# Paper: different but nonempty wording",
        )
        if '["tissue"]' in line
        else line
        for line in notebook["cells"][0]["source"]
    ]
    notebook_path.write_text(json.dumps(notebook) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"tiny\.tissue.*does not match evidence ledger"):
        validate_artifact_consistency(root)


def test_validate_artifact_consistency_checks_weight_feature_count(tmp_path, registry):
    root = _write_consistent_artifact_fixture(tmp_path, registry)
    weight_path = root / "clocks" / "weights" / "tiny.pt"
    model = torch.load(weight_path, weights_only=False, map_location="cpu")
    model.features.append("extra")
    torch.save(model, weight_path)

    with pytest.raises(ValueError, match=r"tiny\.n_features.*effective feature count"):
        validate_artifact_consistency(root)


def test_validate_artifact_consistency_rejects_raw_pool_for_sparse_selected_model(tmp_path, registry):
    root = _write_consistent_artifact_fixture(tmp_path, registry, clock_name="cellpopage")
    registry_path = root / "clocks" / "metadata" / "clock_metadata.json"
    local_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    local_registry["cellpopage"]["n_features"] = 50
    registry_path.write_text(json.dumps(local_registry) + "\n", encoding="utf-8")
    ledger_path = root / "clocks" / "metadata" / "evidence_ledger.jsonl"
    local_ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    local_ledger["fields"]["n_features"]["value"] = 50
    ledger_path.write_text(json.dumps(local_ledger) + "\n", encoding="utf-8")
    notebook_path = root / "clocks" / "notebooks" / "cellpopage.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"][0]["source"] = [
        'model.metadata["n_features"] = 50\n'
        if '["n_features"]' in line
        else line
        for line in notebook["cells"][0]["source"]
    ]
    notebook_path.write_text(json.dumps(notebook) + "\n", encoding="utf-8")
    weight_path = root / "clocks" / "weights" / "cellpopage.pt"
    model = torch.load(weight_path, weights_only=False, map_location="cpu")
    model.metadata["n_features"] = 50
    model.features = [f"feature-{index}" for index in range(50)]
    model.preprocess_name = None
    model.base_model_features = None
    model.base_model = SimpleNamespace(linear=torch.nn.Linear(50, 1))
    with torch.no_grad():
        model.base_model.linear.weight.zero_()
        model.base_model.linear.weight[0, :2] = torch.tensor([1.0, 2.0])
    torch.save(model, weight_path)
    aggregate_path = root / "clocks" / "metadata" / "all_clock_metadata.pt"
    aggregate = torch.load(aggregate_path, weights_only=False, map_location="cpu")
    aggregate["cellpopage"]["n_features"] = 50
    aggregate["cellpopage"].pop("preprocess")
    torch.save(aggregate, aggregate_path)

    with pytest.raises(ValueError, match=r"cellpopage\.n_features.*effective feature count 2"):
        validate_artifact_consistency(root)


def test_effective_model_feature_count_uses_selected_linear_predictors():
    linear = torch.nn.Linear(50, 1)
    with torch.no_grad():
        linear.weight.zero_()
        linear.weight[0, :2] = torch.tensor([1.0, 2.0])
    model = SimpleNamespace(
        features=[f"feature-{index}" for index in range(50)],
        base_model_features=None,
        base_model=SimpleNamespace(linear=linear),
        metadata={"clock_name": "cellpopage"},
    )

    assert _effective_model_feature_count(model) == 2


def test_effective_model_feature_count_has_no_arbitrary_sparsity_boundary():
    linear = torch.nn.Linear(50, 1)
    with torch.no_grad():
        linear.weight.zero_()
        linear.weight[0, :2] = torch.tensor([1.0, 2.0])
    model = SimpleNamespace(
        features=[f"feature-{index}" for index in range(50)],
        base_model_features=None,
        base_model=SimpleNamespace(linear=linear),
        metadata={"clock_name": "ordinary-clock"},
    )

    assert _effective_model_feature_count(model) == 50


def test_effective_model_feature_count_prefers_explicit_base_model_features():
    model = SimpleNamespace(
        features=list("abcde"),
        base_model_features=["a", "c", "e"],
        base_model=SimpleNamespace(),
        preprocess_name="tpm_norm_log1p",
    )

    assert _effective_model_feature_count(model) == 3


@pytest.mark.full_catalog
def test_local_runtime_artifacts_match_registry():
    validate_artifact_consistency(ROOT)


def test_load_ledger_rejects_duplicate_clock_names(tmp_path):
    path = tmp_path / "ledger.jsonl"
    record = {"clock_name": "clock"}
    path.write_text("\n".join((json.dumps(record), json.dumps(record))), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate.*clock"):
        load_ledger(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_load_json_rejects_non_finite_numbers(tmp_path, constant):
    path = tmp_path / "metadata.json"
    path.write_text(f'{{"value": {constant}}}', encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite"):
        load_json(path)


def test_load_json_rejects_nested_duplicate_keys_with_file_context(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text('{"outer": {"value": 1, "value": 2}}', encoding="utf-8")

    with pytest.raises(ValueError, match=r"metadata\.json.*duplicate key.*value"):
        load_json(path)


@pytest.mark.parametrize(
    ("line", "context"),
    [
        ("{", r"line 1"),
        ("null", r"line 1.*object"),
        ("[]", r"line 1.*object"),
        ('{"other": "value"}', r"line 1.*clock_name"),
        ('{"clock_name": 3}', r"line 1.*clock_name"),
        ('{"clock_name": ""}', r"line 1.*clock_name"),
        ('{"clock_name": "clock", "value": NaN}', r"line 1.*non-finite"),
    ],
)
def test_load_ledger_rejects_malformed_records(tmp_path, line, context):
    path = tmp_path / "ledger.jsonl"
    path.write_text(line, encoding="utf-8")

    with pytest.raises(ValueError, match=context):
        load_ledger(path)


def test_load_ledger_requires_alphabetical_order(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text('{"clock_name": "z"}\n{"clock_name": "a"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"line 2.*alphabetical"):
        load_ledger(path)


def test_load_ledger_rejects_nested_duplicate_keys_with_line_context(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text(
        '{"clock_name": "a"}\n{"clock_name": "b", "nested": {"value": 1, "value": 2}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"line 2.*duplicate key.*value"):
        load_ledger(path)


@pytest.mark.parametrize(
    ("mutation", "context"),
    [
        (lambda record: record.__setitem__("tissue", "blood"), r"example\.tissue"),
        (lambda record: record.__setitem__("year", True), r"example\.year"),
        (lambda record: record.__setitem__("data_type", "unknown"), r"example\.data_type"),
        (lambda record: record.pop("approved_by_author"), r"example\.approved_by_author"),
        (lambda record: record.__setitem__("citations", True), r"example\.citations"),
        (lambda record: record.__setitem__("citation", {"raw": "citation"}), r"example\.citation"),
    ],
)
def test_validate_registry_rejects_representative_invalid_values(registry, vocabulary, mutation, context):
    invalid_registry = {"example": copy.deepcopy(next(iter(registry.values())))}
    invalid_registry["example"]["clock_name"] = "example"
    mutation(invalid_registry["example"])

    with pytest.raises(ValueError, match=context):
        validate_registry(invalid_registry, vocabulary)


def test_validate_registry_rejects_unknown_fields(registry, vocabulary):
    invalid_registry = {"example": copy.deepcopy(next(iter(registry.values())))}
    invalid_registry["example"]["clock_name"] = "example"
    invalid_registry["example"]["typo_field"] = "unexpected"

    with pytest.raises(ValueError, match=r"example\.record.*unknown.*typo_field"):
        validate_registry(invalid_registry, vocabulary)


@pytest.mark.parametrize(
    ("mutation", "context"),
    [
        (lambda value: value.__setitem__("schema_version", 2), r"vocabulary\.schema_version"),
        (lambda value: value.__setitem__("array_fields", ["tissue"]), r"vocabulary\.array_fields"),
        (lambda value: value["fields"].pop("species"), r"vocabulary\.fields"),
        (
            lambda value: value["fields"]["species"].__setitem__("description", ""),
            r"vocabulary\.species\.description",
        ),
        (
            lambda value: value["fields"]["species"].__setitem__("values", []),
            r"vocabulary\.species\.values.*nonempty",
        ),
        (
            lambda value: value["fields"]["species"].__setitem__("values", ["a", "a"]),
            r"vocabulary\.species\.values.*unique",
        ),
        (
            lambda value: value["fields"]["species"].__setitem__("values", [True]),
            r"vocabulary\.species\.values",
        ),
        (
            lambda value: value["fields"]["species"].__setitem__("aliases", {"human": "missing"}),
            r"vocabulary\.species\.aliases",
        ),
    ],
)
def test_validate_vocabulary_rejects_malformed_schema(vocabulary, mutation, context):
    invalid_vocabulary = copy.deepcopy(vocabulary)
    mutation(invalid_vocabulary)

    with pytest.raises(ValueError, match=context):
        validate_vocabulary(invalid_vocabulary)


def test_validate_vocabulary_preserves_unique_declared_order(vocabulary):
    declared_order_vocabulary = copy.deepcopy(vocabulary)
    declared_order_vocabulary["fields"]["species"]["values"].reverse()

    validate_vocabulary(declared_order_vocabulary)


@pytest.mark.parametrize(
    ("mutation", "context"),
    [
        (
            lambda record: record["fields"]["year"].__setitem__("status", "guessed"),
            r"example\.year.*status",
        ),
        (
            lambda record: record["fields"]["year"].__setitem__("status", []),
            r"example\.year.*status",
        ),
        (
            lambda record: record["fields"]["year"].__setitem__("source_id", "missing"),
            r"example\.year.*source_id",
        ),
        (
            lambda record: record["fields"]["year"].__setitem__("source_id", {}),
            r"example\.year.*source_id",
        ),
        (
            lambda record: record["fields"]["year"].__setitem__("value", 1900),
            r"example\.year.*value",
        ),
        (
            lambda record: record["fields"]["year"].__setitem__("value", True),
            r"example\.year.*value",
        ),
        (
            lambda record: record["fields"]["year"].__setitem__("value", float(record["fields"]["year"]["value"])),
            r"example\.year.*value",
        ),
        (lambda record: record.__setitem__("clock_name", "wrong"), r"example\.clock_name"),
        (lambda record: record.__setitem__("doi", "https://doi.org/wrong"), r"example\.doi"),
        (lambda record: record.__setitem__("reviewer", ""), r"example\.reviewer"),
        (lambda record: record.__setitem__("sources", []), r"example\.sources"),
        (
            lambda record: record["sources"].append(copy.deepcopy(record["sources"][0])),
            r"example\.sources.*duplicate",
        ),
        (
            lambda record: record["sources"][0].__setitem__("type", "website"),
            r"example\.sources.*type",
        ),
        (
            lambda record: record["sources"][0].__setitem__("type", []),
            r"example\.sources.*type",
        ),
        (
            lambda record: record["sources"][0].__setitem__("url", "http://example.test"),
            r"example\.sources.*url",
        ),
        (
            lambda record: record["sources"][0].__setitem__("url", "https://"),
            r"example\.sources.*url",
        ),
        (
            lambda record: record["sources"][0].__setitem__("accessed", "2026-7-18"),
            r"example\.sources.*accessed",
        ),
    ],
)
def test_validate_evidence_rejects_representative_invalid_values(registry, ledger, mutation, context):
    registry_record = copy.deepcopy(next(iter(registry.values())))
    ledger_record = copy.deepcopy(ledger[registry_record["clock_name"]])
    registry_record["clock_name"] = "example"
    ledger_record["clock_name"] = "example"
    invalid_registry = {"example": registry_record}
    invalid_ledger = {"example": ledger_record}
    mutation(ledger_record)

    with pytest.raises(ValueError, match=context):
        validate_evidence(invalid_registry, invalid_ledger)


@pytest.mark.parametrize(
    ("mutation", "context"),
    [
        (
            lambda record: record.__setitem__("typo_field", "unexpected"),
            r"example\.record.*unknown.*typo_field",
        ),
        (
            lambda record: record["fields"].__setitem__(
                "typo_field", copy.deepcopy(record["fields"]["year"])
            ),
            r"example\.fields.*unknown.*typo_field",
        ),
        (
            lambda record: record["fields"]["year"].__setitem__("typo_field", "unexpected"),
            r"example\.year.*unknown.*typo_field",
        ),
    ],
)
def test_validate_evidence_rejects_unknown_schema_fields(
    registry, ledger, mutation, context
):
    registry_record = copy.deepcopy(next(iter(registry.values())))
    ledger_record = copy.deepcopy(ledger[registry_record["clock_name"]])
    registry_record["clock_name"] = "example"
    ledger_record["clock_name"] = "example"
    mutation(ledger_record)

    with pytest.raises(ValueError, match=context):
        validate_evidence({"example": registry_record}, {"example": ledger_record})


@pytest.mark.parametrize(
    "access_issues",
    [
        "not a list",
        [""],
        [3],
    ],
)
def test_validate_evidence_rejects_invalid_access_issue_shape(
    registry, ledger, access_issues
):
    registry_record = copy.deepcopy(next(iter(registry.values())))
    clock_name = registry_record["clock_name"]
    ledger_record = copy.deepcopy(ledger[clock_name])
    ledger_record["access_issues"] = access_issues

    with pytest.raises(ValueError, match=rf"{clock_name}\.access_issues"):
        validate_evidence({clock_name: registry_record}, {clock_name: ledger_record})


def test_validate_evidence_rejects_resolved_field_called_unresolved(
    registry, ledger
):
    registry_record = copy.deepcopy(next(iter(registry.values())))
    clock_name = registry_record["clock_name"]
    ledger_record = copy.deepcopy(ledger[clock_name])
    ledger_record["access_issues"] = ["The training_target remains unresolved."]
    assert ledger_record["fields"]["training_target"]["status"] != "unresolved"

    with pytest.raises(
        ValueError, match=rf"{clock_name}\.access_issues.*training_target.*resolved"
    ):
        validate_evidence({clock_name: registry_record}, {clock_name: ledger_record})


@pytest.mark.parametrize(
    ("status", "source_type"),
    [
        ("paper-confirmed", "code"),
        ("supplement-confirmed", "paper"),
        ("code-confirmed", "paper"),
        ("author-confirmed", "paper"),
    ],
)
def test_resolved_evidence_status_must_match_source_type(registry, ledger, status, source_type):
    registry_record = copy.deepcopy(next(iter(registry.values())))
    clock_name = registry_record["clock_name"]
    ledger_record = copy.deepcopy(ledger[clock_name])
    evidence = ledger_record["fields"]["year"]
    ledger_record["reviewer"] = "reviewer"
    ledger_record["sources"].append(
        {
            "id": "mismatched-source",
            "type": source_type,
            "url": "https://example.org/source",
            "accessed": "2026-07-18",
        }
    )
    evidence["source_id"] = "mismatched-source"
    evidence["status"] = status
    evidence["locator"] = "page 1"

    with pytest.raises(ValueError, match=rf"{clock_name}\.year.*{status}.*{source_type}"):
        validate_evidence({clock_name: registry_record}, {clock_name: ledger_record})


def test_author_confirmed_evidence_accepts_author_communication_source(registry, ledger):
    registry_record = copy.deepcopy(next(iter(registry.values())))
    clock_name = registry_record["clock_name"]
    ledger_record = copy.deepcopy(ledger[clock_name])
    evidence = ledger_record["fields"]["year"]
    ledger_record["reviewer"] = "metadata-audit"
    ledger_record["sources"].append(
        {
            "id": "author-clarification",
            "type": "author communication",
            "url": "https://github.com/lcamillo/CpGPT",
            "accessed": "2026-07-18",
        }
    )
    evidence.update(
        {
            "status": "author-confirmed",
            "source_id": "author-clarification",
            "source_text": "Direct author clarification",
            "locator": "Direct author clarification in Codex task, 2026-07-18",
        }
    )

    validate_evidence({clock_name: registry_record}, {clock_name: ledger_record})


def test_resolved_evidence_rejects_provisional_assignment(registry, ledger):
    registry_record = copy.deepcopy(next(iter(registry.values())))
    clock_name = registry_record["clock_name"]
    ledger_record = copy.deepcopy(ledger[clock_name])
    ledger_record["reviewer"] = "unassigned"
    ledger_record["fields"]["year"]["status"] = "paper-confirmed"

    with pytest.raises(ValueError, match=rf"{clock_name}\..*reviewer"):
        validate_evidence({clock_name: registry_record}, {clock_name: ledger_record})


def test_resolved_evidence_rejects_provisional_locator(registry, ledger):
    registry_record = copy.deepcopy(next(iter(registry.values())))
    clock_name = registry_record["clock_name"]
    ledger_record = copy.deepcopy(ledger[clock_name])
    ledger_record["reviewer"] = "reviewer"
    ledger_record["fields"]["year"]["status"] = "paper-confirmed"
    ledger_record["fields"]["year"]["locator"] = "pending source audit"

    with pytest.raises(ValueError, match=rf"{clock_name}\.year.*locator"):
        validate_evidence({clock_name: registry_record}, {clock_name: ledger_record})


def test_resolved_evidence_rejects_provisional_source_text(registry, ledger):
    registry_record = copy.deepcopy(next(iter(registry.values())))
    clock_name = registry_record["clock_name"]
    ledger_record = copy.deepcopy(ledger[clock_name])
    evidence = ledger_record["fields"]["year"]
    ledger_record["reviewer"] = "reviewer"
    evidence["status"] = "paper-confirmed"
    evidence["locator"] = "page 1"
    evidence["source_text"] = "pending source audit"

    with pytest.raises(ValueError, match=rf"{clock_name}\.year.*source_text"):
        validate_evidence({clock_name: registry_record}, {clock_name: ledger_record})


def test_final_evidence_sample_corrections_are_synchronized(registry, ledger):
    mammalian = registry["mammalianlifespan"]
    mammalian_evidence = ledger["mammalianlifespan"]
    assert mammalian["year"] == 2024
    assert mammalian["doi"] == "https://doi.org/10.1126/sciadv.adm7273"
    assert mammalian["journal"] == "Science Advances"
    assert mammalian["platform"] == ["Horvath MammalMethylChip40"]
    assert mammalian_evidence["fields"]["platform"]["source_text"] == (
        "All data were generated using the mammalian methylation array "
        "(HorvathMammalMethylChip40), which provides high sequencing depth of "
        "highly conserved CpGs in mammals."
    )

    neusin = registry["neusin"]
    neusin_evidence = ledger["neusin"]
    assert neusin["citation"].endswith("Aging 16 (2024): 13452–13504.")
    assert neusin_evidence["fields"]["n_features"]["source_text"] == (
        "The official Neu-SinCoef.rda object contains 673 rows: one (Intercept) "
        "row and 672 non-intercept CpG coefficient rows."
    )

    intrin = registry["intrinclock"]
    intrin_evidence = ledger["intrinclock"]
    assert intrin["n_features"] == 380
    assert "article reports 381 CpGs" in intrin["notes"]
    assert intrin_evidence["fields"]["n_features"]["source_id"] == "intrin-official-code"
    assert intrin_evidence["fields"]["n_features"]["note"].startswith(
        "The article repeatedly reports 381 CpGs"
    )

    twelve_names = sorted(
        name
        for name in registry
        if name.startswith("twelvecelldeconvolutebloodepic")
    )
    assert len(twelve_names) == 12
    for clock_name in twelve_names:
        entry = registry[clock_name]
        evidence = ledger[clock_name]
        assert entry["training_target"] == ["cell-type proportions"]
        assert evidence["fields"]["training_target"]["source_text"].startswith(
            "IDOL used artificial-mixture ground truth"
        )
        assert evidence["fields"]["population"]["source_text"].startswith(
            "Reference cells were isolated from 41 male and 15 female"
        )
        assert evidence["access_issues"] == []
        assert not {
            "nature-paper",
            "biorxiv-v6",
            "reporting-summary",
        } & {source["id"] for source in evidence["sources"]}

    six_cell_names = [
        name for name in registry if "sixcell" in name
    ]
    assert all(
        registry[name]["training_target"] != ["cell-type proportions"]
        for name in six_cell_names
    )

    exact_training_comment = (
        'model.metadata["training_target"] = ["cell-type proportions"]  '
        "# Paper: IDOL used artificial-mixture ground truth with known, "
        "prespecified proportions of the 12 cell types and optimized libraries "
        "by comparing deconvolution estimates with those known proportions."
    )
    exact_population_comment = (
        'model.metadata["population"] = "adults"  # Paper: Reference cells were '
        "isolated from 41 male and 15 female anonymous healthy donors with mean "
        "age 32.2 years and range 19–58 years, spanning multiple self-identified "
        "ancestries."
    )
    for clock_name in twelve_names:
        source_notebook = json.loads(
            (ROOT / "clocks" / "notebooks" / f"{clock_name}.ipynb").read_text(
                encoding="utf-8"
            )
        )
        docs_notebook = json.loads(
            (
                ROOT
                / "docs"
                / "source"
                / "clock_notebooks"
                / f"{clock_name}.ipynb"
            ).read_text(encoding="utf-8")
        )
        source_lines = [
            line.rstrip("\n")
            for cell in source_notebook["cells"]
            for line in cell.get("source", [])
        ]
        assert exact_training_comment in source_lines
        assert exact_population_comment in source_lines
        assert docs_notebook == source_notebook

    intrin_source = (
        ROOT / "clocks" / "notebooks" / "intrinclock.ipynb"
    ).read_text(encoding="utf-8")
    assert (
        "At lambda.min, the official serialized cv.glmnet model contains 380 "
        "non-zero CpG coefficients, and pyaging contains the identical "
        "380-probe set."
    ) in intrin_source
