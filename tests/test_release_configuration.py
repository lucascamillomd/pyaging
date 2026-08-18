import hashlib
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = (ROOT / "Makefile").read_text(encoding="utf-8")
HF_README = ROOT / "clocks" / "huggingface" / "README.md"
GITIGNORE = (ROOT / ".gitignore").read_text(encoding="utf-8")
WORKFLOW_DIRECTORY = ROOT / ".github" / "workflows"
READTHEDOCS_CONFIG = ROOT / ".readthedocs.yaml"
SDIST_ONLY_INCLUDE = {
    "src/pyaging",
    "README.md",
    "LICENSE",
    "pyproject.toml",
}
FORBIDDEN_SENTINELS = {
    ".venv/sentinel.txt",
    ".worktrees/sentinel.txt",
    "hf_static_data/sentinel.txt",
    "pyaging_data/sentinel.txt",
    "clocks/weights/sentinel.txt",
    "clocks/metadata/sentinel.txt",
    "docs/_build/sentinel.txt",
    "docs/_static/.cache/sentinel.txt",
    "docs/_static/all_clock_metadata.pt",
    ".pytest_cache/sentinel.txt",
    ".ruff_cache/sentinel.txt",
    "dist/sentinel.txt",
    "build/sentinel.txt",
}
EXCLUDED_DOCS_ASSETS = {
    "docs/_static/clocks.json",
    "docs/_static/clock_glossary.csv",
}


def test_release_build_ships_only_the_package_and_metadata():
    with (ROOT / "pyproject.toml").open("rb") as pyproject:
        configuration = tomllib.load(pyproject)

    targets = configuration["tool"]["hatch"]["build"]["targets"]
    assert set(targets["sdist"]["only-include"]) == SDIST_ONLY_INCLUDE
    assert targets["wheel"]["packages"] == ["src/pyaging"]
    assert configuration["tool"]["hatch"]["version"]["path"] == "src/pyaging/__init__.py"


def _build_fixture_sdist(project, output_directory):
    subprocess.run(
        [
            "uv",
            "build",
            "--sdist",
            "--offline",
            "--no-build-isolation",
            "--out-dir",
            str(output_directory),
        ],
        cwd=project,
        env={**os.environ, "VIRTUAL_ENV": sys.prefix},
        check=True,
        capture_output=True,
        text=True,
    )
    return next(output_directory.glob("*.tar.gz"))


def _relative_sdist_members(sdist):
    with tarfile.open(sdist, "r:gz") as archive:
        return {name.split("/", 1)[1] for name in archive.getnames() if "/" in name}


def test_release_sdist_excludes_ignored_sentinels_and_generated_docs_assets(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    shutil.copy2(ROOT / "pyproject.toml", project / "pyproject.toml")
    shutil.copy2(ROOT / "README.md", project / "README.md")
    shutil.copy2(ROOT / "LICENSE", project / "LICENSE")

    package = project / "src" / "pyaging"
    package.mkdir(parents=True)
    package.joinpath("__init__.py").write_text('__version__ = "0.3.1"\n', encoding="utf-8")

    clean_sdist = _build_fixture_sdist(project, tmp_path / "clean-dist")

    for relative_path in FORBIDDEN_SENTINELS | EXCLUDED_DOCS_ASSETS:
        path = project / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("must not ship\n", encoding="utf-8")

    populated_sdist = _build_fixture_sdist(project, tmp_path / "populated-dist")
    members = _relative_sdist_members(populated_sdist)

    assert members.isdisjoint(FORBIDDEN_SENTINELS)
    assert members.isdisjoint(EXCLUDED_DOCS_ASSETS)
    assert members >= {"src/pyaging/__init__.py", "pyproject.toml", "README.md", "LICENSE"}
    assert hashlib.sha256(clean_sdist.read_bytes()).digest() == hashlib.sha256(populated_sdist.read_bytes()).digest()


def _load_workflow(name):
    return yaml.load(
        WORKFLOW_DIRECTORY.joinpath(name).read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )


def _workflow_steps(workflow):
    return [step for job in workflow["jobs"].values() for step in job["steps"]]


def _target_recipe(target):
    lines = MAKEFILE.splitlines()
    header_index = next(index for index, line in enumerate(lines) if line.startswith(f"{target}:"))
    commands = []
    for line in lines[header_index + 1 :]:
        if not line.startswith("\t"):
            break
        commands.append(line.removeprefix("\t"))
    return lines[header_index], commands


def test_readthedocs_installs_the_project_with_its_docs_group_via_uv():
    configuration = yaml.safe_load(READTHEDOCS_CONFIG.read_text(encoding="utf-8"))
    install = next(
        entry for entry in configuration["python"]["install"] if isinstance(entry, dict) and entry.get("method") == "uv"
    )

    assert configuration["sphinx"]["configuration"] == "docs/source/conf.py"
    assert install["command"] == "sync"
    assert "docs" in install["groups"]


def test_docs_target_runs_sphinx_in_managed_environment():
    _, commands = _target_recipe("docs")

    assert "uv run make -C docs html" in commands


def test_makefile_uses_only_hf_publish_targets():
    assert "upload-to-s3" not in MAKEFILE
    legacy_s3_cli = "aws" + " s3"
    assert legacy_s3_cli not in MAKEFILE
    assert "HF_REPO_ID ?= lucascamillomd/pyaging-data" in MAKEFILE
    assert "verify-hf-auth:" in MAKEFILE
    assert "verify-hf-data-repo-public:" in MAKEFILE
    assert "create-hf-data-repo:" in MAKEFILE
    assert "upload-clocks-to-hf:" in MAKEFILE
    assert "upload-static-data-to-hf:" in MAKEFILE


def test_makefile_has_hf_release_defaults():
    assert "VERSION ?= v0.3.1" in MAKEFILE
    assert "HF_REPO_ID ?= lucascamillomd/pyaging-data" in MAKEFILE
    assert "HF_REPO_OWNER ?= lucascamillomd" in MAKEFILE
    assert "HF_STATIC_DIR ?= hf_static_data" in MAKEFILE
    assert "hf_static_data/" in GITIGNORE.splitlines()


def test_hf_targets_guard_creation_and_upload_order():
    assert "hf auth whoami --format json" in MAKEFILE
    assert 'json.load(sys.stdin)["user"]' in MAKEFILE
    assert 'json.load(sys.stdin)["name"]' not in MAKEFILE
    assert 'if [ "$$account" != "$(HF_REPO_OWNER)" ]' in MAKEFILE
    assert 'hf repos create "$(HF_REPO_ID)" --type model --public --exist-ok' in MAKEFILE
    assert 'hf upload "$(HF_REPO_ID)" clocks/huggingface/README.md README.md --type model' in MAKEFILE
    assert 'hf upload "$(HF_REPO_ID)" "$(HF_STATIC_DIR)/repo" . --type model' in MAKEFILE

    weights = 'hf upload "$(HF_REPO_ID)" clocks/weights . --type model'
    metadata = 'hf upload "$(HF_REPO_ID)" clocks/metadata/all_clock_metadata.pt all_clock_metadata.pt --type model'
    assert MAKEFILE.index(weights) < MAKEFILE.index(metadata)
    assert 'hf models info "$(HF_REPO_ID)" --format json' in MAKEFILE


def test_hf_uploads_require_a_public_repository():
    assert "verify-hf-data-repo-public: verify-hf-auth" in MAKEFILE
    assert 'json.load(sys.stdin)["private"]' in MAKEFILE
    assert "upload-clocks-to-hf: verify-hf-data-repo-public" in MAKEFILE
    assert "upload-static-data-to-hf: verify-hf-data-repo-public" in MAKEFILE

    _, create_recipe = _target_recipe("create-hf-data-repo")
    create = next(index for index, command in enumerate(create_recipe) if "hf repos create" in command)
    public_check = create_recipe.index("$(MAKE) verify-hf-data-repo-public")
    card_upload = next(index for index, command in enumerate(create_recipe) if "README.md" in command)
    assert create < public_check < card_upload


def test_release_runs_steps_sequentially_in_one_recipe():
    expected = {
        "release": [
            "version",
            "lint",
            "format",
            "update",
            "build",
            "install",
            "update-clocks-notebooks",
            "update-all-clocks",
            "process-tutorials",
            "test",
            "test-tutorials",
            "docs",
            "upload-clocks-to-hf",
            "tag-hf-data-repo",
            "commit",
            "tag",
        ],
        "release-slim": [
            "version",
            "lint",
            "format",
            "update",
            "build",
            "install",
            "update-all-clocks",
            "test",
            "docs",
            "upload-clocks-to-hf",
            "tag-hf-data-repo",
            "commit",
            "tag",
        ],
    }
    for target, steps in expected.items():
        header, commands = _target_recipe(target)
        assert header == f"{target}:"
        assert commands[:-1] == [f"$(MAKE) {step}" for step in steps]
        assert commands[-1].startswith('@echo "Release $(VERSION)')


def test_parallel_release_dry_run_preserves_publish_sequence():
    result = subprocess.run(
        ["make", "-n", "-j4", "release-slim", "VERSION=v0.3.1"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout
    assert output.index("Running gold standard tests") < output.index("Uploading changed clock weights")
    assert output.index("Building documentation") < output.index("Uploading changed clock weights")
    assert output.index("Uploading changed clock weights") < output.index("Committing and pushing changes")
    assert output.index("Committing and pushing changes") < output.index("Creating and pushing tag")


def test_hf_repository_card_documents_layout_updates_and_access():
    text = HF_README.read_text(encoding="utf-8")
    assert "public repository" in text
    assert "repository root" in text
    assert "downloaded through the standard Hugging Face cache" in text
    assert "`main` branch is the live data release" in text
    assert "Weights are uploaded before aggregate metadata" in text
    assert "need no Hugging Face token" in text


def test_hf_repository_card_documents_security_licensing_and_ownership():
    text = HF_README.read_text(encoding="utf-8")
    assert "license: other" in text
    assert "mixed-provenance" in text
    assert "research-only" in text
    assert "commercial terms" in text
    assert "torch.load(..., weights_only=False)" in text
    assert "lucascamillomd/pyaging" in text
    assert "maintained solely" in text
    assert "lucascamillomd" in text


def test_release_workflow_is_tag_gated_and_publishes_after_verify():
    workflow = _load_workflow("release.yaml")

    assert workflow["on"] == {"push": {"tags": ["v*"]}}
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["publish"]["needs"] == "verify"
    assert workflow["jobs"]["github-release"]["needs"] == "publish"
    assert workflow["jobs"]["github-release"]["permissions"] == {"contents": "write"}


def test_release_verifies_version_tests_and_distribution_before_publish():
    workflow = _load_workflow("release.yaml")
    verify_steps = workflow["jobs"]["verify"]["steps"]
    verify_commands = "\n".join(step["run"] for step in verify_steps if "run" in step)

    assert any(step.get("uses", "").startswith("astral-sh/setup-uv@") for step in verify_steps)
    assert "pip install uv" not in verify_commands
    assert "uv sync --locked" in verify_commands
    assert "src/pyaging/__init__.py" in verify_commands
    assert 'test "${GITHUB_REF_NAME}" = "v${version}"' in verify_commands
    assert "git fetch origin main --no-tags" in verify_commands
    assert 'git merge-base --is-ancestor "$GITHUB_SHA" origin/main' in verify_commands
    assert "not full_catalog and not online" in verify_commands
    assert "uv build" in verify_commands
    assert "twine check dist/*" in verify_commands


def test_release_publishes_with_trusted_publishing():
    workflow = _load_workflow("release.yaml")
    workflow_text = WORKFLOW_DIRECTORY.joinpath("release.yaml").read_text(encoding="utf-8")
    publish = workflow["jobs"]["publish"]
    publish_action = next(step for step in publish["steps"] if step["uses"].startswith("pypa/"))

    environment = publish["environment"]
    assert environment == "pypi" or environment["name"] == "pypi"
    assert publish["permissions"] == {"id-token": "write"}
    assert "PYPI_API_TOKEN" not in workflow_text
    assert "password" not in publish_action.get("with", {})


def test_release_checkout_has_no_persisted_credentials_and_full_history():
    workflow = _load_workflow("release.yaml")
    checkout = next(
        step for step in workflow["jobs"]["verify"]["steps"] if step["uses"].startswith("actions/checkout@")
    )

    assert checkout["with"]["persist-credentials"] == "false"
    assert checkout["with"]["fetch-depth"] == "0"


def test_ci_excludes_large_and_online_tests():
    workflow = _load_workflow("ci.yml")
    triggers = workflow["on"]
    unit_commands = "\n".join(step["run"] for step in workflow["jobs"]["unit"]["steps"] if "run" in step)

    assert set(triggers) == {"push", "pull_request", "schedule", "workflow_dispatch"}
    assert triggers["push"] == {"branches": ["main"]}
    assert triggers["pull_request"] == ""
    assert all(entry["cron"] for entry in triggers["schedule"])
    assert triggers["workflow_dispatch"] == ""
    assert workflow["permissions"] == {"contents": "read"}
    assert "not full_catalog and not online" in unit_commands


def test_ci_covers_supported_platforms_and_tutorials():
    workflow = _load_workflow("ci.yml")
    matrix = workflow["jobs"]["unit"]["strategy"]["matrix"]
    tutorials = workflow["jobs"]["tutorials"]
    tutorial_commands = "\n".join(step["run"] for step in tutorials["steps"] if "run" in step)

    assert matrix == {
        "os": ["ubuntu-latest", "macos-latest"],
        "python-version": ["3.11", "3.12", "3.13", "3.14"],
    }
    assert tutorials["if"] == "github.event_name == 'workflow_dispatch' || github.event_name == 'schedule'"
    assert "uv run --no-sync pytest --nbmake tutorials/" in tutorial_commands
    assert "--ignore=tutorials/tutorial_cpgptgrimage3.ipynb" in tutorial_commands


def test_ci_runs_pinned_workflow_security_check():
    workflow = _load_workflow("ci.yml")
    security_commands = "\n".join(
        step["run"] for step in workflow["jobs"]["workflow-security"]["steps"] if "run" in step
    )

    assert "uvx zizmor==1.26.1 --pedantic .github/workflows" in security_commands


def test_workflow_actions_are_commit_pinned_and_checkouts_discard_credentials():
    for name in ("ci.yml", "release.yaml"):
        workflow = _load_workflow(name)
        action_steps = [step for step in _workflow_steps(workflow) if "uses" in step]
        assert action_steps
        for step in action_steps:
            assert re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[0-9a-f]{40}", step["uses"])
            if step["uses"].startswith("actions/checkout@"):
                assert step["with"]["persist-credentials"] == "false"


def test_workflows_have_named_jobs_and_concurrency_controls():
    for name in ("ci.yml", "release.yaml"):
        workflow = _load_workflow(name)
        assert workflow["concurrency"]["group"]
        assert workflow["concurrency"]["cancel-in-progress"] in {"true", "false"}
        assert all(job.get("name") for job in workflow["jobs"].values())


def test_legacy_chained_workflows_are_removed():
    for name in ("build.yml", "publish.yml", "test.yml", "release.yml"):
        assert not (WORKFLOW_DIRECTORY / name).exists()
