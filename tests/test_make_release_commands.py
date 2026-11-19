"""Exercise release commands in a disposable project without network writes.

``uv`` and bare ``jupyter`` are stubs that record their arguments. ``uv`` never
runs the Hugging Face tagging script or any clock script; it only executes the
inline ``python -c`` helpers that bump the version or parse stubbed JSON. ``git``
is the real binary behind a recording wrapper, pointed at a local bare
``origin``, so commits, pushes and tags never leave the temporary directory.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
REAL_GIT = shutil.which("git")
CPGPT_TUTORIAL = "tutorials/tutorial_cpgptgrimage3.ipynb"

UV_STUB = """\
import json, os, sys
argv = sys.argv[1:]
args = [arg for arg in argv if arg not in ("run", "--no-sync")]
hub_script = args[:2] == ["python", "-c"] and "huggingface_hub" in args[2]
entry = ["uv", "hub-tag", os.environ.get("TAG_HF_TAG")] if hub_script else ["uv", *argv]
with open(os.environ["STUB_LOG"], "a") as log:
    log.write(json.dumps({"cwd": os.getcwd(), "cmd": entry}) + "\\n")
fail = os.environ.get("STUB_UV_FAIL")
if fail and fail in " ".join(args):
    sys.exit(42)
if args[:3] == ["hf", "auth", "whoami"]:
    print(json.dumps({"user": "lucascamillomd"}))
elif args[:3] == ["hf", "models", "info"]:
    print(json.dumps({"private": False, "sha": "0" * 40}))
elif args[:2] == ["python", "-c"] and not hub_script:
    os.execv(sys.executable, [sys.executable, *args[1:]])
sys.exit(0)
"""

RECORDER_STUB = """\
import json, os, sys
with open(os.environ["STUB_LOG"], "a") as log:
    log.write(json.dumps({{"cwd": os.getcwd(), "cmd": [{name!r}, *sys.argv[1:]]}}) + "\\n")
target = {target!r}
if target is None:
    sys.exit(42)
os.execv(target, [target, *sys.argv[1:]])
"""

# Every path `make commit` stages must exist or `git add` rejects the pathspec.
PROJECT_FILES = {
    "uv.lock": "",
    "pyproject.toml": "",
    "README.md": "readme\n",
    "CHANGELOG.md": "",
    ".github/workflows/ci.yml": "",
    "tests/test_placeholder.py": "",
    "clocks/notebooks/clock.ipynb": "{}",
    "clocks/metadata/clock_metadata.json": "{}",
    "tutorials/tutorial_basic.ipynb": "{}",
    CPGPT_TUTORIAL: "{}",
    "docs/source/index.rst": "",
    "docs/_static/placeholder.css": "",
    "docs/development/README.md": "",
}


def _write_stub(path, body):
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(0o755)


def _git(project, *args, env):
    return subprocess.run([REAL_GIT, *args], cwd=project, env=env, check=True, capture_output=True, text=True)


@pytest.fixture
def release_project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    shutil.copy2(ROOT / "Makefile", project / "Makefile")
    package = project / "src" / "pyaging"
    package.mkdir(parents=True)
    package.joinpath("__init__.py").write_text('__version__ = "1.2.3"\n')
    for relative_path, content in PROJECT_FILES.items():
        path = project / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_stub(bin_dir / "uv", UV_STUB)
    _write_stub(bin_dir / "jupyter", RECORDER_STUB.format(name="jupyter", target=None))
    _write_stub(bin_dir / "git", RECORDER_STUB.format(name="git", target=REAL_GIT))

    gitconfig = tmp_path / "gitconfig"
    gitconfig.write_text(
        "[user]\n\tname = Release Test\n\temail = release@example.invalid\n[commit]\n\tgpgsign = false\n"
    )
    log = tmp_path / "commands.jsonl"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "GIT_CONFIG_GLOBAL": str(gitconfig),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "STUB_LOG": str(log),
    }
    for variable in ("MAKEFLAGS", "MFLAGS", "MAKELEVEL", "VERSION"):
        env.pop(variable, None)

    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(origin), env=env)
    _git(project, "init", "-b", "main", env=env)
    _git(project, "add", "-A", env=env)
    _git(project, "commit", "-m", "Initial", env=env)
    _git(project, "remote", "add", "origin", str(origin), env=env)
    _git(project, "push", "-u", "origin", "main", env=env)
    return project, origin, env


def _make(project, env, *arguments):
    return subprocess.run(["make", *arguments], cwd=project, env=env, capture_output=True, text=True)


def _commands(env):
    log = Path(env["STUB_LOG"])
    if not log.exists():
        return []
    return [json.loads(line)["cmd"] for line in log.read_text().splitlines()]


def _index(commands, predicate, description):
    matches = [index for index, command in enumerate(commands) if predicate(command)]
    assert matches, f"{description} was never run"
    return matches[0]


def _reject_pushes(origin):
    hook = origin / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\necho 'push rejected by test hook' >&2\nexit 1\n")
    hook.chmod(0o755)


def test_default_version_does_not_downgrade_package(release_project):
    project, _, env = release_project
    result = _make(project, env, "version")
    assert result.returncode == 0, result.stderr
    assert (project / "src/pyaging/__init__.py").read_text() == '__version__ = "1.2.3"\n'


def test_version_target_updates_package(release_project):
    project, _, env = release_project
    result = _make(project, env, "version", "VERSION=v1.2.4")
    assert result.returncode == 0, result.stderr
    assert (project / "src/pyaging/__init__.py").read_text() == '__version__ = "1.2.4"\n'


def test_failed_clock_notebook_stops_release(release_project):
    project, _, env = release_project
    result = _make(project, {**env, "STUB_UV_FAIL": "nbconvert"}, "update-clocks-notebooks")
    assert result.returncode != 0


def test_clock_metadata_is_stamped_with_the_bare_release_version(release_project):
    project, _, env = release_project
    result = _make(project, env, "update-all-clocks", "VERSION=v0.6.3")
    assert result.returncode == 0, result.stderr
    stamps = [command for command in _commands(env) if "update_all_clocks.py" in command]
    assert stamps == [["uv", "run", "python", "update_all_clocks.py", "0.6.3"]]


def _ci_tutorial_ignores():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    commands = " ".join(step["run"] for step in workflow["jobs"]["tutorials"]["steps"] if "run" in step)
    return sorted(token for token in commands.split() if token.startswith("--ignore="))


def test_tutorial_targets_use_the_project_jupyter_and_ci_ignore_list(release_project):
    project, _, env = release_project
    assert _ci_tutorial_ignores() == [f"--ignore={CPGPT_TUTORIAL}"]

    processed = _make(project, env, "process-tutorials")
    tested = _make(project, env, "test-tutorials")

    assert processed.returncode == 0, processed.stdout + processed.stderr
    assert tested.returncode == 0, tested.stdout + tested.stderr
    commands = _commands(env)
    assert not [command for command in commands if command[0] == "jupyter"]
    executed = [command[-1] for command in commands if "nbconvert" in command]
    assert executed == ["tutorial_basic.ipynb"]
    assert all(command[:4] == ["uv", "run", "--no-sync", "jupyter"] for command in commands if "nbconvert" in command)
    nbmake = next(command for command in commands if "--nbmake" in command)
    assert sorted(arg for arg in nbmake if arg.startswith("--ignore=")) == _ci_tutorial_ignores()


def test_commit_succeeds_when_there_is_nothing_to_commit(release_project):
    project, _, env = release_project
    result = _make(project, env, "commit", "VERSION=v1.2.4")
    assert result.returncode == 0, result.stdout + result.stderr
    commands = _commands(env)
    assert not [command for command in commands if command[:2] == ["git", "commit"]]
    assert ["git", "push"] in commands


def test_commit_fails_when_the_push_is_rejected(release_project):
    project, origin, env = release_project
    (project / "README.md").write_text("changed\n")
    _reject_pushes(origin)
    result = _make(project, env, "commit", "VERSION=v1.2.4")
    assert result.returncode != 0
    assert "Git push failed" in result.stdout


@pytest.mark.parametrize("target", ["release", "release-slim"])
def test_release_pushes_code_before_publishing_data_and_tags_last(release_project, target):
    project, origin, env = release_project
    result = _make(project, env, target, "VERSION=v1.2.4")
    assert result.returncode == 0, result.stdout + result.stderr

    commands = _commands(env)
    milestones = [
        _index(commands, lambda c: c[:2] == ["git", "rev-parse"], "branch preflight"),
        _index(commands, lambda c: c[:2] == ["git", "ls-remote"], "remote tag preflight"),
        _index(commands, lambda c: c[:1] == ["uv"], "first local build step"),
        _index(commands, lambda c: c[:2] == ["git", "commit"], "git commit"),
        _index(commands, lambda c: c == ["git", "push"], "git push"),
        _index(commands, lambda c: c[-1:] == ["clocks/hf_repo_sync.py"], "per-clock upload"),
        _index(commands, lambda c: "all_clock_metadata.pt" in c, "aggregate metadata upload"),
        _index(commands, lambda c: c == ["uv", "hub-tag", "v1.2.4"], "data repository tag"),
        _index(commands, lambda c: "--tag-only" in c, "per-clock tags"),
        _index(commands, lambda c: c[:3] == ["git", "tag", "-a"], "git tag"),
        _index(commands, lambda c: c == ["git", "push", "origin", "v1.2.4"], "git tag push"),
    ]
    assert milestones == sorted(milestones)
    assert milestones[-1] == len(commands) - 1
    assert ["uv", "run", "python", "clocks/hf_repo_sync.py", "--tag", "v1.2.4", "--tag-only"] in commands
    assert _git(origin, "tag", "--list", env=env).stdout.split() == ["v1.2.4"]
    assert "1.2.4" in _git(origin, "show", "main:src/pyaging/__init__.py", env=env).stdout


def test_rejected_push_stops_release_before_any_data_is_published(release_project):
    project, origin, env = release_project
    _reject_pushes(origin)
    result = _make(project, env, "release-slim", "VERSION=v1.2.4")
    assert result.returncode != 0

    commands = _commands(env)
    assert ["git", "push"] in commands
    assert not [command for command in commands if "hf" in command or "hub-tag" in command]
    assert not [command for command in commands if "clocks/hf_repo_sync.py" in command]
    assert not [command for command in commands if command[:2] == ["git", "tag"]]


def _checkout_feature_branch(project, origin, env):
    _git(project, "checkout", "-b", "feature", env=env)


def _tag_locally(project, origin, env):
    _git(project, "tag", "v1.2.4", env=env)


def _tag_on_origin_only(project, origin, env):
    _git(project, "tag", "v1.2.4", env=env)
    _git(project, "push", "origin", "v1.2.4", env=env)
    _git(project, "tag", "-d", "v1.2.4", env=env)


@pytest.mark.parametrize(
    ("version", "arrange", "message"),
    [
        ("v1.2.4", _checkout_feature_branch, "main"),
        ("1.2.4", None, "VERSION"),
        ("v1.2", None, "VERSION"),
        ("v1.2.4rc1", None, "VERSION"),
        ("v1.2.4", _tag_locally, "already exists"),
        ("v1.2.4", _tag_on_origin_only, "already exists on origin"),
    ],
    ids=["feature-branch", "bare-version", "two-part", "suffix", "local-tag", "remote-tag"],
)
@pytest.mark.parametrize("target", ["release", "release-slim"])
def test_release_preflight_refuses_before_changing_anything(release_project, target, version, arrange, message):
    project, origin, env = release_project
    if arrange is not None:
        arrange(project, origin, env)

    result = _make(project, env, target, f"VERSION={version}")

    assert result.returncode != 0
    assert message in result.stdout + result.stderr
    assert not [command for command in _commands(env) if command[0] != "git"]
    assert (project / "src/pyaging/__init__.py").read_text() == '__version__ = "1.2.3"\n'
