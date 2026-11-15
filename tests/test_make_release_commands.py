"""Exercise release commands in a disposable project without network writes."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def release_project(tmp_path):
    shutil.copy2(ROOT / "Makefile", tmp_path / "Makefile")
    package = tmp_path / "src" / "pyaging"
    package.mkdir(parents=True)
    package.joinpath("__init__.py").write_text('__version__ = "1.2.3"\n')
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "args = [arg for arg in sys.argv[1:] if arg not in ('run', '--no-sync')]\n"
        "if args[0] == 'python':\n"
        "    os.execv(sys.executable, [sys.executable, *args[1:]])\n"
        "sys.exit(42)\n"
    )
    uv.chmod(0o755)
    jupyter = bin_dir / "jupyter"
    jupyter.write_text("#!/bin/sh\nexit 42\n")
    jupyter.chmod(0o755)
    return tmp_path, {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}


def test_default_version_does_not_downgrade_package(release_project):
    project, env = release_project
    result = subprocess.run(["make", "version"], cwd=project, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (project / "src/pyaging/__init__.py").read_text() == '__version__ = "1.2.3"\n'


def test_version_target_updates_package(release_project):
    project, env = release_project
    result = subprocess.run(["make", "version", "VERSION=v1.2.4"], cwd=project, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (project / "src/pyaging/__init__.py").read_text() == '__version__ = "1.2.4"\n'


def test_failed_clock_notebook_stops_release(release_project):
    project, env = release_project
    notebooks = project / "clocks/notebooks"
    notebooks.mkdir(parents=True)
    (notebooks / "failing.ipynb").write_text("{}")
    result = subprocess.run(["make", "update-clocks-notebooks"], cwd=project, env=env, capture_output=True, text=True)
    assert result.returncode != 0
