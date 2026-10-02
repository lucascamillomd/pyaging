"""Keep executable sources and user documentation on the current data host."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "src",
    "clocks",
    "docs",
    "tutorials",
    ".github",
    "Makefile",
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "AGENTS.md",
    "CHANGELOG.md",
)
S3_REFERENCE = re.compile(
    r"pyaging\.s3(?:[.-][a-z0-9-]+)*\.amazonaws\.com"
    r"|https?://s3(?:[.-][a-z0-9-]+)*\.amazonaws\.com(?:\.cn)?/pyaging(?=[/?#\s\"']|$)"
    r"|s3://pyaging(?=[/?#\s\"']|$)"
    r"|\bboto3\b|\bawscli\b|\baws\s+(?:--\S+\s+\S+\s+)*s3(?:api)?\s",
    re.IGNORECASE,
)


def test_sources_use_current_data_hosting():
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--", *SOURCE_PATHS],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split("\0")
    offenders = []
    for name in filter(None, tracked):
        file = ROOT / name
        if not file.is_file():
            continue
        try:
            content = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if S3_REFERENCE.search(content):
            offenders.append(name)
    assert offenders == []
