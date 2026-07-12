import re
from pathlib import Path

S3_HOST_PATTERN = re.compile(
    r"https://pyaging\.s3(?:[.-][a-z0-9-]+)*\.amazonaws\.com"
)
NOTEBOOK_ROOTS = (
    Path("clocks/notebooks"),
    Path("docs/source/clock_notebooks"),
)


def test_notebooks_do_not_reference_s3():
    repository_root = Path(__file__).resolve().parents[1]
    offenders = [
        notebook.relative_to(repository_root).as_posix()
        for root in NOTEBOOK_ROOTS
        for notebook in sorted((repository_root / root).glob("*.ipynb"))
        if S3_HOST_PATTERN.search(notebook.read_text())
    ]

    assert offenders == []
