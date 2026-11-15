"""Publish downloaded files only after their contents are complete."""

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def atomic_output_path(destination):
    """Yield a temporary sibling path, then atomically replace the destination."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as temporary_dir:
        temporary_path = Path(temporary_dir) / destination.name
        yield temporary_path
        temporary_path.replace(destination)
