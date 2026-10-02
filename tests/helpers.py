"""Small shared helpers; each call returns fresh, mutable state."""

from pathlib import Path

import pytest
import torch

WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "clocks" / "weights"


def load_local_clock(name):
    """Load local build output on CPU; skip when it is absent."""
    path = WEIGHTS_DIR / f"{name}.pt"
    if not path.exists():
        pytest.skip(f"{path} is build output; generate it by running clocks/notebooks/{name}.ipynb")
    return torch.load(path, weights_only=False, map_location="cpu")


class RecordingLogger:
    """Record warnings while suppressing routine pipeline progress."""

    def __init__(self):
        self.warnings = []

    def warning(self, message, indent_level=2):
        self.warnings.append(message)

    def info(self, message, indent_level=2):
        pass

    error = start_progress = finish_progress = info
