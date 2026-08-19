import io

import pytest
from rich.console import Console

import pyaging.logger._live as live_module


@pytest.fixture(autouse=True)
def non_interactive_display_console(monkeypatch):
    """Keep the live display off by default so test behavior does not depend
    on whether pytest runs attached to a TTY (e.g. `pytest -s`). Tests that
    exercise the display pass their own console explicitly."""
    monkeypatch.setattr(
        live_module,
        "_console",
        Console(file=io.StringIO(), force_terminal=False, force_jupyter=False),
    )
