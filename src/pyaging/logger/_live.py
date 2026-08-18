"""Rich-based live progress display for interactive terminals and notebooks.

When a run is interactive (Jupyter or a TTY), pyaging shows a live step
display instead of plain log lines: pending steps as hollow circles, the
active step as a spinner with its current stage, finished steps as check
marks with timings. The animation runs in a transient region that removes
itself on completion (in notebooks it lives in a widget whose background is
frontend-controlled, so nothing themed lingers), and the final summary is
printed as a regular, theme-aware output. Non-interactive runs (pipes, CI,
pytest) keep the classic text logger.
"""

import time

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

TEAL = "#178fa0"
SAND = "#efc53f"
GREEN = "#2fbf71"
RED = "#e5484d"
MUTED = "#8a93a1"

_console = Console()


def live_display_enabled(verbose: bool, console: Console | None = None) -> bool:
    """Whether the live display should replace plain text logs."""
    active = console or _console
    return bool(verbose and (active.is_jupyter or active.is_interactive))


class ClockRunDisplay:
    """Live step tree for a predict_age run: one row per clock."""

    def __init__(self, clock_names, device: str, console: Console | None = None):
        self.console = console or _console
        self.device = device
        self.order = list(clock_names)
        self.rows = {
            name: {"status": "pending", "stage": "", "seconds": None, "note": None, "t0": None} for name in self.order
        }
        self.started = time.perf_counter()
        self._summary = None
        self._spinner = Spinner("dots", style=TEAL)
        self._live = Live(self, console=self.console, refresh_per_second=12, transient=True)

    # -- lifecycle ----------------------------------------------------------
    def __enter__(self):
        self._live.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._live.stop()
        if exc_type is not None:
            failed = [n for n in self.order if self.rows[n]["status"] == "running"]
            for name in failed:
                self.rows[name]["status"] = "failed"
            label = failed[0] if failed else "run"
            self.console.print(Text.assemble(("✗ ", f"bold {RED}"), (f"predict_age failed at {label}", RED)))
        elif self._summary is not None:
            self.console.print(self._summary)
        return False

    def start_clock(self, name: str, stage: str = "loading weights"):
        row = self.rows[name]
        row["status"] = "running"
        row["stage"] = stage
        row["t0"] = time.perf_counter()

    def stage(self, name: str, stage: str):
        self.rows[name]["stage"] = stage

    def note(self, name: str, note: str):
        self.rows[name]["note"] = note

    def finish_clock(self, name: str):
        row = self.rows[name]
        row["status"] = "done"
        row["seconds"] = time.perf_counter() - row["t0"] if row["t0"] else None
        row["stage"] = ""

    def finish(self, n_samples: int):
        """Compose the summary panel printed once the live region clears."""
        elapsed = time.perf_counter() - self.started
        done = [n for n in self.order if self.rows[n]["status"] == "done"]
        timing = Table.grid(padding=(0, 2))
        for name in done:
            seconds = self.rows[name]["seconds"]
            timing.add_row(
                Text(name, style="bold"),
                Text(f"{seconds:.1f}s" if seconds is not None else "", style=MUTED, justify="right"),
            )
        lines = [
            Text.assemble(
                ("✓ ", f"bold {GREEN}"),
                (f"{len(done)} clock{'s' if len(done) != 1 else ''}", "bold"),
                (f" · {n_samples} samples · {elapsed:.1f}s · {self.device}", MUTED),
            ),
            timing,
            Text("results in adata.obs · clock metadata in adata.uns", style=MUTED),
        ]
        for name in self.order:
            if self.rows[name]["note"]:
                lines.append(Text.assemble(("⚠ ", SAND), (f"{name}: {self.rows[name]['note']}", MUTED)))
        self._summary = Panel(
            Group(*lines),
            title=Text("pyaging · predict_age", style=TEAL),
            title_align="left",
            border_style=TEAL,
            padding=(0, 2),
            expand=False,
        )

    # -- rendering ----------------------------------------------------------
    def __rich_console__(self, console, options):
        header = Table.grid(padding=(0, 1))
        header.add_row(
            self._spinner,
            Text.assemble(
                ("predict_age", "bold"),
                (f" · {len(self.order)} clock{'s' if len(self.order) != 1 else ''} · {self.device}", MUTED),
            ),
        )
        yield header
        for name in self.order:
            row = self.rows[name]
            if row["status"] == "pending":
                yield Text.assemble(("  ○ ", MUTED), (name, MUTED))
            elif row["status"] == "running":
                grid = Table.grid(padding=(0, 1))
                grid.add_row(Text("  "), self._spinner, Text(name, style="bold"), Text(row["stage"], style=TEAL))
                yield grid
            elif row["status"] == "failed":
                yield Text.assemble(("  ✗ ", f"bold {RED}"), (name, "bold"), (" failed", RED))
            else:
                seconds = f" {row['seconds']:.1f}s" if row["seconds"] is not None else ""
                note = f"  ⚠ {row['note']}" if row["note"] else ""
                yield Text.assemble(("  ✓ ", f"bold {GREEN}"), (name, "bold"), (seconds, MUTED), (note, SAND))


class SimpleStep:
    """Transient spinner while working, one theme-aware line after."""

    def __init__(self, label: str, console: Console | None = None):
        self.console = console or _console
        self.label = label
        self._live = None

    def __enter__(self):
        spinner = Spinner("dots", text=Text(self.label, style=TEAL), style=TEAL)
        self._live = Live(spinner, console=self.console, refresh_per_second=12, transient=True)
        self._live.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._live.stop()
        if exc_type is not None:
            self.console.print(Text.assemble(("✗ ", f"bold {RED}"), (f"{self.label} failed", RED)))
        return False

    def done(self, message: str):
        self.console.print(Text.assemble(("✓ ", f"bold {GREEN}"), (message, "")))
