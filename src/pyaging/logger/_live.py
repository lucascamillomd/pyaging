"""Rich-based live progress display for interactive terminals and notebooks.

When a run is interactive (Jupyter or a TTY), pyaging shows a live step
display instead of plain log lines: pending steps as hollow circles, the
active step as a spinner with its current stage, finished steps as check
marks with timings. On completion the whole region collapses into a compact
summary panel, so only the result stays in the output. Non-interactive runs
(pipes, CI, pytest) keep the classic text logger.
"""

import time

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

INK = "#132238"
TEAL = "#178fa0"
SAND = "#efc53f"
MUTED = "grey58"

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
        self._spinner = Spinner("dots", style=TEAL)
        self._live = Live(self, console=self.console, refresh_per_second=12, transient=False)

    # -- lifecycle ----------------------------------------------------------
    def __enter__(self):
        self._live.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            for row in self.rows.values():
                if row["status"] == "running":
                    row["status"] = "failed"
            self._live.update(self, refresh=True)
        self._live.stop()
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
        """Collapse the live region into the final summary panel."""
        elapsed = time.perf_counter() - self.started
        done = [n for n in self.order if self.rows[n]["status"] == "done"]
        timing = Text()
        for i, name in enumerate(done):
            if i:
                timing.append(" · ", style=MUTED)
            timing.append(name, style="bold")
            seconds = self.rows[name]["seconds"]
            if seconds is not None:
                timing.append(f" {seconds:.1f}s", style=MUTED)
        lines = [
            Text.assemble(
                ("✓ ", "bold green"),
                (f"{len(done)} clock{'s' if len(done) != 1 else ''}", "bold"),
                (f" · {n_samples} samples · {elapsed:.1f}s · {self.device}", MUTED),
            ),
            timing,
            Text("results in adata.obs · clock metadata in adata.uns", style=MUTED),
        ]
        notes = [(n, self.rows[n]["note"]) for n in self.order if self.rows[n]["note"]]
        for name, note in notes:
            lines.append(Text.assemble(("⚠ ", SAND), (f"{name}: {note}", MUTED)))
        panel = Panel(Group(*lines), title="pyaging · predict_age", title_align="left", border_style=TEAL, expand=False)
        self._live.update(panel, refresh=True)

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
                yield Text.assemble(("  ✗ ", "bold red"), (name, "bold"), (" failed", "red"))
            else:
                seconds = f" {row['seconds']:.1f}s" if row["seconds"] is not None else ""
                note = f"  ⚠ {row['note']}" if row["note"] else ""
                yield Text.assemble(("  ✓ ", "bold green"), (name, "bold"), (seconds, MUTED), (note, SAND))


class SimpleStep:
    """Spinner-while-working, one-line summary after, for small operations."""

    def __init__(self, label: str, console: Console | None = None):
        self.console = console or _console
        self.label = label
        self._status = None

    def __enter__(self):
        self._status = self.console.status(Text(self.label, style=TEAL), spinner="dots", spinner_style=TEAL)
        self._status.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._status.stop()
        return False

    def done(self, message: str):
        self.console.print(Text.assemble(("✓ ", "bold green"), (message, "")))
