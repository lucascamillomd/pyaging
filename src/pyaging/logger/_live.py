"""Rich-based live progress display for interactive terminals and notebooks.

pyaging's user-facing functions take a ``verbose`` level (bools keep working
since ``True == 1``):

- ``0`` - silent.
- ``1`` - compact live display: an animated step tree with progress bars that
  collapses into a summary. Falls back to the classic text logs when output
  is not interactive (pipes, CI).
- ``2`` - the live display plus every pipeline log message as persistent
  detail lines under the step that produced it.

In notebooks the animation is pushed through an IPython display handle that
updates one regular output in place - not an ipywidgets container, whose
background is frontend-controlled (white in VS Code dark mode) and which
leaves an empty output slot behind. The summary replaces the animation in the
same output, so a finished cell holds exactly one themed block. Terminals use
rich's native Live with a transient region and a printed summary.
"""

import contextlib
import threading
import time

from rich.console import Console, Group
from rich.jupyter import _render_segments
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

TEAL = "#178fa0"
SAND = "#efc53f"
GREEN = "#2fbf71"
RED = "#e5484d"
MUTED = "#8a93a1"
TRACK = "#3a4351"

_console = Console()


def verbosity(verbose) -> int:
    """Normalize the verbose argument (bool or int) to a 0-2 level."""
    return max(0, min(2, int(verbose)))


def live_display_enabled(verbose, console: Console | None = None) -> bool:
    """Whether the live display should replace plain text logs."""
    active = console or _console
    return bool(verbosity(verbose) >= 1 and (active.is_jupyter or active.is_interactive))


def _bar(completed: float, total: float | None, width: int = 24, pulse: bool = False, uniform: bool = False):
    return ProgressBar(
        total=total,
        completed=completed,
        width=width,
        pulse=pulse,
        style=TRACK,
        complete_style=TEAL,
        finished_style=TEAL if uniform else GREEN,
        pulse_style=TEAL,
    )


class DisplayLogger:
    """Duck-typed stand-in for pyaging's Logger that feeds a live display.

    At verbose level 2 the pipeline functions log through this shim, so every
    message the classic logs would print becomes a detail line under the step
    that produced it.
    """

    def __init__(self, sink):
        self._sink = sink

    def _record(self, message):
        self._sink(str(message).strip())

    def info(self, message, *args, **kwargs):
        self._record(message)

    def warning(self, message, *args, **kwargs):
        self._record(message)

    def error(self, message, *args, **kwargs):
        self._record(message)

    def start_progress(self, message, *args, **kwargs):
        self._record(message)

    def __getattr__(self, name):
        # Any other Logger method (log_time, report_progress, done, ...) is a no-op
        def _noop(*args, **kwargs):
            return None

        return _noop


class _JupyterRegion:
    """Animates a renderable by updating one IPython display handle in place."""

    def __init__(self, console: Console, renderable):
        self.console = console
        self.renderable = renderable
        self._handle = None
        self._stop = threading.Event()
        self._thread = None

    def _html(self, renderable):
        segments = list(self.console.render(renderable, self.console.options))
        return _render_segments(segments)

    def start(self):
        from IPython.display import HTML, display

        self._handle = display(HTML(self._html(self.renderable)), display_id=True)
        self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._thread.start()

    def _refresh_loop(self):
        from IPython.display import HTML

        while not self._stop.wait(0.08):
            # a failed frame must never kill the run
            with contextlib.suppress(Exception):
                self._handle.update(HTML(self._html(self.renderable)))

    def close(self, final=None):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._handle is not None:
            from IPython.display import HTML

            self._handle.update(HTML(self._html(final) if final is not None else ""))


class _TerminalRegion:
    """Animates a renderable with rich Live; the summary is printed after."""

    def __init__(self, console: Console, renderable):
        self.console = console
        self._live = Live(renderable, console=console, refresh_per_second=12, transient=True)

    def start(self):
        self._live.start()

    def close(self, final=None):
        self._live.stop()
        if final is not None:
            self.console.print(final)


def _make_region(console: Console, renderable):
    return _JupyterRegion(console, renderable) if console.is_jupyter else _TerminalRegion(console, renderable)


class ClockRunDisplay:
    """Live step tree for a predict_age run: one row per clock."""

    def __init__(self, clock_names, device: str, console: Console | None = None, detailed: bool = False):
        self.console = console or _console
        self.device = device
        self.detailed = detailed
        self.order = list(clock_names)
        self.rows = {
            name: {
                "status": "pending",
                "stage": "",
                "seconds": None,
                "note": None,
                "t0": None,
                "progress": None,
                "details": [],
            }
            for name in self.order
        }
        self.started = time.perf_counter()
        self._summary = None
        self._spinner = Spinner("dots", style=TEAL)
        self._region = _make_region(self.console, self)

    # -- lifecycle ----------------------------------------------------------
    def __enter__(self):
        self._region.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            failed = [n for n in self.order if self.rows[n]["status"] == "running"]
            for name in failed:
                self.rows[name]["status"] = "failed"
            label = failed[0] if failed else "run"
            final = Text.assemble(("✗ ", f"bold {RED}"), (f"predict_age failed at {label}", RED))
        else:
            final = self._summary
        self._region.close(final)
        return False

    def start_clock(self, name: str, stage: str = "loading weights"):
        row = self.rows[name]
        row["status"] = "running"
        row["stage"] = stage
        row["t0"] = time.perf_counter()

    def stage(self, name: str, stage: str):
        row = self.rows[name]
        row["stage"] = stage
        row["progress"] = None

    def progress(self, name: str, completed: int, total: int):
        self.rows[name]["progress"] = (completed, total)

    def note(self, name: str, note: str):
        self.rows[name]["note"] = note

    def detail(self, name: str, message: str):
        if message:
            self.rows[name]["details"].append(message)

    def finish_clock(self, name: str):
        row = self.rows[name]
        row["status"] = "done"
        row["seconds"] = time.perf_counter() - row["t0"] if row["t0"] else None
        row["stage"] = ""
        row["progress"] = None

    def finish(self, n_samples: int):
        """Compose the summary that replaces the animation on completion."""
        elapsed = time.perf_counter() - self.started
        done = [n for n in self.order if self.rows[n]["status"] == "done"]
        slowest = max((self.rows[n]["seconds"] or 0) for n in done) if done else 0
        lines = [
            Text.assemble(
                ("✓ ", f"bold {GREEN}"),
                (f"{len(done)} clock{'s' if len(done) != 1 else ''}", "bold"),
                (f" · {n_samples} samples · {elapsed:.1f}s · {self.device}", MUTED),
            ),
            Text(),
        ]
        if self.detailed:
            for name in done:
                seconds = self.rows[name]["seconds"]
                lines.append(
                    Text.assemble(
                        ("✓ ", f"bold {GREEN}"),
                        (name, "bold"),
                        (f" {seconds:.1f}s" if seconds is not None else "", MUTED),
                    )
                )
                for message in self.rows[name]["details"]:
                    lines.append(Text(f"  · {message}", style=MUTED))
        else:
            timing = Table.grid(padding=(0, 2))
            for name in done:
                seconds = self.rows[name]["seconds"]
                share = (seconds or 0) / slowest if slowest else 0
                timing.add_row(
                    Text(name, style="bold"),
                    Text(f"{seconds:.1f}s" if seconds is not None else "", style=MUTED, justify="right"),
                    _bar(share, 1.0, width=14, uniform=True),
                )
            lines.append(timing)
        lines.append(Text())
        lines.append(Text("results in adata.obs · clock metadata in adata.uns", style=MUTED))
        for name in self.order:
            if self.rows[name]["note"]:
                lines.append(Text.assemble(("⚠ ", SAND), (f"{name}: {self.rows[name]['note']}", MUTED)))
        self._summary = Panel(
            Group(*lines),
            title=Text("pyaging · predict_age", style=f"bold {TEAL}"),
            title_align="left",
            border_style=TEAL,
            padding=(0, 2),
            expand=False,
        )

    # -- rendering ----------------------------------------------------------
    def __rich_console__(self, console, options):
        done_count = sum(1 for n in self.order if self.rows[n]["status"] == "done")
        elapsed = time.perf_counter() - self.started
        header = Table.grid(padding=(0, 1))
        header.add_row(
            self._spinner,
            Text.assemble(("predict_age", "bold"), (f" · {self.device}", MUTED)),
            _bar(done_count, len(self.order), width=18),
            Text(f"{done_count}/{len(self.order)} clocks · {elapsed:.0f}s", style=MUTED),
        )
        yield header
        for name in self.order:
            row = self.rows[name]
            if row["status"] == "pending":
                yield Text.assemble(("  ○ ", MUTED), (name, MUTED))
            elif row["status"] == "running":
                grid = Table.grid(padding=(0, 1))
                cells = [Text("  "), self._spinner, Text(name, style="bold"), Text(row["stage"], style=TEAL)]
                if row["progress"] and row["progress"][1] > 1:
                    completed, total = row["progress"]
                    cells.append(_bar(completed, total, width=16))
                    cells.append(Text(f"{completed}/{total}", style=MUTED))
                grid.add_row(*cells)
                yield grid
                if self.detailed:
                    for message in row["details"][-4:]:
                        yield Text(f"      · {message}", style=MUTED)
            elif row["status"] == "failed":
                yield Text.assemble(("  ✗ ", f"bold {RED}"), (name, "bold"), (" failed", RED))
            else:
                seconds = f" {row['seconds']:.1f}s" if row["seconds"] is not None else ""
                note = f"  ⚠ {row['note']}" if row["note"] else ""
                yield Text.assemble(("  ✓ ", f"bold {GREEN}"), (name, "bold"), (seconds, MUTED), (note, SAND))
                if self.detailed:
                    for message in row["details"]:
                        yield Text(f"      · {message}", style=MUTED)


class SimpleStep:
    """Animated pulse bar while working; a summary line replaces it after.

    ``update(label)`` changes the stage text mid-run. ``done(message)`` inside
    the ``with`` block sets the completion line. Used without entering the
    context (e.g. cache hits), ``done`` prints directly.
    """

    def __init__(self, label: str, console: Console | None = None, detailed: bool = False):
        self.console = console or _console
        self.label = label
        self.detailed = detailed
        self.details = []
        self.started = time.perf_counter()
        self._spinner = Spinner("dots", style=TEAL)
        self._region = None
        self._final = None

    def __enter__(self):
        self._region = _make_region(self.console, self)
        self._region.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            final = Text.assemble(("✗ ", f"bold {RED}"), (f"{self.label} failed", RED))
        else:
            final = self._final
        self._region.close(final)
        self._region = None
        return False

    def update(self, label: str):
        self.label = label

    def detail(self, message: str):
        if message:
            self.details.append(str(message).strip())

    def done(self, message: str):
        text = Text.assemble(("✓ ", f"bold {GREEN}"), (message, ""))
        if self.detailed and self.details:
            text = Group(text, *(Text(f"  · {m}", style=MUTED) for m in self.details))
        if self._region is not None:
            self._final = text
        else:
            self.console.print(text)

    def __rich_console__(self, console, options):
        elapsed = time.perf_counter() - self.started
        grid = Table.grid(padding=(0, 1))
        grid.add_row(
            self._spinner,
            Text(self.label, style="bold"),
            _bar(0, None, width=18, pulse=True),
            Text(f"{elapsed:.0f}s", style=MUTED),
        )
        yield grid
        if self.detailed:
            for message in self.details[-4:]:
                yield Text(f"  · {message}", style=MUTED)
