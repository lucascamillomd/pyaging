"""Rich-based live progress display for interactive terminals and notebooks.

When ``verbose=True`` and the run is interactive (Jupyter or a TTY), pyaging
shows a live step display instead of plain log lines: pending steps as hollow
circles, the active step as a spinner with its current stage and progress
bars, finished steps as check marks with timings, and a compact summary once
the run completes. Pipeline warnings (missing features, research-only clocks,
...) surface on the display and persist in the summary. Non-interactive runs
keep the classic text logs, and ``verbose=False`` stays silent.

In notebooks the animation is pushed through an IPython display handle that
updates one regular output in place - not an ipywidgets container, whose
background is frontend-controlled (white in VS Code dark mode) and which
leaves an empty output slot behind. The summary replaces the animation in the
same output, so a finished cell holds exactly one themed block. Terminals use
rich's native Live with a transient region and a printed summary.
"""

import contextlib
import sys
import threading
import time

from rich.console import Console, Group
from rich.jupyter import _render_segments
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

TEAL = "#178fa0"
TEAL_BRIGHT = "#7adfec"
SAND = "#efc53f"
GREEN = "#2fbf71"
RED = "#e5484d"
MUTED = "#8a93a1"
TRACK = "#3a4351"

# Claude Code's spinner: ping-pong sparkle frames at 120 ms per frame, and
# the black-circle glyph for completed items with an elbow for sub-lines.
_SPARKLE_BASE = ["·", "✢", "✳", "✶", "✻", "✽"] if sys.platform == "darwin" else ["·", "✢", "*", "✶", "✻", "✽"]
SPARKLE_FRAMES = _SPARKLE_BASE + _SPARKLE_BASE[::-1]
DOT = "⏺" if sys.platform == "darwin" else "●"
ELBOW = "⎿"

_console = Console()


class Sparkle:
    """Claude-Code-style sparkle spinner: time-based ping-pong frames."""

    def __init__(self, style: str = SAND):
        self.style = style

    def __rich_console__(self, console, options):
        frame = int(time.perf_counter() * 1000 / 120) % len(SPARKLE_FRAMES)
        yield Text(SPARKLE_FRAMES[frame], style=self.style)

    def __rich_measure__(self, console, options):
        from rich.measure import Measurement

        return Measurement(1, 1)


class Shimmer:
    """Status text with a highlight sweeping across it, one char per 200 ms.

    The glimmer position runs right-to-left over the text width plus a
    20-column overshoot, so the highlight periodically leaves the text and
    the message rests between sweeps - same cycle Claude Code uses.
    """

    def __init__(self, message: str, base: str = TEAL, glow: str = f"bold {TEAL_BRIGHT}"):
        self.message = message
        self.base = base
        self.glow = glow

    def __rich_console__(self, console, options):
        width = len(self.message)
        cycle = width + 20
        position = width + 10 - (int(time.perf_counter() * 1000 / 200) % cycle)
        text = Text()
        for index, char in enumerate(self.message):
            text.append(char, style=self.glow if abs(index - position) <= 1 else self.base)
        yield text

    def __rich_measure__(self, console, options):
        from rich.measure import Measurement

        width = len(self.message)
        return Measurement(width, width)


def live_display_enabled(verbose, console: Console | None = None) -> bool:
    """Whether the live display should replace plain text logs."""
    active = console or _console
    return bool(verbose and (active.is_jupyter or active.is_interactive))


def _bar(completed: float, total: float | None, width: int = 24, pulse: bool = False):
    return ProgressBar(
        total=total,
        completed=completed,
        width=width,
        pulse=pulse,
        style=TRACK,
        complete_style=TEAL,
        finished_style=TEAL,
        pulse_style=TEAL,
    )


class DisplayLogger:
    """Duck-typed stand-in for pyaging's Logger that feeds a live display.

    While the live display is active the pipeline functions log through this
    shim: warnings and errors surface on the display, info-level messages are
    dropped (the display already narrates the stages).
    """

    def __init__(self, warn_sink):
        self._warn = warn_sink

    def _record(self, message):
        self._warn(str(message).replace("⚠️", "").strip())

    def warning(self, message, *args, **kwargs):
        self._record(message)

    def error(self, message, *args, **kwargs):
        self._record(message)

    def __getattr__(self, name):
        # Every other Logger method (info, start_progress, log_time, ...) is a no-op
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

    def __init__(self, clock_names, device: str, console: Console | None = None):
        self.console = console or _console
        self.device = device
        self.order = list(clock_names)
        self.rows = {
            name: {"status": "pending", "stage": "", "seconds": None, "t0": None, "progress": None, "warnings": []}
            for name in self.order
        }
        self.started = time.perf_counter()
        self._summary = None
        self._spinner = Sparkle()
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
            final = Text.assemble((f"{DOT} ", f"bold {RED}"), (f"predict_age failed at {label}", RED))
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

    def warn(self, name: str, message: str):
        if message:
            self.rows[name]["warnings"].append(message)

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
        timing = Table.grid(padding=(0, 2))
        for name in done:
            seconds = self.rows[name]["seconds"]
            timing.add_row(
                Text(name, style="bold"),
                Text(f"{seconds:.1f}s" if seconds is not None else "", style=MUTED, justify="right"),
            )
        lines = [
            Text.assemble(
                (f"{DOT} ", GREEN),
                (f"{len(done)} clock{'s' if len(done) != 1 else ''}", "bold"),
                (f" · {n_samples} samples · {elapsed:.1f}s · {self.device}", MUTED),
            ),
            Text(),
            timing,
            Text(),
            Text.assemble((f"{ELBOW} ", MUTED), ("results in adata.obs · clock metadata in adata.uns", MUTED)),
        ]
        for name in self.order:
            for message in self.rows[name]["warnings"]:
                lines.append(Text.assemble((f"{ELBOW} ", MUTED), ("⚠ ", SAND), (f"{name}: ", "bold"), (message, MUTED)))
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
                cells = [Text("  "), self._spinner, Text(name, style="bold"), Shimmer(row["stage"])]
                if row["progress"] and row["progress"][1] > 1:
                    completed, total = row["progress"]
                    cells.append(_bar(completed, total, width=16))
                    cells.append(Text(f"{completed}/{total}", style=MUTED))
                grid.add_row(*cells)
                yield grid
            elif row["status"] == "failed":
                yield Text.assemble((f"  {DOT} ", f"bold {RED}"), (name, "bold"), (" failed", RED))
            else:
                seconds = f" ({row['seconds']:.1f}s)" if row["seconds"] is not None else ""
                yield Text.assemble((f"  {DOT} ", GREEN), (name, "bold"), (seconds, MUTED))
                for message in row["warnings"]:
                    yield Text.assemble((f"    {ELBOW} ", MUTED), ("⚠ ", SAND), (message, MUTED))
            if row["status"] == "running":
                for message in row["warnings"]:
                    yield Text.assemble((f"    {ELBOW} ", MUTED), ("⚠ ", SAND), (message, MUTED))


class SimpleStep:
    """Animated pulse bar while working; a summary line replaces it after.

    ``update(label)`` changes the stage text mid-run. ``done(message)`` inside
    the ``with`` block sets the completion line. ``warn(message)`` keeps a
    pipeline warning visible under the final line. Used without entering the
    context (e.g. cache hits), ``done`` prints directly.
    """

    def __init__(self, label: str, console: Console | None = None):
        self.console = console or _console
        self.label = label
        self.warnings = []
        self.started = time.perf_counter()
        self._spinner = Sparkle()
        self._region = None
        self._final = None

    def __enter__(self):
        self._region = _make_region(self.console, self)
        self._region.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            final = Text.assemble((f"{DOT} ", f"bold {RED}"), (f"{self.label} failed", RED))
        else:
            final = self._final
        self._region.close(final)
        self._region = None
        return False

    def update(self, label: str):
        self.label = label

    def warn(self, message: str):
        if message:
            self.warnings.append(str(message).strip())

    def done(self, message: str):
        text = Text.assemble((f"{DOT} ", GREEN), (message, ""))
        if self.warnings:
            elbows = (Text.assemble((f"  {ELBOW} ", MUTED), ("⚠ ", SAND), (m, MUTED)) for m in self.warnings)
            text = Group(text, *elbows)
        if self._region is not None:
            self._final = text
        else:
            self.console.print(text)

    def __rich_console__(self, console, options):
        elapsed = time.perf_counter() - self.started
        grid = Table.grid(padding=(0, 1))
        grid.add_row(
            self._spinner,
            Shimmer(self.label, base="default", glow=f"bold {TEAL_BRIGHT}"),
            _bar(0, None, width=18, pulse=True),
            Text(f"{elapsed:.0f}s", style=MUTED),
        )
        yield grid
        for message in self.warnings:
            yield Text.assemble((f"  {ELBOW} ", MUTED), ("⚠ ", SAND), (message, MUTED))
