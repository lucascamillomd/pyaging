import io
from concurrent.futures import ThreadPoolExecutor

import pytest
from rich.console import Console

from pyaging.logger._live import ClockRunDisplay, SimpleStep, display_enabled


@pytest.fixture
def hf_progress_settings():
    """Restore the public HF settings touched by the progress tests."""
    from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars

    was_disabled = are_progress_bars_disabled()
    enable_progress_bars()
    try:
        yield
    finally:
        (disable_progress_bars if was_disabled else enable_progress_bars)()


def _forced_console(buffer):
    return Console(file=buffer, force_terminal=True, width=100, color_system=None)


def test_display_enabled_tracks_verbose():
    assert display_enabled(True) is True
    assert display_enabled(False) is False
    assert display_enabled(0) is False


def test_clock_run_display_full_lifecycle_collapses_to_summary():
    buffer = io.StringIO()
    display = ClockRunDisplay(["horvath2013", "altumage"], "cpu", console=_forced_console(buffer))
    with display:
        display.start_clock("horvath2013")
        display.stage("horvath2013", "predicting")
        display.finish_clock("horvath2013")
        display.start_clock("altumage")
        display.warn("altumage", "research use only")
        display.finish_clock("altumage")
        display.finish(n_samples=32)
    output = buffer.getvalue()

    assert "predict_age" in output
    assert "horvath2013" in output
    assert "altumage" in output
    assert "32 samples" in output
    assert "research use only" in output


def test_clock_run_display_marks_running_clock_failed_on_exception():
    buffer = io.StringIO()
    display = ClockRunDisplay(["horvath2013"], "cpu", console=_forced_console(buffer))
    try:
        with display:
            display.start_clock("horvath2013")
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert display.rows["horvath2013"]["status"] == "failed"
    assert "failed" in buffer.getvalue()


def test_simple_step_prints_summary_line():
    buffer = io.StringIO()
    step = SimpleStep("downloading data.pkl", console=_forced_console(buffer))
    with step:
        pass
    step.done("example data at pyaging_data/data.pkl")

    assert "example data at pyaging_data/data.pkl" in buffer.getvalue()


def test_disabled_display_renders_nothing():
    buffer = io.StringIO()
    console = _forced_console(buffer)
    display = ClockRunDisplay(["horvath2013"], "cpu", console=console, enabled=False)
    with display:
        display.start_clock("horvath2013")
        display.finish_clock("horvath2013")
        display.finish(n_samples=5)
    step = SimpleStep("quiet work", console=console, enabled=False)
    with step:
        step.done("should not appear")

    assert buffer.getvalue() == ""


def test_display_logger_routes_warnings_and_drops_info():
    from pyaging.logger._live import DisplayLogger

    captured = []
    shim = DisplayLogger(captured.append)
    shim.info("The preprocessing method is scale", indent_level=2)
    shim.start_progress("Check features started")
    shim.log_time()
    shim.warning("⚠️ 12% of features missing and imputed with defaults", indent_level=2)

    assert captured == ["12% of features missing and imputed with defaults"]


def test_pipeline_warnings_persist_in_the_summary():
    buffer = io.StringIO()
    display = ClockRunDisplay(["horvath2013"], "cpu", console=_forced_console(buffer))
    with display:
        display.start_clock("horvath2013")
        display.warn("horvath2013", "12% of features missing and imputed with defaults")
        display.finish_clock("horvath2013")
        display.finish(n_samples=32)

    assert "12% of features missing and imputed with defaults" in buffer.getvalue()


def test_running_clock_shows_batch_progress():
    buffer = io.StringIO()
    console = _forced_console(buffer)
    display = ClockRunDisplay(["horvath2013"], "cpu", console=console)
    display.start_clock("horvath2013", "predicting")
    display.progress("horvath2013", 3, 10)
    console.print(display)

    assert "3/10" in buffer.getvalue()


def test_plain_region_prints_intro_and_final_for_captured_output():
    buffer = io.StringIO()
    plain_console = Console(file=buffer, force_terminal=False, force_jupyter=False, width=100)
    step = SimpleStep("downloading data.pkl", console=plain_console)
    with step:
        step.done("example data at pyaging_data/data.pkl")
    output = buffer.getvalue()

    assert "downloading data.pkl" in output
    assert "example data at pyaging_data/data.pkl" in output


def test_simple_step_payload_renders_under_completion_line():
    buffer = io.StringIO()
    step = SimpleStep("loading metadata", console=_forced_console(buffer))
    with step:
        step.payload("horvath2013 · altumage · phenoage")
        step.done("3 clocks available")
    output = buffer.getvalue()

    assert "3 clocks available" in output
    assert "horvath2013 · altumage · phenoage" in output


def test_disabled_step_suppresses_payloads_too():
    buffer = io.StringIO()
    step = SimpleStep("loading metadata", console=_forced_console(buffer), enabled=False)
    with step:
        step.payload("should not appear")
        step.done("also hidden")

    assert buffer.getvalue() == ""


def test_quiet_hf_bars_preserves_other_library_progress_settings(hf_progress_settings):
    from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars

    from pyaging.logger._live import quiet_hf_bars

    disable_progress_bars("other-library")
    with quiet_hf_bars(False):
        assert are_progress_bars_disabled("other-library")

    assert are_progress_bars_disabled("other-library")
    assert not are_progress_bars_disabled()


def test_quiet_hf_bars_does_not_silence_other_threads(hf_progress_settings):
    from huggingface_hub.utils import are_progress_bars_disabled

    from pyaging.logger._live import quiet_hf_bars

    with quiet_hf_bars(False), ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(are_progress_bars_disabled).result() is False


def test_silent_step_disables_its_hf_download_bar(monkeypatch, capsys):
    from huggingface_hub.utils import tqdm

    from pyaging.logger._live import live_step
    from pyaging.utils._hf import download_hf_file

    def download_with_progress(**kwargs):
        progress_class = kwargs.get("tqdm_class", tqdm)
        with progress_class(total=1, desc="hub download", disable=False) as bar:
            bar.update(1)
        return "cached.pt"

    monkeypatch.setattr("pyaging.utils._hf.hf_hub_download", download_with_progress)

    with live_step("loading data", False):
        assert download_hf_file("data.pt") == "cached.pt"

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
