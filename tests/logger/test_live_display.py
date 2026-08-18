import io

from rich.console import Console

from pyaging.logger._live import ClockRunDisplay, SimpleStep, live_display_enabled


def _forced_console(buffer):
    return Console(file=buffer, force_terminal=True, width=100, color_system=None)


def test_live_display_disabled_when_output_is_not_interactive():
    plain = Console(file=io.StringIO(), force_terminal=False, force_jupyter=False)
    assert live_display_enabled(True, console=plain) is False
    assert live_display_enabled(False, console=plain) is False


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


def test_live_display_enabled_when_interactive_and_verbose():
    interactive = Console(file=io.StringIO(), force_terminal=True)
    assert live_display_enabled(False, console=interactive) is False
    assert live_display_enabled(True, console=interactive) is True


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
