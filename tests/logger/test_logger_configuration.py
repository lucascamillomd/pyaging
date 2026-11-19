import logging

import pytest

import pyaging.logger._logger as logger_module
from pyaging.logger import Logger, LoggerManager


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_nested_progress_reports_each_steps_own_elapsed_time(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(logger_module.time, "time", clock)
    logger = Logger("pyaging.test.nested-progress")
    monkeypatch.setattr(logger.logger, "disabled", True)

    logger.start_progress("outer")
    clock.now = 10.0
    logger.start_progress("inner")
    clock.now = 11.0
    logger.finish_progress("inner")
    inner = logger.time_passed
    clock.now = 12.0
    logger.finish_progress("outer")

    assert inner == pytest.approx(1.0)
    assert logger.time_passed == pytest.approx(12.0)


def test_unmatched_finish_progress_still_measures_from_last_log_time(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(logger_module.time, "time", clock)
    logger = Logger("pyaging.test.unmatched-progress")
    monkeypatch.setattr(logger.logger, "disabled", True)

    clock.now = 5.0
    logger.log_time()
    clock.now = 7.5
    logger.finish_progress("step")

    assert logger.time_passed == pytest.approx(2.5)


def test_gen_logger_leaves_a_user_configured_logger_alone(monkeypatch):
    user_logger = logging.getLogger("pyaging-test-user-app")
    monkeypatch.setattr(user_logger, "level", logging.WARNING)
    monkeypatch.setattr(user_logger, "propagate", True)
    monkeypatch.setattr(user_logger, "handlers", [])

    LoggerManager.gen_logger("pyaging-test-user-app")

    assert user_logger.level == logging.WARNING
    assert user_logger.propagate is True
    assert user_logger.handlers == []


def test_gen_logger_leaves_the_root_logger_alone(monkeypatch):
    root = logging.getLogger()
    monkeypatch.setattr(root, "level", logging.ERROR)
    monkeypatch.setattr(root, "propagate", True)
    monkeypatch.setattr(root, "handlers", [])

    LoggerManager.gen_logger("")

    assert root.level == logging.ERROR
    assert root.propagate is True
    assert root.handlers == []


def test_gen_logger_keeps_a_level_the_user_set_on_the_package_logger(monkeypatch):
    package_logger = logging.getLogger("pyaging")
    monkeypatch.setattr(package_logger, "level", logging.ERROR)

    LoggerManager.gen_logger("pyaging")

    assert package_logger.level == logging.ERROR


def test_explicit_level_is_still_applied_to_an_existing_logger(monkeypatch):
    existing = logging.getLogger("pyaging-test-explicit-level")
    monkeypatch.setattr(existing, "level", logging.WARNING)

    Logger("pyaging-test-explicit-level", level=logging.ERROR)

    assert existing.level == logging.ERROR


def test_gen_logger_still_configures_a_logger_it_creates():
    logger = LoggerManager.gen_logger("pyaging-test-fresh-logger-name")

    assert logger.logger.level == logging.INFO
    assert logger.logger.propagate is False
    assert isinstance(logger.logger_stream_handler, logging.StreamHandler)


def test_progress_reporting_works_when_the_first_handler_is_not_a_stream(monkeypatch):
    name = "pyaging.test.null-handler"
    stream = logging.StreamHandler()
    monkeypatch.setattr(logging.getLogger(name), "handlers", [logging.NullHandler(), stream])
    logger = Logger(name)
    monkeypatch.setattr(logger.logger, "disabled", True)

    logger.report_progress(percent=50)
    logger.start_progress("step")
    logger.finish_progress("step")

    assert logger.logger_stream_handler is stream


def test_progress_reporting_works_without_any_stream_handler(monkeypatch):
    name = "pyaging.test.only-null-handler"
    monkeypatch.setattr(logging.getLogger(name), "handlers", [logging.NullHandler()])
    logger = Logger(name)

    logger.report_progress(percent=50)
    logger.start_progress("step")
    logger.finish_progress("step")
    logger.request_report_hook(1, 8192, 100)


def test_single_block_download_reports_completion_once(monkeypatch):
    logger = Logger("pyaging.test.single-block")
    reported = []
    monkeypatch.setattr(logger, "report_progress", lambda **kwargs: reported.append(kwargs))

    logger.request_report_hook(0, 8192, 100)
    logger.request_report_hook(1, 8192, 100)

    assert len(reported) == 1


def test_multi_block_download_reports_intermediate_and_final_progress(monkeypatch):
    logger = Logger("pyaging.test.multi-block")
    reported = []
    monkeypatch.setattr(
        logger, "report_progress", lambda **kwargs: reported.append(kwargs.get("percent") or kwargs["count"])
    )

    for block in range(4):
        logger.request_report_hook(block, 100, 300)

    assert reported == [pytest.approx(100 / 3), pytest.approx(200 / 3), 100]
