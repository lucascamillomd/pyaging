import logging

import pytest

from pyaging.logger import Logger, LoggerManager, main_set_level


def test_constructing_logger_preserves_dependency_logging_configuration(monkeypatch):
    dependency = logging.getLogger("anndata")
    monkeypatch.setattr(dependency, "level", logging.WARNING)
    monkeypatch.setattr(dependency, "propagate", True)

    Logger("pyaging.test.configuration")

    assert dependency.level == logging.WARNING
    assert dependency.propagate is True


def test_main_logger_does_not_reconfigure_dynamo(monkeypatch):
    other_package = logging.getLogger("dynamo")
    monkeypatch.setattr(other_package, "level", logging.ERROR)
    monkeypatch.setattr(LoggerManager.main_logger.logger, "level", logging.INFO)

    main_set_level(logging.DEBUG)

    assert other_package.level == logging.ERROR
    assert LoggerManager.main_logger.logger.level == logging.DEBUG


def test_namespaced_context_restores_namespace_after_exception():
    logger = Logger("pyaging.test.namespace")

    with pytest.raises(RuntimeError), logger.namespaced_context("temporary"):
        raise RuntimeError("interrupted")

    assert logger.namespace == "pyaging.test.namespace"


def test_empty_file_download_progress_does_not_divide_by_zero():
    logger = Logger("pyaging.test.empty-download")

    logger.request_report_hook(0, 8192, 0)

    assert logger.report_hook_percent_state is None


def test_progress_logger_accepts_unsized_iterators():
    assert list(LoggerManager.progress_logger(iter([1, 2, 3]))) == [1, 2, 3]
