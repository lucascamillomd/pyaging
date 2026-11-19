import pytest


@pytest.fixture(autouse=True)
def offline_weight_prefetch(monkeypatch):
    """Keep multi-clock tests offline: their clocks are built in-process, so
    resolving weight files on the Hub would only add network calls."""
    monkeypatch.setattr("pyaging.predict._pred.prefetch_clock_weights", lambda *args, **kwargs: None)
