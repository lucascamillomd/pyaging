import pytest

from pyaging.utils._hf import _clear_resolved_weights
from pyaging.utils._utils import _clear_clock_metadata_cache


@pytest.fixture(autouse=True)
def fresh_process_caches():
    """Isolate tests from weights and metadata resolved earlier in the process."""
    _clear_resolved_weights()
    _clear_clock_metadata_cache()
    yield
    _clear_resolved_weights()
    _clear_clock_metadata_cache()
