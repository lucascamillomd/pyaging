"""Explicit, bounded reuse of prepared models across prediction calls."""

from collections import OrderedDict
from numbers import Integral

import torch

from ..utils._hf import get_data_revision


class ClockCache:
    """Keep a bounded number of prepared clocks for repeated datasets.

    Pass the same instance to ``predict_age(..., clock_cache=cache)`` to reuse
    loaded models, including their device transfer and evaluation setup. Keys
    include the lowercase clock name, device, and ``PYAGING_DATA_REVISION``.
    Input matrices and cohort preprocessing are never cached.

    ``maxsize`` is a positive integer, defaulting to 2. A cache miss evicts the
    least recently used clock *before* loading when the cache is full, so a
    failed replacement load may leave one fewer cached model. Models remain
    on their selected device until eviction, ``clear()``, or cache disposal.

    The cache represents a snapshot: call ``clear()`` to pick up changed files
    at a mutable Hugging Face revision such as ``main``. No global cache is
    created. An instance is intended for sequential calls, not shared threads.
    """

    def __init__(self, maxsize: int = 2):
        if isinstance(maxsize, bool) or not isinstance(maxsize, Integral) or maxsize <= 0:
            raise ValueError("maxsize must be a positive integer.")
        self._maxsize = int(maxsize)
        self._models = OrderedDict()

    @property
    def maxsize(self) -> int:
        """Maximum number of prepared models held by this cache."""
        return self._maxsize

    def clear(self) -> None:
        """Release all model references held by this cache."""
        self._models.clear()

    def _get_or_load(self, clock_name, device, load):
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        key = (clock_name.lower(), str(device), get_data_revision())
        if key in self._models:
            self._models.move_to_end(key)
            return self._models[key]
        if len(self._models) >= self.maxsize:
            self._models.popitem(last=False)
        model = load()
        self._models[key] = model
        return model
