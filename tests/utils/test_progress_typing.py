from collections.abc import Callable
from typing import get_type_hints

from pyaging.utils import progress


def test_progress_is_annotated_as_returning_a_decorator():
    return_type = get_type_hints(progress)["return"]

    assert return_type is not type(None)
    assert getattr(return_type, "__origin__", return_type) is Callable
