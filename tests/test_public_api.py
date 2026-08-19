import importlib
import inspect

import pytest

import pyaging
from pyaging.models import _base_models, _models

SUBPACKAGES = ["data", "logger", "models", "predict", "preprocess", "utils"]


@pytest.mark.parametrize("subpackage", SUBPACKAGES)
def test_all_names_resolve_and_are_unique(subpackage):
    module = importlib.import_module(f"pyaging.{subpackage}")
    exported = module.__all__

    assert len(exported) == len(set(exported)), "duplicate names in __all__"
    missing = [name for name in exported if not hasattr(module, name)]
    assert missing == [], f"__all__ lists names the package does not define: {missing}"


def _public_classes(module):
    return {
        name
        for name, obj in vars(module).items()
        if inspect.isclass(obj) and obj.__module__ == module.__name__ and not name.startswith("_")
    }


def test_models_all_matches_defined_model_classes():
    expected = _public_classes(_models) | _public_classes(_base_models)

    assert set(pyaging.models.__all__) == expected


def test_star_import_exposes_every_export():
    namespace = {}
    exec("from pyaging.models import *", namespace)

    assert set(pyaging.models.__all__) <= set(namespace)
