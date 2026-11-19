"""PAOPAC's scikit-learn and statsmodels dependencies load only when it runs."""

import subprocess
import sys


def test_importing_pyaging_does_not_import_statsmodels():
    code = "import sys, pyaging, pyaging.models._paopac; print('statsmodels' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


def test_paopac_module_has_no_top_level_sklearn_or_statsmodels_import():
    import ast
    import inspect

    from pyaging.models import _paopac

    tree = ast.parse(inspect.getsource(_paopac))
    top_level = [node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom)]
    modules = [node.module if isinstance(node, ast.ImportFrom) else node.names[0].name for node in top_level]
    assert not [name for name in modules if name.split(".")[0] in {"sklearn", "statsmodels"}]
