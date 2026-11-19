"""scikit-learn is only needed for mean/median/constant imputation, so importing pyaging must not load it."""

import subprocess
import sys


def test_import_pyaging_does_not_import_sklearn():
    code = "import sys, pyaging; print(sorted(m for m in sys.modules if m.split('.')[0] == 'sklearn'))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"
