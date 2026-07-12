from importlib.metadata import version

import pyaging


def test_package_version_is_0_3_0_everywhere():
    assert pyaging.__version__ == "0.3.0"
    assert version("pyaging") == "0.3.0"
