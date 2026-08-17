from importlib.metadata import version

import pyaging


def test_package_version_matches_installed_metadata():
    assert version("pyaging") == pyaging.__version__
