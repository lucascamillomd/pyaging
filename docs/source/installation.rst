Installation
============

*Please note that pyaging requires python version 3.11 or newer.*

pyaging now has been released to PyPi and can easily be installed via:

.. code-block:: bash

    pip install pyaging

Alternatively, it can be installed by cloning our GitHub repository and using pip:

.. code-block:: bash

    git clone https://github.com/lucascamillomd/pyaging.git
    pip install pyaging/ --user

Or by simply going to the cloned repository if you have uv installed:

.. code-block:: bash

    git clone https://github.com/lucascamillomd/pyaging.git
    cd pyaging/
    uv sync

Lastly, it can be installed from source:

.. code-block:: bash

    pip install git+https://github.com/lucascamillomd/pyaging

.. note::

    The histone mark clocks can only be used when the optional dependency pyBigWig is also installed. Currently, pyBigWig is not supported on Windows.

Installation with histone mark clock support
--------------------------------------------

To use histone mark clocks, you need to install pyaging with the optional pyBigWig dependency:

.. code-block:: bash

    pip install pyaging[histone]

When installing from a cloned repository with uv and optional dependencies:

.. code-block:: bash

    git clone https://github.com/lucascamillomd/pyaging.git
    cd pyaging/
    uv sync --extra histone

Or from source:

.. code-block:: bash

    pip install git+https://github.com/lucascamillomd/pyaging#egg=pyaging[histone]

Pinning the clock weights
-------------------------

Clock weights are not shipped inside the package. They are downloaded on demand from
per-clock repositories under the `pyaging Hugging Face organization
<https://huggingface.co/pyaging>`_, and they resolve from the ``main`` branch at call
time. The weights can change independently of the installed package version.
Package-only releases, including 0.5.3, do not upload weights or create new data tags.

Set ``PYAGING_DATA_REVISION`` to a release tag to pin every download to one revision:

.. code-block:: bash

    PYAGING_DATA_REVISION=v0.5.0 python my_analysis.py

Equivalently, from inside Python, before the first ``predict_age`` call:

.. code-block:: python

    import os

    os.environ["PYAGING_DATA_REVISION"] = "v0.5.0"

The variable is read at call time, so it also accepts any commit SHA. Pin it whenever an
analysis has to stay reproducible. Choose an existing data tag compatible with the package,
rather than assuming every package version has a matching data tag. A newer weight file
paired with older preprocessing code can change a prediction without raising an error.
When a tag is missing from a per-clock repository, pyaging tries the same tag in
the shared data repository. Tag coverage varies by clock: tAge currently has a
``v0.5.1`` tag, and is absent from the shared ``v0.5.2`` snapshot. Check that the
chosen tag covers your requested clocks and their supporting assets. pyaging
never substitutes ``main`` when an explicitly requested revision is unavailable.
