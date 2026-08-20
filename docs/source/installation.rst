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
time. That means the weights move forward when a new pyaging release is published, even
for an environment whose installed pyaging did not change.

Set ``PYAGING_DATA_REVISION`` to a release tag to pin every download to one revision:

.. code-block:: bash

    PYAGING_DATA_REVISION=v0.5.0 python my_analysis.py

Equivalently, from inside Python, before the first ``predict_age`` call:

.. code-block:: python

    import os

    os.environ["PYAGING_DATA_REVISION"] = "v0.5.0"

The variable is read at call time, so it also accepts any commit SHA. Pin it whenever an
analysis has to stay reproducible, and pin it to the tag matching your installed version
whenever you are deliberately staying on an older pyaging: a clock's weights and the code
that preprocesses them are versioned together, and a newer weight file paired with older
code can change a prediction without raising an error.
