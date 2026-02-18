Installation
============

*Please note that pyaging is supported by python versions 3.9, 3.10, 3.11, and 3.12.**

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
-------------------------------------------

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