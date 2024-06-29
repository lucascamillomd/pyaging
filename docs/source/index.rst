.. pyaging documentation master file, created by
   sphinx-quickstart on Sun Nov 19 17:35:20 2023.
   This file is the entry point to the pyaging package documentation.

.. raw:: html

   <center>

.. image:: https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat
   :target: https://pyaging.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/pyaging.svg
   :target: https://pypi.python.org/pypi/pyaging
   :alt: PyPI version

.. image:: https://img.shields.io/github/license/rsinghlab/pyaging.svg
   :target: https://github.com/rsinghlab/pyaging/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/badge/DOI-10.1093%2Fbioinformatics%2Fbtae200-blue.svg
   :target: https://doi.org/10.1093/bioinformatics/btae200
   :alt: DOI

.. raw:: html

   </center>

Welcome to the Documentation for pyaging
========================================

`pyaging` is a Python package designed for biological aging clocks analysis, leveraging a GPU-optimized PyTorch backend. `pyaging` aims to be a comprehensive toolkit for researchers and scientists in the field of aging.

.. image:: ../_static/pyaging_graphical_abstract.pdf
   :align: center
   :alt: Pyaging Graphical Abstract

.. raw:: html

   <br><br>

Explore Our GitHub Repository
-----------------------------
Discover more about `pyaging` and contribute to our growing community on `GitHub <https://github.com/rsinghlab/pyaging>`_.

Contents
--------

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   clock_glossary

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/tutorial_utils
   tutorials/tutorial_dnam_illumina_human_array
   tutorials/tutorial_dnam_illumina_mammalian_array
   tutorials/tutorial_dnam_rrbs
   tutorials/tutorial_histonemarkchipseq
   tutorials/tutorial_atacseq
   tutorials/tutorial_rnaseq
   tutorials/tutorial_bloodchemistry

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   pyaging

.. toctree::
   :maxdepth: 1
   :caption: Clock implementation

   clock_implementation

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   faq
   references

Indices and Tables
------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`