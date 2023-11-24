<p align="center">
  <img height="150" src="logo.png" />
</p>

##

[![beta](https://img.shields.io/badge/status-beta-yellow)](https://github.com/rsinghlab/pyaging)
[![test](https://github.com/rsinghlab/pyaging/actions/workflows/test.yml/badge.svg)](https://github.com/rsinghlab/pyaging/actions/workflows/test.yml)
[![build](https://github.com/rsinghlab/pyaging/actions/workflows/build.yml/badge.svg)](https://github.com/rsinghlab/pyaging/actions/workflows/build.yml)
[![publish](https://github.com/rsinghlab/pyaging/actions/workflows/publish.yml/badge.svg)](https://github.com/rsinghlab/pyaging/actions/workflows/publish.yml)
[![upload](https://img.shields.io/pypi/v/pyaging?logo=PyPI)](https://pypi.org/project/pyaging/) 
[![download](https://static.pepy.tech/badge/pyaging)](https://pepy.tech/project/pyaging)
[![star](https://img.shields.io/github/stars/rsinghlab/pyaging?logo=GitHub&color=red)](https://github.com/rsinghlab/pyaging/stargazers)

<!--
[![documentation](https://readthedocs.org/projects/pyaging/badge/?version=latest)](https://pyaging.readthedocs.io/en/latest/)
-->

## 🐍 **pyaging**: A GPU-Optimized Python Compendium for Biological Aging Research

`pyaging` is a cutting-edge Python package designed for the longevity research community, offering a comprehensive suite of GPU-optimized biological aging clocks.

<!--
[Installation](https://pyaging.readthedocs.io/en/latest/installation.html) - [Quick Start](https://pyaging.readthedocs.io/en/latest/quickstart.html) - [Tutorials](https://pyaging.readthedocs.io/en/latest/tutorials.html) - [API Reference](https://pyaging.readthedocs.io/en/latest/api.html) - [Citation](https://www.sciencedirect.com/science/article/pii/S0092867421015774?via%3Dihub) - [Theoretical Background](https://pyaging.readthedocs.io/en/latest/theory.html)
-->

With a growing number of aging clocks, comparing and analyzing them can be challenging. `pyaging` simplifies this process, allowing researchers to input various molecular layers (DNA methylation, histone ChIP-Seq, ATAC-seq, transcriptomics, etc.) and quickly analyze them using multiple aging clocks, thanks to its GPU-backed infrastructure. This makes it an ideal tool for large datasets and multi-layered analysis.

## 📝 To-Do List

- [X] Condense `download_data` into a single function in `utils`
- [X] Implement tests for each module 
- [X] Enhance tutorials for better user experience
- [X] Integrate `black` for PEP8 compliant code formatting
- [ ] Improve and expand `readthedocs` documentation
- [X] Review and update docstrings for all functions
- [ ] Move helper functions to new .py files
- [ ] Move preprocessing and postprocessing to models
- [ ] Launch a dedicated Read The Docs website
- [ ] Add option to specify download directory for datasets
- [ ] Include mammalian array example datasets
- [ ] Rename datasets to reflect source publications
- [ ] Add feature to control logging verbosity
- [ ] Integrate scAge and scRNAseq clocks
- [ ] Incorporate murine DNA methylation and proteomic clocks
- [ ] Add PhenoAge based on blood chemistry analysis

## ❓ Can't find an aging clock?

If you have recently developed an aging clock and would like it to be integrated into `pyaging`, please [email us](lucas_camillo@alumni.brown.edu). We aim to incorporate it within two weeks!

## 💬 Community Discussion
For coding-related queries, feedback, and discussions, please visit our [GitHub Issues](https://github.com/rsinghlab/pyaging/issues) page.
