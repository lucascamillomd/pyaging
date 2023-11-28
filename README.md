<p align="center">
  <img height="160" src="docs/_static/logo.png" />
</p>

##

[![test](https://github.com/rsinghlab/pyaging/actions/workflows/test.yml/badge.svg)](https://github.com/rsinghlab/pyaging/actions/workflows/test.yml)
[![build](https://github.com/rsinghlab/pyaging/actions/workflows/build.yml/badge.svg)](https://github.com/rsinghlab/pyaging/actions/workflows/build.yml)
[![publish](https://github.com/rsinghlab/pyaging/actions/workflows/publish.yml/badge.svg)](https://github.com/rsinghlab/pyaging/actions/workflows/publish.yml)
[![release](https://github.com/rsinghlab/pyaging/actions/workflows/release.yml/badge.svg)](https://github.com/rsinghlab/pyaging/actions/workflows/release.yml)
[![documentation](https://readthedocs.org/projects/pyaging/badge/?version=latest)](https://pyaging.readthedocs.io/en/latest/)
[![beta](https://img.shields.io/badge/status-beta-yellow)](https://github.com/rsinghlab/pyaging)
[![upload](https://img.shields.io/pypi/v/pyaging?logo=PyPI)](https://pypi.org/project/pyaging/) 
[![download](https://static.pepy.tech/badge/pyaging)](https://pepy.tech/project/pyaging)
[![star](https://img.shields.io/github/stars/rsinghlab/pyaging?logo=GitHub&color=red)](https://github.com/rsinghlab/pyaging/stargazers)

## 🐍 **pyaging**: a Python-based compendium of GPU-optimized aging clocks

`pyaging` is a cutting-edge Python package designed for the longevity research community, offering a comprehensive suite of GPU-optimized biological aging clocks.

[Installation](https://pyaging.readthedocs.io/en/latest/installation.html) - [Search, cite, and get metadata](https://pyaging.readthedocs.io/en/latest/tutorial_utils.html) - [Bulk DNA methylation](https://pyaging.readthedocs.io/en/latest/tutorial_dnam.html) - [Bulk histone mark ChIP-Seq](https://pyaging.readthedocs.io/en/latest/tutorial_histonemarkchipseq.html) - [Bulk ATAC-Seq](https://pyaging.readthedocs.io/en/latest/tutorial_atacseq.html) - [Bulk RNA-Seq](https://pyaging.readthedocs.io/en/latest/tutorial_rnaseq.html) - [API Reference](https://pyaging.readthedocs.io/en/latest/pyaging.html)

With a growing number of aging clocks, comparing and analyzing them can be challenging. `pyaging` simplifies this process, allowing researchers to input various molecular layers (DNA methylation, histone ChIP-Seq, ATAC-seq, transcriptomics, etc.) and quickly analyze them using multiple aging clocks, thanks to its GPU-backed infrastructure. This makes it an ideal tool for large datasets and multi-layered analysis.

## 📝 To-Do List

- [X] Integrate `black` for PEP8 compliant code formatting
- [X] Improve and expand `readthedocs` documentation
- [X] Add option to specify download directory for datasets
- [X] Rename datasets to reflect source publications
- [X] Include mammalian array example datasets
- [X] Add feature to control logging verbosity
- [ ] Publish to PyPi
- [ ] Add percentage of missing features in anndata after age prediction for each clock
- [ ] Add table in readthedocs with all clock metadata
- [ ] Add and populate "Notes" tab in metadata
- [ ] Integrate scAge and scRNAseq clocks (and datasets)
- [ ] Incorporate murine DNA methylation and proteomic clocks (and datasets)
- [ ] Add PhenoAge based on blood chemistry analysis (and datasets)

## ❓ Can't find an aging clock?

If you have recently developed an aging clock and would like it to be integrated into `pyaging`, please [email us](lucas_camillo@alumni.brown.edu). We aim to incorporate it within two weeks!

## 💬 Community Discussion
For coding-related queries, feedback, and discussions, please visit our [GitHub Issues](https://github.com/rsinghlab/pyaging/issues) page.
