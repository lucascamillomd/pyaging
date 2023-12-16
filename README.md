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
[![bioRxiv](https://img.shields.io/badge/bioRxiv-DOI-purple.svg)](https://doi.org/10.1101/2023.11.28.569069)
[![upload](https://img.shields.io/pypi/v/pyaging?logo=PyPI)](https://pypi.org/project/pyaging/) 
[![download](https://static.pepy.tech/badge/pyaging)](https://pepy.tech/project/pyaging)
[![star](https://img.shields.io/github/stars/rsinghlab/pyaging?logo=GitHub&color=red)](https://github.com/rsinghlab/pyaging/stargazers)

## 🐍 **pyaging**: a Python-based compendium of GPU-optimized aging clocks

`pyaging` is a cutting-edge Python package designed for the longevity research community, offering a comprehensive suite of GPU-optimized biological aging clocks.

[Installation](https://pyaging.readthedocs.io/en/latest/installation.html) - [Clock gallery](https://pyaging.readthedocs.io/en/latest/clock_glossary.html) - [Search, cite, and get metadata](https://pyaging.readthedocs.io/en/latest/tutorial_utils.html) - [Bulk DNA methylation](https://pyaging.readthedocs.io/en/latest/tutorial_dnam.html) - [Bulk histone mark ChIP-Seq](https://pyaging.readthedocs.io/en/latest/tutorial_histonemarkchipseq.html) - [Bulk ATAC-Seq](https://pyaging.readthedocs.io/en/latest/tutorial_atacseq.html) - [Bulk RNA-Seq](https://pyaging.readthedocs.io/en/latest/tutorial_rnaseq.html) - [Blood chemistry](https://pyaging.readthedocs.io/en/latest/tutorial_bloodchemistry.html) - [API Reference](https://pyaging.readthedocs.io/en/latest/pyaging.html)

With a growing number of aging clocks and biomarkers of aging, comparing and analyzing them can be challenging. `pyaging` simplifies this process, allowing researchers to input various molecular layers (DNA methylation, histone ChIP-Seq, ATAC-seq, transcriptomics, etc.) and quickly analyze them using multiple aging clocks, thanks to its GPU-backed infrastructure. This makes it an ideal tool for large datasets and multi-layered analysis.

## ❓ Can't find an aging clock?

If you have recently developed an aging clock and would like it to be integrated into `pyaging`, please [email us](lucas_camillo@alumni.brown.edu). We aim to incorporate it within two weeks! We are also happy to adapt to any licensing terms for commercial entities.

## 💬 Community Discussion
For coding-related queries, feedback, and discussions, please visit our [GitHub Issues](https://github.com/rsinghlab/pyaging/issues) page.

## 📖 Citation

To cite `pyaging`, please use the following:

```
@article {de_Lima_Camillo_pyaging,
	author = {Lucas Paulo de Lima Camillo},
	title = {pyaging: a Python-based compendium of GPU-optimized aging clocks},
	elocation-id = {2023.11.28.569069},
	year = {2023},
	doi = {10.1101/2023.11.28.569069},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2023/11/29/2023.11.28.569069},
	journal = {bioRxiv}
}
```

## 📝 To-Do List

- [X] Incorporate more murine DNA methylation clocks
- [X] Add torch data loader for age prediction of large datasets
- [ ] Add other blood chemistry biological age clocks (KD age)
- [ ] Incorporate proteomic clocks (and datasets)
- [ ] Integrate scAge clocks (this is proving to be difficult)
- [ ] Integrate scRNAseq clocks (and datasets)

