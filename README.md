<p align="center">
  <img height="160" src="docs/_static/logo.png" />
</p>

##

[![test](https://github.com/rsinghlab/pyaging/actions/workflows/test.yml/badge.svg)](https://github.com/rsinghlab/pyaging/actions/workflows/test.yml)
[![build](https://github.com/rsinghlab/pyaging/actions/workflows/build.yml/badge.svg)](https://github.com/rsinghlab/pyaging/actions/workflows/build.yml)
[![publish](https://github.com/rsinghlab/pyaging/actions/workflows/publish.yml/badge.svg)](https://github.com/rsinghlab/pyaging/actions/workflows/publish.yml)
[![release](https://github.com/rsinghlab/pyaging/actions/workflows/release.yml/badge.svg)](https://github.com/rsinghlab/pyaging/actions/workflows/release.yml)
[![documentation](https://readthedocs.org/projects/pyaging/badge/?version=latest)](https://pyaging.readthedocs.io/en/latest/)
[![DOI](https://img.shields.io/badge/DOI-10.1093%2Fbioinformatics%2Fbtae200-blue.svg)](https://doi.org/10.1093/bioinformatics/btae200)
[![upload](https://img.shields.io/pypi/v/pyaging?logo=PyPI)](https://pypi.org/project/pyaging/) 
[![download](https://static.pepy.tech/badge/pyaging)](https://pepy.tech/project/pyaging)
[![star](https://img.shields.io/github/stars/rsinghlab/pyaging?logo=GitHub&color=red)](https://github.com/rsinghlab/pyaging/stargazers)

## 🐍 **pyaging**: a Python-based compendium of GPU-optimized aging clocks

`pyaging` is a cutting-edge Python package designed for the longevity research community, offering a comprehensive suite of GPU-optimized biological aging clocks.

[Installation](https://pyaging.readthedocs.io/en/latest/installation.html) - [Clock gallery](https://pyaging.readthedocs.io/en/latest/clock_glossary.html) - [Search, cite, get metadata and clock parameters](https://pyaging.readthedocs.io/en/latest/tutorial_utils.html) - [Illumina Human Methylation Arrays](https://pyaging.readthedocs.io/en/latest/tutorial_dnam_illumina_human_array.html) - [Illumina Mammalian Methylation Arrays](https://pyaging.readthedocs.io/en/latest/tutorial_dnam_illumina_mammalian_array.html) - [RRBS DNA methylation](https://pyaging.readthedocs.io/en/latest/tutorial_dnam_rrbs.html) - [Bulk histone mark ChIP-Seq](https://pyaging.readthedocs.io/en/latest/tutorial_histonemarkchipseq.html) - [Bulk ATAC-Seq](https://pyaging.readthedocs.io/en/latest/tutorial_atacseq.html) - [Bulk RNA-Seq](https://pyaging.readthedocs.io/en/latest/tutorial_rnaseq.html) - [Blood chemistry](https://pyaging.readthedocs.io/en/latest/tutorial_bloodchemistry.html) - [API Reference](https://pyaging.readthedocs.io/en/latest/pyaging.html)

With a growing number of aging clocks and biomarkers of aging, comparing and analyzing them can be challenging. `pyaging` simplifies this process, allowing researchers to input various molecular layers (DNA methylation, histone ChIP-Seq, ATAC-seq, transcriptomics, etc.) and quickly analyze them using multiple aging clocks, thanks to its GPU-backed infrastructure. This makes it an ideal tool for large datasets and multi-layered analysis.

## ❓ Can't find an aging clock?

If you have recently developed an aging clock and would like it to be integrated into `pyaging`, please [email us](lucas_camillo@alumni.brown.edu). We aim to incorporate it within two weeks! We are also happy to adapt to any licensing terms for commercial entities.

## 💬 Community Discussion
For coding-related queries, feedback, and discussions, please visit our [GitHub Issues](https://github.com/rsinghlab/pyaging/issues) page.

## 📖 Citation

To cite `pyaging`, please use the following:

```
@article{de_Lima_Camillo_pyaging,
    author = {de Lima Camillo, Lucas Paulo},
    title = "{pyaging: a Python-based compendium of GPU-optimized aging clocks}",
    journal = {Bioinformatics},
    pages = {btae200},
    year = {2024},
    month = {04},
    issn = {1367-4811},
    doi = {10.1093/bioinformatics/btae200},
    url = {https://doi.org/10.1093/bioinformatics/btae200},
    eprint = {https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btae200/57218155/btae200.pdf},
}
```

## 📝 To-Do List

- [ ] Add SymphonyAge

