<p align="center">
  <img height="160" src="docs/_static/logo.png" />
</p>

##

[![CI](https://github.com/lucascamillomd/pyaging/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/lucascamillomd/pyaging/actions/workflows/ci.yml)
[![Release](https://github.com/lucascamillomd/pyaging/actions/workflows/release.yaml/badge.svg)](https://github.com/lucascamillomd/pyaging/actions/workflows/release.yaml)
[![Documentation](https://readthedocs.org/projects/pyaging/badge/?version=latest)](https://pyaging.readthedocs.io/en/latest/)
[![DOI](https://img.shields.io/badge/DOI-10.1093%2Fbioinformatics%2Fbtae200-blue.svg)](https://doi.org/10.1093/bioinformatics/btae200)
[![PyPI](https://img.shields.io/pypi/v/pyaging?logo=pypi)](https://pypi.org/project/pyaging/)
[![Downloads](https://static.pepy.tech/badge/pyaging)](https://pepy.tech/project/pyaging)
[![Stars](https://img.shields.io/github/stars/lucascamillomd/pyaging.svg?label=stars&style=flat&logo=github&color=red)](https://github.com/lucascamillomd/pyaging/stargazers)

## 🐍 **pyaging**: a Python-based compendium of GPU-optimized aging clocks

`pyaging` is a cutting-edge Python package designed for the longevity research community, offering a comprehensive suite of GPU-optimized biological aging clocks.

[Installation](https://pyaging.readthedocs.io/en/latest/installation.html) - [Clock gallery](https://pyaging.readthedocs.io/en/latest/clock_glossary.html) - [Search, cite, get metadata and clock parameters](https://pyaging.readthedocs.io/en/latest/tutorials/tutorial_utils.html) - [Illumina Human Methylation Arrays](https://pyaging.readthedocs.io/en/latest/tutorials/tutorial_dnam_illumina_human_array.html) - [Illumina Mammalian Methylation Arrays](https://pyaging.readthedocs.io/en/latest/tutorials/tutorial_dnam_illumina_mammalian_array.html) - [RRBS DNA methylation](https://pyaging.readthedocs.io/en/latest/tutorials/tutorial_dnam_rrbs.html) - [Bulk histone mark ChIP-Seq](https://pyaging.readthedocs.io/en/latest/tutorials/tutorial_histonemarkchipseq.html) - [Bulk ATAC-Seq](https://pyaging.readthedocs.io/en/latest/tutorials/tutorial_atacseq.html) - [Bulk RNA-Seq](https://pyaging.readthedocs.io/en/latest/tutorials/tutorial_rnaseq.html) - [Blood chemistry](https://pyaging.readthedocs.io/en/latest/tutorials/tutorial_bloodchemistry.html) - [CpGPTGrimAge3](https://pyaging.readthedocs.io/en/latest/tutorials/tutorial_cpgptgrimage3.html) - [API Reference](https://pyaging.readthedocs.io/en/latest/pyaging.html)

With a growing number of aging clocks and biomarkers of aging, comparing and analyzing them can be challenging. `pyaging` simplifies this process, allowing researchers to input various molecular layers (DNA methylation, histone ChIP-Seq, ATAC-seq, transcriptomics, etc.) and quickly analyze them using multiple aging clocks, thanks to its GPU-backed infrastructure. This makes it an ideal tool for large datasets and multi-layered analysis.

## 📦 Installation

`pyaging` requires Python 3.11 or newer and is available on PyPI:

```bash
pip install pyaging
```

To use the histone mark clocks, install the optional `pyBigWig` dependency as well (not supported on Windows):

```bash
pip install pyaging[histone]
```

## 🚀 Quickstart

```python
import pandas as pd
import pyaging as pya

pya.data.download_example_data("GSE139307")
df = pd.read_pickle("pyaging_data/GSE139307.pkl")

adata = pya.pp.df_to_adata(df)
pya.pred.predict_age(adata, ["Horvath2013", "AltumAge", "DunedinPACE"])
adata.obs.head()
```

Clock weights are downloaded on demand from per-clock repositories under the [`pyaging` Hugging Face organization](https://huggingface.co/pyaging) (example data comes from [`lucascamillomd/pyaging-data`](https://huggingface.co/lucascamillomd/pyaging-data)). Set the `PYAGING_DATA_REVISION` environment variable to a release tag (e.g. `v0.3.1`) to pin downloads to a specific data revision for reproducibility; it defaults to `main`, the live data release.

## ❓ Can't find an aging clock?

If you have recently developed an aging clock and would like it to be integrated into `pyaging`, please [email me](lucas_camillo@alumni.brown.edu). I aim to incorporate it within one to two weeks! I'm also happy to adapt to any licensing terms for commercial entities.

## 💬 Community Discussion
For coding-related queries, feedback, and discussions, please visit our [GitHub Issues](https://github.com/lucascamillomd/pyaging/issues) page.

## 📝 Changelog
Notable changes, including breaking ones, are recorded in [`CHANGELOG.md`](https://github.com/lucascamillomd/pyaging/blob/main/CHANGELOG.md). Per-version release artifacts and the commit log are on the [GitHub Releases](https://github.com/lucascamillomd/pyaging/releases) page.

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
