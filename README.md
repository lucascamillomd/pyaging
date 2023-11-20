<p align="center">
  <img height="150" src="logo.png" />
</p>

<!--![Logo](https://url-to-your-logo/logo.png)-->

##

[![beta](https://img.shields.io/badge/status-beta-yellow)](https://github.com/rsinghlab/pyaging)
[![upload](https://img.shields.io/pypi/v/pyaging?logo=PyPI)](https://pypi.org/project/pyaging/) 
[![download](https://static.pepy.tech/badge/pyaging)](https://pepy.tech/project/pyaging)
[![star](https://img.shields.io/github/stars/rsinghlab/pyaging?logo=GitHub&color=red)](https://github.com/rsinghlab/pyaging/stargazers)

<!--
[![build](https://github.com/rsinghlab/pyaging/actions/workflows/python-package.yml/badge.svg)](https://github.com/rsinghlab/pyaging/actions/workflows/python-package.yml)
[![documentation](https://readthedocs.org/projects/pyaging/badge/?version=latest)](https://pyaging.readthedocs.io/en/latest/)
[![upload_python_package](https://github.com/rsinghlab/pyaging/actions/workflows/python-publish.yml/badge.svg)](https://github.com/rsinghlab/pyaging/actions/workflows/python-publish.yml)
[![test](https://github.com/rsinghlab/pyaging/actions/workflows/python-plain-run-test.yml/badge.svg)](https://github.com/rsinghlab/pyaging/actions/workflows/python-plain-run-test.yml)
-->

## 🐍 **pyaging**: a Python-based compendium of GPU-optimized biological aging clocks

pyaging is a Python-based package with the latest and greatest aging clocks available to the longevity research community.

<!--
[Installation](https://dynamo-release.readthedocs.io/en/latest/ten_minutes_to_dynamo.html#how-to-install) - [Ten minutes to dynamo](https://dynamo-release.readthedocs.io/en/latest/ten_minutes_to_dynamo.html) - [Tutorials](https://dynamo-release.readthedocs.io/en/latest/notebooks/Differential_geometry.html) - [API](https://dynamo-release.readthedocs.io/en/latest/API.html) - [Citation](https://www.sciencedirect.com/science/article/pii/S0092867421015774?via%3Dihub) - [Theory](https://dynamo-release.readthedocs.io/en/latest/notebooks/Primer.html)
-->

Given the increasing number of aging clocks, it is often cumbersome to compare them side by side. Moreover, for large amounts of data, the time for inference starts to become considerable. For such, we have developed a single package that can take in different molecular layers (DNA methylation, histone ChIP-Seq, ATAC-seq, transcriptomics, etc.) and utilize several aging clocks in a GPU-backed manner for fast inference. Moreover, this allows easy analysis of paired data from more than one molecular layer. 

## 📝 To-Do List

- [ ] Add and run tests
- [ ] Use black for PEP8 formatting
- [ ] Check docstring for each function
- [ ] Add documentation with sphinx
- [ ] Create Read The Docs website
- [ ] Create tutorials
- [ ] Add documentation to each function
- [ ] Add argument to choose dir for file download (as opposed to always "pyaging_data")
- [ ] Add scAge
- [ ] Add murine DNAm clocks
- [ ] Add proteomic clocks
- [ ] Add blood chemistry PhenoAge
- [ ] Add more example data!

## 📰 News
* 19/11/2023: First commit!

## 💬 Discussion 
Please use GitHub issue tracker to report coding-related [issues](https://github.com/rsinghlab/pyaging/issues) of pyaging.


