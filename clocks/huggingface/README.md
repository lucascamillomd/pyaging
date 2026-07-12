---
license: other
library_name: pyaging
tags:
  - biology
  - aging
  - biological-age
  - pytorch
---

# pyaging data

This public repository contains the model weights and data files used by
[`lucascamillomd/pyaging`](https://github.com/lucascamillomd/pyaging).

## Contents

- Root-level `*.pt` files are the current pyaging clock models.
- `all_clock_metadata.pt` is the live aggregate clock catalog.
- Root-level example files support the pyaging tutorials.
- `supporting_files/` contains dependencies used to construct or document clocks.

Files used by the Python package are intentionally stored at the repository root so
`hf_hub_download(..., local_dir="pyaging_data")` preserves existing flat local paths.
The `main` branch is the live data release and may change independently of the Python
package version.

## Licensing and provenance

This is a mixed-provenance research collection, so the repository license is `other`.
The pyaging BSD license does not grant additional rights to third-party clock weights or
source datasets. Consult each clock's embedded metadata, cited publication, and notes
before use. Some clocks are marked research-only or have separate commercial terms.

## Security

Clock files are trusted Python/PyTorch objects loaded by pyaging with
`torch.load(..., weights_only=False)`. Loading a malicious pickle can execute code. Only
load these files from this official repository and review unexpected repository changes.

## Publishing policy

The repository is maintained solely by Lucas Paulo de Lima Camillo (`lucascamillomd`).
Weights are uploaded before aggregate metadata so the catalog never advertises a missing
clock file. Public users need no Hugging Face token to download files.
