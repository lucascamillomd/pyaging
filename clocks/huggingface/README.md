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

Files used by the Python package are intentionally stored at the repository root and
downloaded through the standard Hugging Face cache.
The `main` branch is the live data release and may change independently of the Python
package version.

## Harmonized clock metadata

The clock catalogue uses controlled, multi-valued metadata so clocks can be
filtered consistently:

- `tissue` records the biological material used to develop or train the model.
- `platform` records the measurement platform used for model development.
- `predicts` describes how to interpret the value returned by the packaged
  model.
- `training_target` records the outcome used to fit or derive the model.
- `unit` records the physical or statistical unit of the returned,
  postprocessed value.

Each of these fields is an array of controlled terms, even when a clock has only
one value. Precise wording from the paper, supplement, implementation, or author
communication is retained in the notebooks' same-line metadata comments and in
the field-level evidence ledger. The canonical
[`clock_metadata.json`](https://github.com/lucascamillomd/pyaging/blob/main/clocks/metadata/clock_metadata.json)
registry and
[`evidence_ledger.jsonl`](https://github.com/lucascamillomd/pyaging/blob/main/clocks/metadata/evidence_ledger.jsonl)
are maintained in the pyaging repository.

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
