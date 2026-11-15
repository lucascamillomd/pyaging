# PAOPAC original-interface parity fixtures

`native_oracle.npz` contains synthetic inputs, original standardized matrices,
native LightGBM raw predictions, and final predictions produced by the authors'
unchanged Windows `pre.OrganAgePredictor.predict` extension. It contains no
participant data. `generate_oracle.py` documents the reproducible generator;
`oracle_provenance.json` records library versions and checksums.

Primary sources:

- [PAOPAC author repository](https://github.com/JackieHanLab/PAOPAC/tree/a166bc763ffd1395a96c6557db889706fcc9ef85)
- [Public v1.0.0 model.bin](https://github.com/JackieHanLab/PAOPAC/releases/download/v1.0.0/model.bin)
- [Original preprint](https://doi.org/10.64898/2026.04.24.720503)

The oracle ran in Debian bookworm, Wine 8.0, and Windows CPython 3.9.13, under an
isolated Docker container with networking disabled. NumPy 1.26.4, pandas 2.2.3,
scikit-learn 1.6.1, statsmodels 0.14.4, and LightGBM 4.6.0 supplied the original
extension's Python dependencies. Only `pickle.loads` was replaced: it returns
an estimator backed by the original Conventional booster text and feature names.
Every transform, output correction and row association runs inside the author's
compiled predictor. The fitted model uses all 800 trees (no best-iteration cut).

`author_conventional.txt.gz` is the fitted model text extracted from the pinned
author model, compressed without modification. SHA-256 of the uncompressed text:
`10eb9c389ff5fe8fc9adb9c0c27430fcc43dcf628c43ecf42b70e4f8dbfb31d5`.
The original pickle includes identical numerical parameters in `handle` and
`_handle`; only their `version=v3` and `version=v4` headers differ. The retained
text is `_handle` (v4). This fixture lets offline tests verify all 800 trees
against an independent runtime. The author's MIT license is preserved in
`LICENSE`.

The generator takes a source directory containing the pinned extension,
`model.bin`, and extracted `Conventional.txt`, followed by an output directory.
The builder verifies the public assets, reproduces the author's local
Fernet/zlib loading sequence, uses a restricted inert-record pickle reader, and
retains only numerical trees for distribution. The packaging key is available
inside the public MIT loader; no credentials or private assets are used.

Rebuild from the original public assets with
`uv run --with cryptography python clocks/build_paopac.py`; `--source-dir` accepts
a cache containing `model.bin`, `pre.cp39-win_amd64.pyd`, and `LICENSE`. LightGBM,
Wine, and the Windows dependencies are required only to regenerate the oracle,
not to build or use the converted Torch clock.

Cases cover complete data, optional protein-matrix TDI, absent/all-NaN/partly
missing proteins, exact duplicate columns (including NaNs), a singleton,
constant chronological age, and missing age values. Exact duplicate NPX columns
are averaged before exponentiation, skipping NaNs. After that step, names are
uppercased; newly colliding names are rejected by the original interface. Its
singleton correction returns the supplied chronological age, and NaN ages
produce NaN final outputs. These are original cohort-dependent behaviors, not
evidence that a single sample has an independently estimated biological age.
