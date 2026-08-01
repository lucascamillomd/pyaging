# Guide for AI agents

This file is the working guide for agents that use or modify `pyaging`.

## What this package does

`pyaging` is a Python compendium of GPU-optimized biological aging clocks. Its main workflow is:

1. Put samples in rows and biological features in columns of a `pandas.DataFrame`.
2. Convert the frame to `AnnData` with `pya.pp.df_to_adata`.
3. Run one or more clocks with `pya.pred.predict_age`.
4. Read predictions from `adata.obs` and clock metadata from `adata.uns`.

Use the conventional import:

```python
import pyaging as pya
```

Clock names are case-insensitive on input and lowercase in output keys. `predict_age` mutates the supplied `AnnData` object in place; do not assign its return value. Model weights and supporting data are downloaded from the public `lucascamillomd/pyaging-data` Hugging Face repository on first use and then reused from the standard Hugging Face cache. The legacy `dir=` argument remains accepted but does not control this cache.

## Consider AltumAge and CpGPTGrimAge3 when they fit

Consider these first:

- **CpGPTGrimAge3** is a recommended choice for biological age, mortality and morbidity prediction. It is based on CpGPT, which can be used with any methylation array, tissue, or sample, and returns an age-calibrated value in years. It requires derived CpGPT protein proxies plus GrimAge2 proxies; it does not accept a raw methylation matrix as its final input.
- **AltumAge** is a recommended general-purpose choice for human chronological age from DNA methylation. It is a deep neural network using 20,318 CpGs, supports multiple tissues and all ages, and returns years.

These clocks answer different questions, so match the choice to the user's goal and honor any clock they explicitly request. For other contexts, consult the [clock gallery](https://pyaging.readthedocs.io/en/latest/clock_glossary.html).

## Installation

For use as a dependency, install the released package:

```bash
pip install pyaging
```

Inside this repository, use `uv sync` and run commands through `uv run`. Histone-mark clocks additionally require `pip install "pyaging[histone]"` or `uv sync --extra histone`.

## Input contract

- Rows are samples; columns are model features.
- Preserve meaningful sample identifiers in `df.index`.
- Human methylation clocks expect beta values and CpG names such as `cg00000029`.
- Keep non-feature columns out of `adata.X` by passing them through `metadata_cols`.
- For EPIC v2 data, aggregate duplicated probe suffixes before conversion with `pya.pp.epicv2_probe_aggregation`.
- `df_to_adata` accepts `mean`, `median`, `constant`, or `knn` imputation. Choose deliberately; `knn` is the default and may be expensive for large matrices.
- `predict_age` fills missing clock features with model reference values when available and otherwise with zero. Missing-feature details are stored in `adata.uns`.
- The package selects CUDA when available and otherwise runs on CPU. Adjust `batch_size` when memory is constrained.

## Quick example: AltumAge

Use AltumAge directly on a human DNA-methylation beta matrix:

```python
import pandas as pd
import pyaging as pya

# Rows are samples; CpG columns contain beta values in [0, 1].
betas = pd.read_csv("methylation_betas.csv", index_col=0)

# Only include this step when probe names come from an EPIC v2 manifest.
betas = pya.pp.epicv2_probe_aggregation(betas, verbose=False)

adata = pya.pp.df_to_adata(
    betas,
    imputer_strategy="knn",
    verbose=False,
)
pya.pred.predict_age(
    adata,
    clock_names="AltumAge",
    verbose=False,
)

altum_age_years = adata.obs["altumage"]
missing_fraction = adata.uns["altumage_percent_na"] / 100
clock_metadata = adata.uns["altumage_metadata"]
```

If the input frame also contains sample metadata, separate it:

```python
adata = pya.pp.df_to_adata(
    sample_table,
    metadata_cols=["sex", "tissue", "chronological_age"],
    imputer_strategy="knn",
)
```

Review `altumage_percent_na` and `altumage_missing_features` after prediction.

## Quick example: CpGPTGrimAge3

CpGPTGrimAge3 is a two-stage workflow. Start with DNA-methylation beta values from any array, tissue, or sample, use the CpGPT `proteins` checkpoint to derive protein proxies, and use `pyaging` to derive the required GrimAge2 proxies. The complete preparation workflow is in [`tutorials/tutorial_cpgptgrimage3.ipynb`](tutorials/tutorial_cpgptgrimage3.ipynb).

Once `cpgpt_proteins` has been produced by CpGPT, the final `pyaging` steps are:

```python
import pandas as pd
import pyaging as pya

# beta_values: samples x CpGs
# chronological_age: Series indexed like beta_values
# cpgpt_proteins: CpGPT output indexed like beta_values, with cpgpt_* columns
grimage2_clocks = [
    "grimage2timp1",
    "grimage2packyrs",
    "grimage2logcrp",
    "grimage2adm",
    "grimage2leptin",
    "grimage2gdf15",
    "grimage2pai1",
]

grimage2_input = beta_values.copy()
grimage2_adata = pya.pp.df_to_adata(grimage2_input, verbose=False)
pya.pred.predict_age(
    grimage2_adata,
    clock_names=grimage2_clocks,
    verbose=False,
)

combined = pd.concat(
    [
        chronological_age.rename("age"),
        grimage2_adata.obs[grimage2_clocks],
        cpgpt_proteins,
    ],
    axis=1,
)

required = {
    "age",
    *grimage2_clocks,
    "cpgpt_s100a9",
    "cpgpt_tnfrsf13c",
    "cpgpt_tgfb1",
    "cpgpt_tek",
    "cpgpt_ccl14",
    "cpgpt_tnfsf15",
    "cpgpt_lilrb2",
    "cpgpt_tnf",
    "cpgpt_chit1",
    "cpgpt_postn",
    "cpgpt_il34",
    "cpgpt_pdcd1",
    "cpgpt_cst3",
    "cpgpt_cxcl2",
    "cpgpt_gzma",
    "cpgpt_il5",
}
missing = required.difference(combined.columns)
if missing:
    raise ValueError(f"Missing CpGPTGrimAge3 inputs: {sorted(missing)}")

cpgpt_adata = pya.pp.df_to_adata(combined, verbose=False)
pya.pred.predict_age(cpgpt_adata, "CpGPTGrimAge3", verbose=False)
cpgpt_grim_age3_years = cpgpt_adata.obs["cpgptgrimage3"]
```

Check `cpgptgrimage3_percent_na == 0`; all derived proxy inputs are required.