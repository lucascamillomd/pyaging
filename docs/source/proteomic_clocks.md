# Proteomic clocks

Proteomic clocks require the assay, protein identifiers and normalization used
by each model. Matching gene names alone is not enough. Olink NPX is a log2
relative-abundance measure, not a concentration: negative values can be valid,
and protein values must not be exponentiated, min–max scaled or z-scored unless
the selected clock's preparation procedure calls for it. pyaging does not
convert raw assay counts to NPX or harmonize batches and platforms automatically.

The catalogue's **proteomics** filter currently includes PAC and 46 OrganAge
models. `pya.utils.get_feature_ranges(clock_name)` reports the model's input
units; an unbounded NPX range is not a claim that measurements from different
panels, batches or sample materials are interchangeable.

## PAC

`pac` is Kuo et al.'s mortality-trained Proteomic Aging Clock. It requires the
128 protein columns released by the authors, plus `age` in years. It returns
mortality-equivalent age in years, using the original Gompertz parameters.

The model was developed on UK Biobank **plasma**, measured with **Olink Explore
3072**. Supply assay-normalized NPX values, using the authors' lowercase names
(including `fut3_fut5` and `ntprobnp`). No additional centering or cohort scaling
is performed at inference. All 129 columns must be present; absent proteins or
age are not silently replaced by zeros. Explicit NaNs propagate. The original
study used k-nearest-neighbor imputation during data preparation; any imputation
in your analysis is a separate, deliberate choice.

```python
import anndata
import pyaging as pya

# npx: samples x proteins, already normalized for the intended Olink assay.
# age: numeric Series in years with the same sample identifiers.
inputs = npx.copy()
inputs.columns = inputs.columns.str.lower()
inputs["age"] = age.reindex(inputs.index)
if inputs.columns.duplicated().any():
    raise ValueError("Resolve duplicate protein identifiers before scoring")
adata = anndata.AnnData(inputs)  # preserves any NaNs without implicit imputation
pya.pred.predict_age(adata, "pac")
pac_years = adata.obs["pac"]
```

Original implementation: [Kuo lab PAC](https://github.com/kuo-lab-uchc/PAC).

## OrganAge

The public names are `organagechronological{organ}` and
`organagemortality{organ}`, for example `organagechronologicalbrain`,
`organagemortalitybrain`, `organagechronologicalconventional`, and
`organagemortalityconventional`. Each family has 23 outputs. The `conventional`
models use proteins across organs; organ-specific models predict from
circulating proteins associated with that organ, not a biopsy of the organ.

The retained models use the original authors' recommended **first fold of the
full Olink Explore 3072 models**, trained on UK Biobank plasma. Supply original,
case-sensitive protein symbols, including `NTproBNP` and `HLA-DRA`, with Olink
NPX values. pyaging applies the original coefficient sum and chronological
intercept without extra centering or scaling.

- Chronological models return predicted age in **years**.
- Mortality models return **relative natural-log mortality hazard**, not years.
- Missing columns have zero contribution, matching the author's example's
  omission of absent proteins. This is not a reference-mean imputation; missing
  influential proteins can substantially change the result. Check
  `adata.uns["<clock_name>_missing_features"]` and `percent_na`.
- NaNs in supplied columns propagate. `df_to_adata` may impute them beforehand;
  use that preprocessing deliberately or construct `AnnData` directly.
- Female and Male variants use protein features and do not require a sex input.

For non-Olink assays, the [authors' guidance](https://github.com/ludgergoeminne/organAging#3-other-data-types)
recommends making protein distributions approximately symmetric, subtracting
protein-wise means, dividing by protein-wise SDs, and multiplying by the
**full-model UKB SDs in Table S3**. The optional `Example_Script.R` SD-matching
block differs by omitting mean subtraction. Neither preparation is silently
applied by pyaging. Record the procedure and reference cohort used; cohort
rescaling can change a sample's estimate when other samples change. Serum,
plasma, other Olink panels, SomaScan, mass spectrometry and concentration assays
must not be assumed interchangeable.

Starting in 0.5.5, the 44 reduced Olink Explore 1536 models are removed from the
current catalogue. The 46 full-model names drop `olink3000`, with unchanged
coefficients. For example,
`organagechronologicalolink3000brain` becomes `organagechronologicalbrain`.
Archived 0.5.4 weights and release tags remain available for reproducibility.
The shorter names still require the platform-specific preparation above.

## The six-clock Insilico study

[Zhavoronkov et al., Nature Biotechnology (2026)](https://doi.org/10.1038/s41587-026-03286-y)
compared these six endpoints on serum measured with Olink Explore 3072:

| Paper endpoint | pyaging support |
|---|---|
| OrganAge chronological | `organagechronologicalconventional` |
| OrganAge mortality | `organagemortalityconventional`, on the original log-hazard scale |
| PAC | `pac` |
| ProtAge | Not included: trained models and reference preprocessing assets require author access |
| ipfP3GPT | Not included: authors now restrict weights and execution to UK Biobank RAP |
| PAOPAC | Not included: public inference release is an opaque CPython 3.9 Windows binary; portable model and preprocessing source are unavailable |

This is support for the original models, not an automatic reproduction of the
trial analysis. The paper used conventional OrganAge models, standardized each
protein and matched its SD to the original reference, and converted mortality
outputs to years. Its assay was Explore 3072; the original OrganAge authors
recommend the full models for this assay. The trial's public demo also shows a
reduced-model example, so the panel measurement alone does not prove the exact
coefficient file used for every published trial result.

The comparison paper obtained ProtAge weights through direct collaboration and
describes fitted reference min–max scaling followed by median centering. A
feature-importance table cannot substitute for those trained models. PAOPAC's
public binary does not expose enough normalization and feature information to
validate a portable implementation. These three clocks are deliberately absent
from the executable catalogue rather than represented by incomplete predictors.
