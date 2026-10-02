# Proteomic clocks

Proteomic clocks require the assay, protein identifiers and normalization used
by each model. Matching gene names alone is not enough. Olink NPX is a log2
relative-abundance measure, not a concentration: negative values can be valid,
and protein values must not be exponentiated, min–max scaled or z-scored unless
the selected clock's preparation procedure calls for it. pyaging does not
convert raw assay counts to NPX or harmonize batches and platforms automatically.

The catalogue's **proteomics** filter currently includes HPS, PAC, PAOPAC and 46 OrganAge
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

## HPS

`hps` is Kuo et al.'s **Healthspan Proteomic Score**, published in
[PNAS in 2025](https://doi.org/10.1073/pnas.2414086122). It returns a probability
from 0 to 1 of avoiding the study's first major-condition-or-death endpoint over
the next ten years. Higher scores indicate better modeled healthspan. The
endpoint includes cancer excluding nonmelanoma skin cancer, diabetes, heart
failure, myocardial infarction, stroke, COPD, dementia, and death. The score
does not express years of life or guarantee the absence of every disease.

Supply all **86 protein NPX columns plus `age` in years**. The model was developed
from UK Biobank plasma measured with **Olink Explore 3072**, with within-batch and
across-batch intensity normalization. Use the original lowercase protein
identifiers. `nppb` and `ntprobnp` are separate required assays; retain both.
HPS applies no further scaling, normalization or imputation during prediction.
Missing columns raise an error, and supplied NaNs propagate. The study used
upstream k-nearest-neighbor imputation with `k=10`; any imputation remains a
separate data-preparation step.

```python
import anndata
import pyaging as pya

# npx: samples x proteins, with assay-normalized Olink Explore 3072 NPX values.
# age: numeric Series in years, indexed by the same sample identifiers.
inputs = npx.copy()
inputs.columns = inputs.columns.str.lower()
inputs["age"] = age.reindex(inputs.index)
if inputs.columns.duplicated().any():
    raise ValueError("Resolve duplicate protein identifiers before scoring")
adata = anndata.AnnData(inputs)  # preserves NaNs without implicit imputation
pya.pred.predict_age(adata, "hps")
hps_probability = adata.obs["hps"]
```

The Gompertz model was fitted in 30,184 adults initially free of the conditions
in its healthspan definition. Their healthy cohort ranged from 39 to 70 years.
It uses chronological age as a predictor and does not return an age-adjusted
residual. The [August 2025 correction](https://doi.org/10.1073/pnas.2520058122)
changes author affiliations only. The
[original HPS repository](https://github.com/kuo-lab-uchc/HPS) provides the fitted
parameters; the publication is CC BY-NC-ND 4.0, and the repository does not
declare a code license. HPS is marked for research use in pyaging.

## PAOPAC

`paopac` implements the **Conventional** output from the
[authors' MIT-licensed v1.0.0 release](https://github.com/JackieHanLab/PAOPAC/releases/tag/v1.0.0),
associated with the [2026 PAOPAC preprint](https://doi.org/10.64898/2026.04.24.720503).
The original model has 800 LightGBM trees and returns years after a correction
fitted to the supplied cohort. It uses **189 protein predictors**, an optional
`TDI` input, and required chronological `age`: 191 public features in total.
Supply plasma Olink Explore 3072 NPX values and age in years.

The original interface averages exact duplicate protein names on the NPX scale,
then uppercases names. Names that become duplicates only after uppercasing are
rejected. It exponentiates NPX with `2**NPX`, replaces missing or absent values
with zero on that linear scale, and fits `StandardScaler` across the complete
submitted cohort. After tree prediction, it subtracts a LOWESS fit of
predicted-minus-chronological age against chronological age, using `frac=0.75`
and the original statsmodels defaults. These steps were verified by executing
the authors' compiled interface; the preprint's full methods were inaccessible
during this audit. pyaging performs these steps, so supply NPX without applying
the exponentiation or standardization yourself.

`TDI` is normally absent and becomes zero. If supplied, it must be a column in
the protein matrix and undergoes the same exponential transform as the protein
inputs. A `TDI` column in observation metadata is ignored. This reproduces the
author interface and is not a conventional adjustment for a raw Townsend
deprivation score. Preserve NaNs in the input to retain the original
linear-scale zero-fill rule; imputing NPX to zero beforehand produces a
different input because `2**0` is one.

```python
import anndata
import pyaging as pya

# npx: the full cohort, samples x assay-normalized Olink NPX proteins.
# age: numeric Series in years, indexed by the same sample identifiers.
inputs = npx.copy()
inputs["age"] = age.reindex(inputs.index)
adata = anndata.AnnData(inputs)  # preserves NaNs and original protein names
pya.pred.predict_age(adata, "paopac")
paopac_years = adata.obs["paopac"]
```

**Cohort composition changes the result**, including when the same sample is
submitted with different participants or ages. The complete prediction call,
not `batch_size`, defines that cohort. A singleton returns exactly its supplied
chronological age. Cohorts with all proteins missing may also collapse to their
chronological ages, and equal-age ties can make LOWESS results depend on row
order. These are verified behaviors of the original interface. They make
individual scoring or longitudinal comparisons across separately formed tiny
cohorts inappropriate. The output is not a portable age-acceleration residual.
PAOPAC is marked for research use.

The Nature Biotechnology trial supplement identifies the Conventional endpoint
as `Han_2025(Conventional)` in worksheet S2, cell F4. This does not establish that
its private trial artifact or preparation was identical to the later public
release implemented here. The original MIT license is preserved with the
converted model; the preprint is CC BY 4.0.

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
| PAOPAC | `paopac`, the public Conventional release with its cohort-dependent correction; exact trial-version equivalence is unverified |

This is support for the original models, not an automatic reproduction of the
trial analysis. The paper used conventional OrganAge models, standardized each
protein and matched its SD to the original reference, and converted mortality
outputs to years. Its assay was Explore 3072; the original OrganAge authors
recommend the full models for this assay. The trial's public demo also shows a
reduced-model example, so the panel measurement alone does not prove the exact
coefficient file used for every published trial result.

The comparison paper obtained ProtAge weights through direct collaboration and
describes fitted reference min–max scaling followed by median centering. A
feature-importance table cannot substitute for those trained models. ProtAge
and ipfP3GPT remain absent from the executable catalogue because their required
fitted assets are unavailable through the public author sources. PAOPAC's
public model and compiled interface were recovered and independently checked
against native predictions; its cohort-dependent behavior is documented above.
