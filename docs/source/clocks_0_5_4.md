# Clocks added in 0.5.4

These models use the original authors' coefficients and prediction code. The
build scripts pin source revisions and checksums, and the regression fixtures
were produced with the authors' R implementations. Source provenance accompanies
the downloadable weights.

## DNA methylation

| Clock name | Output | Interpretation |
|---|---|---|
| `celldrift` | Population doublings | Replication-associated methylation drift calibrated to cumulative population doublings. |
| `miage` | Relative mitotic age | A methylation-derived measure of cumulative cell divisions, identifiable up to an unknown common scale factor. |
| `pcbrainage` | Years | Chronological age from brain methylation. |
| `intrinclock370` | Years | The original IntrinClock `lambda.1se` fit used by its author demo. |
| `dnamphenoageprc` | PhenoAge contribution | The 55 polycomb-related CpGs' weighted contribution, without an intercept. |
| `dnamphenoagenonprc` | PhenoAge contribution | The complementary 458 CpGs' weighted contribution, without an intercept. |

The PhenoAge partitions are not independently calibrated ages. With complete
beta data, adding their two scores and the original DNAmPhenoAge intercept
reconstructs the full clock.
They follow the authors' `imputation = FALSE` path: absent CpGs and supplied NaN
beta values contribute zero. Any imputation performed before prediction changes
that behavior; choose it deliberately.

The existing `intrinclock` remains the 380-CpG `lambda.min` implementation.
`intrinclock370` selects a different regularization column of the original fitted
model. Its CpGs and coefficients differ; it is not made by deleting ten terms
from the 380-CpG clock.

The standalone PCGrimAge components are `pcgrimagepackyrs`, `pcgrimageadm`,
`pcgrimageb2m`, `pcgrimagecystatinc`, `pcgrimagegdf15`, `pcgrimageleptin`,
`pcgrimagepai1`, and `pcgrimagetimp1`. They return the original PC-based smoking
and protein proxies rather than the final composite age. Consult each model's
feature list for required `age` (years) and `female` (0 or 1) covariates.
Include these as feature columns alongside the CpGs. Every selected covariate
must be finite at prediction time; the models do not supply demographic defaults.
Choose upstream imputation deliberately, since `df_to_adata` can impute missing
values before the model sees them.

```python
import pyaging as pya

# betas: samples by CpGs, with complete beta values in [0, 1].
adata = pya.pp.df_to_adata(betas, verbose=False)
pya.pred.predict_age(
    adata,
    ["CellDRIFT", "MiAge", "IntrinClock370", "DNAmPhenoAgePRC", "DNAmPhenoAgeNonPRC"],
    verbose=False,
)
scores = adata.obs
```

MiAge follows the authors' bounded numerical optimization with five starting
values. Its solver runs on the CPU, including when other clocks in the call run
on a GPU. It returns the same sample ordering and supports ordinary batching.

## Proteomic OrganAge

OrganAge includes 90 scalar outputs: chronological and mortality models for both
the Olink 3000 and 1500 panels. Each uses the authors' recommended fold-one
coefficients. The 3000 panel has 23 outputs and the 1500 panel has 22; the latter
has no thyroid model. Outputs include organ-specific models and the conventional,
organismal, and multiorgan models.

Names follow `organage{target}olink{panel}{organ}`. For example:

```python
# npx: samples by case-sensitive protein symbols, using assay-normalized Olink NPX.
# This example assumes complete values. Choose imputation explicitly if values are missing.
# Example symbols include ADIPOQ, CD300LG, FABP4, LEP, and NTproBNP.
proteins = pya.pp.df_to_adata(npx, verbose=False)
pya.pred.predict_age(
    proteins,
    ["organagechronologicalolink3000brain", "organagemortalityolink3000brain"],
    verbose=False,
)
brain_age_years = proteins.obs["organagechronologicalolink3000brain"]
brain_log_hazard = proteins.obs["organagemortalityolink3000brain"]
```

Use protein symbols, not UniProt accessions or Olink assay identifiers. Input
values are the assay-normalized NPX measurements used by the authors; pyaging
does not additionally standardize them within the input cohort or convert other
proteomic assays to this scale. Chronological outputs are years. Mortality
outputs are relative log-hazard scores, without conversion to years.

Absent proteins contribute zero. Supplied NaN values propagate unless they are
imputed before prediction. Inspect each clock's `*_percent_na` and
`*_missing_features` entries in `adata.uns` to assess missing protein coverage.
The authors' academic noncommercial license applies to the OrganAge weights and
is included in their Hugging Face repositories.

## Original sources

- [CellDRIFT](https://github.com/MorganLevineLab/CellDRIFT)
- [MiAge author website](https://www.columbia.edu/~sw2206/softwares.htm)
- [PCBrainAge](https://github.com/MorganLevineLab/calcPCBrainAge)
- [PC clocks and PCGrimAge components](https://github.com/MorganLevineLab/PC-Clocks)
- [PhenoAge partitions](https://github.com/MorganLevineLab/prcPhenoAge)
- [IntrinClock author code](https://doi.org/10.5281/zenodo.10426597)
- [Proteomic OrganAge](https://github.com/ludgergoeminne/organAging)
