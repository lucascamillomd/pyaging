# Clinical PhenoAge Gompertz correction

The [2019 Liu et al. correction](https://doi.org/10.1371/journal.pmed.1002760)
is the source for clinical PhenoAge in pyaging 0.6.1. It supersedes the
2018 calibration described below. The corrected artifact uses a log-hazard
intercept of −19.907 and an alkaline phosphatase coefficient of 0.00188.
The runtime evaluates the printed mortality-to-age equation in log space:

```text
PhenoAge = 141.50 + log1p(0.00553 * 1.51714 * exp(xb) / 0.0076927) / 0.09165
```

The leading 1 inside the published logarithm gives 141.50 years as the lower
limit. Use the rebuilt weights with pyaging 0.6.1 or later.

## Historical pyaging 0.5.7 calibration

The [original Levine et al. supplement](https://cdn.aging-us.com/article/101414/supplementary/SD1/0/aging-v10i4-101414-supplementary-material-SD1.pdf), checked 2026-10-02, distinguishes the Cox variable-selection penalty `lambda = 0.0192` (page 1) from the Gompertz ancillary parameter `gamma = 0.0076927` (page 2). Table S1 on page 10 reports the latter rounded to 0.0077.

Earlier pyaging versions used the selection penalty in the Gompertz calculation. Version 0.5.7 uses the published gamma, retaining the stable algebraic form of the mortality-to-age conversion:

```text
PhenoAge = 141.50225 + [xb + log(0.00553 * expm1(120 * 0.0076927) / 0.0076927)] / 0.090165
```

For an identical finite log hazard `xb`, this reduces the old result by **9.619364901703968 years**. Very old implementations that rounded mortality to zero or one could also produce nonfinite results, so a blanket subtraction is not a substitute for recalculation. The rebuilt notebook stores the printed Table S1 coefficients directly in float64 instead of first rounding them to float32; this removes an additional small numerical discrepancy. Feature names, raw CRP units and the existing CRP floor are unchanged.

This concerns clinical `phenoage`. It does not change the separate fitted `dnamphenoage`, PC or PRC/non-PRC methylation models, or the separately fitted Sao Paulo model.

The notebook now explains both constants, executes fixed reference checks and no longer deletes unrelated files in its working directory. A separate R 4.5.3 evaluation gives 9.37497303957636, 42.64730709381026 and 75.91964114804416 years for log hazards -12, -9 and -6, respectively. Regression tests fail on the previous implementation and match these values after the correction. The native formula and its published mortality/CDF expression are checked independently.

The corrected Hugging Face artifact, model card, config, provenance and shared aggregate require **pyaging >=0.5.7**. Torch objects refer to the installed Python class: updating a cached weight object alone cannot replace the formula in an older package. Historical release tags remain untouched.
