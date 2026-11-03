# DNAm median fallback validation for pyaging 0.6.0

## Reference generation

The shipped lookup contains 984,158 canonical CpGs. Hannum contributed medians
for 470,043 CpGs across 656 whole-blood samples. BoA contributed medians for
894,006 CpGs across 500 samples. The two cohorts share 379,891 CpGs. The
remaining 90,152 Hannum and 514,115 BoA CpGs use their single cohort median.
The packaged resource records source URLs, checksums, counts, and the
aggregation rule. [Hannum GEO record](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE40279) ·
[BoA GEO record](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE246337)

Two full builds produced identical compressed bytes. The resource SHA-256 is
`880afc302e9f5b4af266f05958e3b1af17f5217ec106a4c39011c5cbe3b290d9`.

Independent checks read the source rows and recomputed these medians:

| CpG | Hannum median | BoA median | Packaged beta |
| --- | ---: | ---: | ---: |
| cg00000029 | 0.47498965 | 0.51372326 | 0.49435645 |
| cg00000108 | 0.94484165 | | 0.94484165 |
| cg00000221 | | 0.92236831 | 0.92236831 |

```sh
uv run python clocks/build_dnam_feature_medians.py \
  GSE40279_series_matrix.txt.gz GSE246337_betas.csv.gz \
  src/pyaging/data/dnam_feature_medians.json.gz
```

## Held-out predictions

The benchmark uses deposited beta values from the Dutch whole-blood cohort
GSE41169 and the IDH-mutant low-grade glioma cohort GSE279950, independent of
the two reference cohorts. The baseline for each clock includes only samples
with finite beta values in [0, 1] for all its inputs. The benchmark checks exact
prediction equality on these complete inputs with the fallback enabled and
disabled. Hannum lacked six baseline CpGs in the glioma matrix, so the
glioma comparison uses DNAmPhenoAge. [GSE41169](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE41169) ·
[GSE279950](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE279950)

For each clock and cohort, remove the same columns for both fill methods.
Random masks use NumPy seed 600 and one draw per fraction. The observed panel
mask removes CpGs absent from the deposited GSE224218 processed matrix.
That mask reflects the study's processing and array coverage together.
[Observed panel source](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE224218)

Each pair in the results table is zero fill / median fill. Changes are absolute
differences from complete-input predictions in years. Chronological MAE is
reported for Hannum in blood.

| Tissue | Clock | Samples | Mask | Covered / masked | Median change | P95 change | Age MAE |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| Whole blood | hannum | 91 | Random 5% | 4/4 | 4.21 / 0.97 | 4.81 / 1.57 | 5.07 / 3.09 |
| Whole blood | hannum | 91 | Random 10% | 7/7 | 1.78 / 4.81 | 4.69 / 6.15 | 3.53 / 5.26 |
| Whole blood | hannum | 91 | Random 25% | 17/18 | 15.40 / 9.26 | 22.87 / 12.19 | 15.77 / 9.00 |
| Whole blood | hannum | 91 | Observed panel | 8/8 | 8.45 / 3.21 | 11.75 / 4.84 | 8.49 / 4.02 |
| Whole blood | dnamphenoage | 75 | Random 5% | 26/26 | 17.83 / 1.13 | 18.97 / 2.19 |  |
| Whole blood | dnamphenoage | 75 | Random 10% | 51/51 | 24.41 / 1.17 | 27.29 / 3.43 |  |
| Whole blood | dnamphenoage | 75 | Random 25% | 128/128 | 29.02 / 5.93 | 35.79 / 9.85 |  |
| Whole blood | dnamphenoage | 75 | Observed panel | 3/3 | 0.80 / 0.15 | 1.17 / 0.38 |  |
| Glioma | dnamphenoage | 109 | Random 5% | 26/26 | 11.52 / 5.38 | 18.55 / 10.98 |  |
| Glioma | dnamphenoage | 109 | Random 10% | 51/51 | 38.75 / 14.53 | 49.36 / 25.14 |  |
| Glioma | dnamphenoage | 109 | Random 25% | 128/128 | 26.31 / 61.26 | 42.25 / 77.19 |  |
| Glioma | dnamphenoage | 109 | Observed panel | 3/3 | 1.66 / 0.83 | 2.83 / 2.01 |  |

Median fill reduced both change statistics in 10 of the 12 comparisons.
The Hannum 10% blood mask increased median prediction change from 1.78 to
4.81 years and age MAE from 3.53 to 5.26 years. The 25% glioma mask increased
DNAmPhenoAge's median change from 26.31 to 61.26 years. The blood-derived
reference can produce substantial shifts with extensive missingness in
tumor tissue. These comparisons measure sensitivity to selected masks and
do not establish clock validity for glioma.

```sh
uv run python clocks/validate_dnam_feature_medians.py \
  GSE41169_series_matrix.txt.gz hannum.pt dnamphenoage.pt \
  --blood --panel GSE224218_processed_data.txt.gz
uv run python clocks/validate_dnam_feature_medians.py \
  GSE279950_MPT_tabProcessed_PBADY.csv.gz dnamphenoage.pt \
  --panel GSE224218_processed_data.txt.gz
```

| Validation input | SHA-256 |
| --- | --- |
| GSE41169_series_matrix.txt.gz | `2f215e8b72d3c97c7793114f72cd2251f1ef2b225924c88a7bc1c414043663a0` |
| GSE279950_MPT_tabProcessed_PBADY.csv.gz | `6576be2b1e9e0da03b35ee3d5f3246bf39b712bdc31e0ffdcf7c30f844c87324` |
| GSE224218_processed_data.txt.gz | `ad862b172364303ba7da02649c56b806985e8bf741bbc41059bf2d69cb7c7103` |
| hannum.pt | `ef675563e3df7184afade17a81094a242fb1899c022621fcba58d2e4b10b21d8` |
| dnamphenoage.pt | `a0e384e57d1d69f523f7e76cefc78d2467caa2a6c38f190dff25f1498516a47b` |

The weights came from the [Hannum](https://huggingface.co/pyaging/hannum)
and [DNAmPhenoAge](https://huggingface.co/pyaging/dnamphenoage) repositories.
