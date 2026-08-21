# tAge reference fixtures

Ground truth for the pyaging tAge clocks, produced by running the authors' own
R preprocessing pipeline and their published sklearn models on the example data
that ships with their package. Every numeric tAge test compares against these
files; nothing here was computed by a pyaging code path, so parity tests cannot
pass by agreeing with themselves.

## Provenance

| | |
|---|---|
| Reference implementation | https://github.com/Gladyshev-Lab/tAge |
| Clone commit | `0dba58fba356fecfbbb7c6f0cb27ced59ee6f99f` (2026-07-31, "Logo improvements") |
| Models | Zenodo record [18763485](https://zenodo.org/records/18763485) |
| Input data | `inst/extdata/Exprs_example.csv` + `Metadata_example.csv` from the clone |
| Generators | `clocks/generate_tage_fixtures.R`, then `clocks/generate_tage_fixtures.py` |

Both generators are deterministic: re-running them regenerates identical
contents for every file here. The uncompressed stage CSVs, `input_metadata.csv`,
and `expected_predictions.json` come back byte-for-byte. The `.csv.gz`
artifacts are produced by a separate `gzip -9n` step (see "Regenerating"), which
the generators do not perform themselves; `-n` is required because gzip
otherwise embeds the source filename and mtime in the header, which would make
the committed bytes irreproducible. With `-n` the compressed bytes are
reproducible too, as the header timestamp is zeroed.

The example dataset is 24 mouse samples — kidney and skeletal muscle, Klotho KO
and wild type, 6 per cell — with raw counts for 57 010 mouse Ensembl gene IDs.
`input_metadata.csv` carries `Mouse.ID`, `Genotype`, `Sex`, `Tissue`; note that
it has **no age column**, so the units conclusion below rests on the authors'
source constants rather than on a correlation against known ages.

The full sample set is used; no subsetting was needed to stay under the
pre-commit file size limit (largest committed file: 3.0 MB, limit 12 MB).

`input_expression.csv.gz` and `input_metadata.csv` are the R script's
`write.csv` round-trip of the two upstream files rather than byte copies — R
adds quoting the originals lack. The parsed contents are identical (verified
with `DataFrame.equals`: 57 010 x 24 and 24 x 4).

## Reference group

The relative (`_diff`) clocks require reference-centred input.
`reference_group_sample_ids.txt` holds the 12 wild-type samples
(`Genotype == "WT"`, both tissues), which is the reference group used for the
`center_refgroup` stage and predictions.

## Pipeline stages

The R script mirrors the `scaled_diff` branch of `tAge::tAge_preprocessing()`
(`R/preprocessing.R:413`) stage for stage, dumping the intermediate matrix after
each one. Resolved function names, in order:

| Fixture | tAge function | What it does |
|---|---|---|
| — | `make_ExpressionSet(exprs, meta)` | wraps counts + metadata into a Biobase `ExpressionSet` |
| — | `filter_genes(count_threshold = 10, percent_threshold = 20)` | keeps genes with count >= 10 in >= 20 % of samples |
| `after_mapping` | `map_genes(species = "mouse", gene_mapping_type = "Ensembl")` | maps mouse Ensembl IDs to **Entrez** IDs (15 991 genes) |
| `after_rle` | `RLE_normalization` | edgeR `calcNormFactors(method = "RLE")` scaled by library size, times 1e7 |
| `after_log` | `log_transform` | `log10(x + 1)` |
| `after_scale` | `scale_eset` | base R `scale()`, i.e. z-score **down columns (per sample)** |
| `after_align` | `tAge:::.align_to_gene_list(x, load_gene_list())` | see below |
| `after_center_all` | `control_subtraction(column_name = NULL, control_label = NULL)` | subtracts the per-gene median over **all 24 samples** |
| `after_center_refgroup` | `control_subtraction(column_name = "Genotype", control_label = "WT")` | subtracts the per-gene median over the **12 WT samples** |

`filter_genes` output is not dumped separately because `after_mapping` is taken
immediately after it, and `map_genes` is the step that changes the gene ID
space that everything downstream is keyed on.

### The `after_align` stage

This stage is not in the original task sketch; it exists because the real
pipeline has it. `.align_to_gene_list` (`R/preprocessing.R:350`, an internal
function, hence the `:::`) reindexes the matrix onto the package's fixed clock
gene list `inst/extdata/Gene_list_all_4.6.txt` (18 696 mouse Entrez IDs):
rows are reordered to the gene list's order, genes not in the list are dropped,
and genes in the list but absent from the data are **padded with `NA`**.

The padding is deliberate, not a defect. At prediction time the model pipeline's
`SimpleImputer` fills those `NA`s with the training-set median for each gene,
which is the authors' intended handling of genes an experiment did not measure.
Any reimplementation must pad with `NA` and let the imputer act — filling with
zeros instead would silently shift predictions.

Because alignment happens *before* centring, the `NA`s propagate: the
`after_center_*` matrices carry them too.

## Matrix orientation and NA semantics

All stage CSVs are written in **R orientation: genes as rows, samples as
columns**. Row names are gene IDs (mouse Ensembl for `input_expression.csv`,
mouse Entrez from `after_mapping` onward), column names are sample IDs.
Consumers must transpose to get sklearn's samples x features layout.

Shapes and missingness:

| Fixture | Shape (genes x samples) | All-NA genes |
|---|---|---|
| `input_expression` | 57 010 x 24 | 0 |
| `after_mapping`, `after_rle`, `after_log`, `after_scale` | 15 991 x 24 | 0 |
| `after_align`, `after_center_all`, `after_center_refgroup` | 18 696 x 24 | 3 542 |

Missingness is all-or-nothing per gene: the 85 008 NA cells in the aligned
matrices are exactly the 3 542 padded genes times 24 samples. No gene is
partially observed, so a gene is either fully numeric or fully `NA` (for
example Entrez `100009600`, the first row of `after_center_all.csv`).

Of the 10 487 features the models actually use, 375 are all-NA in this dataset
and are supplied entirely by the imputer.

## How the prediction script handles NA and feature selection

`clocks/generate_tage_fixtures.py` follows `inst/python/tage_predict.py` from
the clone:

1. Read the stage CSV and transpose if the model's genes match the index better
   than the columns (`orient()` mirrors `predict_tAge`'s `idx_overlap >
   cols_overlap` test); stringify the column labels, since the model's feature
   names are strings while pandas reads the Entrez IDs as integers.
2. Select exactly `feature_names_in_`, in the model's order
   (`matrix.loc[:, features]`, mirroring `_align_features`). This both subsets
   18 696 -> 10 487 and fixes the order. Missing features are a hard error, not
   a fill — the alignment stage has already guaranteed every one is present.
3. `NA` is passed through to `clock.predict` untouched, letting the pipeline's
   own imputer handle it.
4. Rescale the chronological clock only (see units below).

Both clocks carry the identical 10 487-name feature list in the identical order.

## Model objects

Each `.pkl` is a `sklearn.pipeline.Pipeline` (fitted under scikit-learn 1.3.2),
not a bare estimator, with four steps:

```
imputation        SimpleImputer      (fills the padded NA genes with training medians)
scaler            StandardScaler
featureselection  SelectKBest        (k = 10487, i.e. a pass-through here)
estimator         ElasticNet         (alpha = 0.001, l1_ratio = 0.2; 1 839 of 10 487 coefficients nonzero)
```

Because the imputer, scaler, and selector are fitted parts of the model, a
pyaging port must reproduce all four steps, not just the ElasticNet
coefficients. `feature_names_in_` comes from the `Pipeline`.

Loading these under a modern scikit-learn needs one shim, which both the
authors' script and ours apply: models fitted before scikit-learn 1.3 have a
`SimpleImputer` without the `_fill_dtype` attribute that later versions expect,
so it is set from `statistics_.dtype`. Unpickling also emits
`InconsistentVersionWarning`; the predictions were verified to reproduce
byte-identically regardless.

SHA-256 of the Zenodo downloads (the `.pkl` files themselves are not committed —
the script re-downloads them):

```
66b33cbad312b16a53733444defb4ec0b59a896b73c9472501d1f68b2a3184c1  EN_Chronoage_Multispecies_Multitissue_scaleddiff.pkl
458df7680cfa1422fb0aea42dcb1959142d55b9195e993e26bacee4807525566  EN_Mortality_Multispecies_Multitissue_scaleddiff.pkl
```

## Units

**`tage`: months of mouse chronological age.** The chronological clocks are
trained against age expressed as a fraction of species maximum lifespan, so the
raw ElasticNet output is a lifespan fraction. Converting it to age units means
multiplying by the species maximum lifespan, and
`inst/python/tage_predict.py:19` gives that table:

```python
PREDICTIONS_SPECIES_ADJ = {"human": 122.5, "mouse": 48, "rat": 50.4, "monkey": 39}
```

Hence `mouse_max_lifespan_months = 48` in `expected_predictions.json`; the
authors' comment states the result is "years for human, months for rodents".

**`tagemortality`: log10(hazard ratio), never rescaled.** The rescaling applies
only to chronological clocks. `inst/extdata/clocks_metadata.csv` carries this as
a per-model `lifespan_scaled` flag — `TRUE` for
`EN_Chronoage_Multispecies_Multitissue_scaleddiff.pkl`, `FALSE` for
`EN_Mortality_Multispecies_Multitissue_scaleddiff.pkl` — which is what the
`LIFESPAN_SCALED` constant in our script encodes.

### Why the values are negative

`tage_center_all` runs from about -9.8 to +8.2 months and `tage_center_refgroup`
from about -6.5 to +11.4, both straddling zero. This is expected and is not a
sign of a broken pipeline. These are **relative** (`_diff`) clocks: their input
is expression with a reference-group median already subtracted, so a prediction
is an age *difference* against that reference, not an absolute age. A sample
that looks transcriptomically like the reference group scores near zero;
negative means younger-looking than the reference, positive means older-looking.

The two centrings differ by roughly a constant offset (about +3.2 months for
`tage`), which is exactly what shifting the reference from all 24 samples to the
12 wild-type samples should do — the Klotho KO animals score older, so
excluding them from the reference lowers the baseline.

Absolute-age interpretation would require the non-`diff` clocks, which are not
covered by these fixtures.

## Files

Committed:

- `input_expression.csv.gz` — raw counts, mouse Ensembl x 24 samples
- `input_metadata.csv` — sample annotations
- `after_mapping.csv.gz`, `after_rle.csv.gz`, `after_log.csv.gz`,
  `after_scale.csv.gz`, `after_align.csv.gz`, `after_center_all.csv.gz`,
  `after_center_refgroup.csv.gz` — stage-wise intermediates
- `reference_group_sample_ids.txt` — the 12 WT sample IDs
- `expected_predictions.json` — predictions for both clocks under both centrings
- `README.md` — this file

The two `.pkl` model files are **not** committed; `generate_tage_fixtures.py`
downloads them from Zenodo on demand.

## Regenerating

```bash
git clone https://github.com/Gladyshev-Lab/tAge "$SCRATCH/tAge"
git -C "$SCRATCH/tAge" checkout 0dba58fba356fecfbbb7c6f0cb27ced59ee6f99f
Rscript clocks/generate_tage_fixtures.R "$SCRATCH/tAge" tests/data/tage
uv run python clocks/generate_tage_fixtures.py tests/data/tage

# Compress. -9n is required, not just -9: -n omits the source filename and
# mtime from the gzip header, without which the committed bytes are not
# reproducible. Do not drop it.
cd tests/data/tage
gzip -9n input_expression.csv after_*.csv
```

The R script writes uncompressed `.csv` and neither generator compresses
anything, so the `gzip -9n` step above is a required part of regenerating —
without it the R script's output does not match what is committed here.

The uncompressed stage CSVs are about 50 MB in total and are deliberately not
committed. `.gitignore` carries `tests/data/tage/*.csv` with an exception for
`input_metadata.csv`, so leaving them in place after a regeneration cannot
accidentally commit them; delete them or leave them, either is safe.

The R script installs missing Bioconductor/CRAN dependencies (`Biobase`,
`edgeR`, `reticulate`, `ggplot2`) into a local library beside the clone. It
honours `TAGE_FIXTURE_SAMPLES`, a comma-separated sample-ID list, if the
fixtures ever need to be shrunk. The Python script reads the `.csv.gz` files
directly, so it can be re-run against the committed fixtures without re-running
R.
