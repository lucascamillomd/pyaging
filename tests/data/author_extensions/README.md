# Original-author IntrinClock and PhenoAge references

`provenance.json` pins the author repository/archive and source checksums.
`generate_reference.R` generates the fixtures from those original files using
R 4.5.3 and glmnet 5.1. Inputs are deterministic synthetic beta matrices, with
zero/one extremes, distinct index patterns, and supplied NaNs for the PhenoAge
partitions. No methylCIPHER or Biolearn values enter these reference calculations.

```sh
Rscript tests/data/author_extensions/generate_reference.R \
  /tmp/pyaging-prc-original /tmp/intrinclock-original tests/data/author_extensions
uv run pytest tests/models/test_author_extensions.py
```

The two author PhenoAge functions are sourced without changes. Their `data()`
lookup is redirected to the exact original bundled `.rda` files so running the
oracle does not require installing an R package solely to locate its data.
The `imputation=FALSE` branch skips absent CpGs and supplied NaNs. Both
partitions omit the intercept. Maximum differences for the built artifacts
are below `6e-13` on these fixtures.

IntrinClock uses native `predict(cv.glmnet, newx)` with its default `lambda.1se`
selection. The original `returnAge` function is evaluated directly from the
first expression in the author demo. The selected coefficient column has 370
nonzero CpGs; the maximum built-artifact difference is `8.53e-14` years.

`secondary_comparison.json` separately records agreement with methylCIPHERv2.
It is not an implementation source. The original PRC package's DESCRIPTION has
an unresolved license placeholder; no upstream MIT grant is inferred.

Build weights with the optional build-only readers:

```sh
uv run --with rdata --with remotezip python clocks/build_author_extensions.py \
  --intrin-source /tmp/intrinclock-original --prc-source /tmp/pyaging-prc-original
```
