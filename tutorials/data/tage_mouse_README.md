# Mouse RNA-seq example for tAge

The transcriptomics tutorial loads the count matrix and metadata distributed by
the [tAge authors](https://github.com/Gladyshev-Lab/tAge). The source contains
57,010 mouse Ensembl genes and 24 bulk RNA-seq samples: kidney and skeletal
muscle from the same 12 male mice, six wild type and six Klotho knockout.
Sample labels and counts are retained. Ages are not supplied.

## Sources

Both files are pinned to commit `0dba58fba356fecfbbb7c6f0cb27ced59ee6f99f`.

| File | SHA-256 |
| --- | --- |
| [Exprs_example.csv](https://raw.githubusercontent.com/Gladyshev-Lab/tAge/0dba58fba356fecfbbb7c6f0cb27ced59ee6f99f/inst/extdata/Exprs_example.csv) | `6abfaa8a73988156d0fcf4a680eddcc31dbbb419b8fa05d241e2e8c67cce5494` |
| [Metadata_example.csv](https://raw.githubusercontent.com/Gladyshev-Lab/tAge/0dba58fba356fecfbbb7c6f0cb27ced59ee6f99f/inst/extdata/Metadata_example.csv) | `9ba1c6f53f4adaacc7a6e0f5a73b5f12ef12189a1d5b835469f3362c7bc70d70` |

The parsed counts match the independently generated
[`tests/data/tage/input_expression.csv.gz`](../../tests/data/tage/input_expression.csv.gz)
fixture. The notebook transposes the matrix to samples by genes, checks sample
alignment, and joins the four metadata columns. It does not synthesize, impute,
rescale, or select genes before calling pyaging.

## Use and interpretation

The tutorial processes each tissue separately and uses its six wild-type samples
as the reference. Both tissues come from the same animals, so they are not
independent biological replicates. The small groups illustrate the API and
do not establish clock calibration or accuracy.

This example does not reproduce the paper's analysis excluding Klotho. It uses pyaging's packaged `scaled_diff` models
and their preprocessing defaults; the upstream package has additional models
and options. See the [authors' current example](https://gladyshev-lab.github.io/tAge/articles/tage-bulk.html)
and [publication](https://doi.org/10.1038/s41586-026-10542-3).

The upstream data and models use the
[MGB Open Access License 1.0](https://github.com/Gladyshev-Lab/tAge/blob/0dba58fba356fecfbbb7c6f0cb27ced59ee6f99f/LICENSE)
for non-commercial, non-revenue-generating academic use. The notebook downloads
the original files directly; this change does not mirror them to Hugging Face
or redistribute a modified count matrix.
