#!/usr/bin/env Rscript
# Generate tAge parity fixtures from the authors' reference implementation
# (https://github.com/Gladyshev-Lab/tAge).
#
# Run from the repo root:
#   Rscript clocks/generate_tage_fixtures.R <path-to-tAge-clone> tests/data/tage
#
# Missing R dependencies (edgeR, reticulate) are installed into a local library
# under the clone directory, the same trick clocks/notebooks/kdmage.ipynb uses.
#
# The stage order mirrors tAge_preprocessing() for the scaled_diff space:
#   filter_genes -> map_genes -> RLE_normalization -> log_transform ->
#   scale_eset -> .align_to_gene_list -> control_subtraction

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("usage: Rscript generate_tage_fixtures.R <tAge-clone-dir> <out-dir>")
}
tage_dir <- normalizePath(args[1], mustWork = TRUE)
out_dir <- args[2]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

local_library <- file.path(tage_dir, "..", "rlib")
dir.create(local_library, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(normalizePath(local_library), .libPaths()))

ensure_bioc <- function(package) {
  if (requireNamespace(package, quietly = TRUE)) {
    return(invisible(NULL))
  }
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = "https://cloud.r-project.org", lib = local_library)
  }
  BiocManager::install(package, lib = local_library, ask = FALSE, update = FALSE)
}
ensure_cran <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) {
    install.packages(package, repos = "https://cloud.r-project.org", lib = local_library)
  }
}
ensure_bioc("Biobase")
ensure_bioc("edgeR")
ensure_cran("reticulate")
ensure_cran("ggplot2")

devtools::load_all(tage_dir, quiet = TRUE)

# The example data ships as raw counts, genes (mouse Ensembl IDs) x samples.
exprs <- read.csv(file.path(tage_dir, "inst/extdata/Exprs_example.csv"),
                  row.names = 1, check.names = FALSE)
meta <- read.csv(file.path(tage_dir, "inst/extdata/Metadata_example.csv"),
                 row.names = 1, check.names = FALSE)

# Sample subsetting hook: set TAGE_FIXTURE_SAMPLES to a comma-separated list of
# sample ids to shrink the fixtures. Empty means "use every sample".
keep <- Sys.getenv("TAGE_FIXTURE_SAMPLES")
if (nzchar(keep)) {
  keep <- trimws(strsplit(keep, ",")[[1]])
  exprs <- exprs[, keep, drop = FALSE]
  meta <- meta[keep, , drop = FALSE]
}

# Reference group for the relative ("diff") clocks: the wild-type animals.
REF_COLUMN <- "Genotype"
REF_LABEL <- "WT"
ref_ids <- rownames(meta)[meta[[REF_COLUMN]] == REF_LABEL]
stopifnot(length(ref_ids) > 0)

dump <- function(eset, name) {
  m <- if (inherits(eset, "ExpressionSet")) Biobase::exprs(eset) else as.matrix(eset)
  write.csv(as.data.frame(m), file.path(out_dir, paste0(name, ".csv")))
  cat(sprintf("  %-22s %5d genes x %2d samples\n", name, nrow(m), ncol(m)))
  invisible(m)
}

eset <- make_ExpressionSet(exprs, meta, verbose = FALSE)

filtered <- filter_genes(eset, count_threshold = 10, percent_threshold = 20, verbose = FALSE)
mapped <- map_genes(filtered, species = "mouse", gene_mapping_type = "Ensembl", verbose = FALSE)
dump(mapped, "after_mapping")

rle <- RLE_normalization(mapped, verbose = FALSE)
dump(rle, "after_rle")

logged <- log_transform(rle, verbose = FALSE)
dump(logged, "after_log")

scaled <- scale_eset(logged, verbose = FALSE)
dump(scaled, "after_scale")

# Alignment to the clock gene list pads absent genes with NA on purpose: the
# model pipeline's imputer fills them with the training-set median.
aligned <- tAge:::.align_to_gene_list(scaled, load_gene_list())
dump(aligned, "after_align")

cent_all <- control_subtraction(aligned, column_name = NULL, control_label = NULL, verbose = FALSE)
dump(cent_all, "after_center_all")

cent_ref <- control_subtraction(aligned, column_name = REF_COLUMN, control_label = REF_LABEL, verbose = FALSE)
dump(cent_ref, "after_center_refgroup")

writeLines(ref_ids, file.path(out_dir, "reference_group_sample_ids.txt"))
write.csv(exprs, file.path(out_dir, "input_expression.csv"))
write.csv(meta, file.path(out_dir, "input_metadata.csv"))

cat("Wrote fixtures to ", out_dir, "\n", sep = "")
cat("Reference group: ", REF_COLUMN, " == ", REF_LABEL, " (",
    length(ref_ids), " samples)\n", sep = "")
