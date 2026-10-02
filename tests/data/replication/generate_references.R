# Original-author R execution, independent of the pyaging implementation.
# Usage: Rscript generate_references.R /tmp/CellDRIFT-original /tmp/celldrift_fixture_input.csv /tmp/celldrift_expected.csv
args <- commandArgs(trailingOnly=TRUE)
suppressPackageStartupMessages(library(glmnet))
for (f in list.files(file.path(args[1], 'data'), pattern='\\.rda$', full.names=TRUE)) load(f)
source(file.path(args[1], 'R', 'CellDRIFT.R'))
input <- as.matrix(read.csv(args[2], row.names=1, check.names=FALSE))
full <- CellDRIFT(input)
dropped <- seq.int(1, ncol(input), by=13)
partial <- CellDRIFT(input[, -dropped, drop=FALSE])
write.csv(data.frame(sample=rownames(input), full=full, partial=partial), args[3], row.names=FALSE)
cat('R ', R.version.string, '; glmnet ', as.character(packageVersion('glmnet')), '\n', sep='')
