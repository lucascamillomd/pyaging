# Execute the original authors' inference function and fitted parameter object.
# Usage: Rscript generate_miage_references.R /tmp/MiAge-original-mirror /tmp/miage_fixture_input.csv output.csv
args <- commandArgs(trailingOnly=TRUE)
source(file.path(args[1], 'function_library.r'))
load(file.path(args[1], 'site_specific_parameters.Rdata'))
input <- as.matrix(read.csv(args[2], row.names=1, check.names=FALSE))
full <- mitotic.age(t(input), methyl.age[[1]], methyl.age[[2]], methyl.age[[3]])
dropped <- seq.int(1, ncol(input), by=13)
partial <- mitotic.age(t(input[, -dropped, drop=FALSE]), methyl.age[[1]][-dropped], methyl.age[[2]][-dropped], methyl.age[[3]][-dropped])
write.csv(data.frame(sample=rownames(input), full=full, partial=partial), args[3], row.names=FALSE)
cat(R.version.string, '\n')
