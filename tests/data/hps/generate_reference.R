# Usage: Rscript generate_reference.R ORIGINAL_HPS_DIRECTORY FIXTURE_DIRECTORY
# Source and execute the original author's function without modification.
args <- commandArgs(trailingOnly=TRUE)
source(file.path(args[1], "HPS.R"))
input <- read.csv(file.path(args[2], "input_npx.csv"), row.names=1, check.names=FALSE)
results <- list()
for (case in c("complete", "present_na", "reversed_features")) {
  data <- input
  if (case == "present_na") data[1, "ace2"] <- NA_real_
  if (case == "reversed_features") data <- data[, rev(seq_len(ncol(data)))]
  prediction <- HPS(data)
  results[[length(results) + 1L]] <- data.frame(
    case=case, sample_id=rownames(data), prediction=as.vector(prediction)
  )
}
stopifnot(is.null(HPS(input[, names(input) != "ace2"])))
write.csv(do.call(rbind, results), file.path(args[2], "expected_predictions.csv"), row.names=FALSE, na="NA")
cat(R.version.string, "\n")
