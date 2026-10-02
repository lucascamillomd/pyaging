# Usage: Rscript generate_reference.R ORIGINAL_PAC_DIRECTORY FIXTURE_DIRECTORY
# Execute the pinned original author's function, not a rewritten formula.
args <- commandArgs(trailingOnly=TRUE)
source(file.path(args[1], "pac_proteomic_age.R"))
input <- read.csv(file.path(args[2], "input_npx.csv"), row.names=1, check.names=FALSE)
results <- list()
for (case in c("complete", "present_na", "reversed_features")) {
  data <- input
  if (case == "present_na") data[1, "ada2"] <- NA_real_
  if (case == "reversed_features") data <- data[, rev(seq_len(ncol(data)))]
  prediction <- pac_proteomic_age(data)
  results[[length(results) + 1L]] <- data.frame(
    case=case, sample_id=rownames(data), prediction=as.vector(prediction)
  )
}
stopifnot(is.null(suppressMessages(pac_proteomic_age(input[, names(input) != "ada2"]))))
write.csv(do.call(rbind, results), file.path(args[2], "expected_predictions.csv"), row.names=FALSE, na="NA")
cat(R.version.string, "\n")
