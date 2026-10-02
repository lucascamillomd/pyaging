# Run with Rscript generate_reference.R /tmp/organage-sources OUTPUT.csv
# The scoring expressions are read verbatim from the pinned authors' script,
# so this oracle does not reimplement the Python port.
args <- commandArgs(trailingOnly=TRUE)
root <- args[1]
output <- args[2]
script <- readLines(file.path(root, "scripts/R/Example_Script.R"), warn=FALSE)
score_lines <- grep("predicted.ages\\[\\[.*\\]\\]\\[\\[\"predicted\"\\]\\]\\[\\[k\\]\\] <-", script, value=TRUE)
stopifnot(length(score_lines) == 2L)
coefficient_line <- grep("^    coefs <-", script, value=TRUE)[1]
input <- read.csv(file.path(root, "input_npx.csv"), row.names=1, check.names=FALSE)
paths <- sort(list.files(file.path(root, "data/output_Python"), pattern="\\.csv$", recursive=TRUE, full.names=TRUE))
results <- list()
for (path in paths) {
    mortality <- grepl("/mortality_based_models/", path, fixed=TRUE)
    reduced <- grepl("/feature_reduced/", path, fixed=TRUE)
    organ <- strsplit(basename(path), "_", fixed=TRUE)[[1]][1]
    clock <- paste0("organage", if(mortality) "mortality" else "chronological", "olink", if(reduced) "1500" else "3000", tolower(gsub("-", "", organ, fixed=TRUE)))
    coefficients <- list(gen1=list(), gen2=list())
    generation <- if(mortality) "gen2" else "gen1"
    coefficients[[generation]][[1]] <- unlist(read.csv(path, header=TRUE, check.names=FALSE)[1, , drop=FALSE])
    for (case in c("complete", "omitted_proteins", "present_na")) {
        df.full <- input
        if(case == "omitted_proteins") df.full <- df.full[, seq_len(ncol(df.full)) %% 3 != 0, drop=FALSE]
        if(case == "present_na") {
            protein <- setdiff(names(coefficients[[generation]][[1]]), "Intercept")[1]
            df.full[1, protein] <- NA_real_
        }
        k <- 1L
        predicted.ages <- list(gen1=list(predicted=list()), gen2=list(predicted=list()))
        # This line is identical in both original scoring loops except the gen key.
        current_line <- gsub('gen1', generation, coefficient_line, fixed=TRUE)
        eval(parse(text=current_line))
        eval(parse(text=score_lines[if(mortality) 2L else 1L]))
        results[[length(results)+1L]] <- data.frame(clock=clock, case=case, sample_id=rownames(df.full), prediction=unname(predicted.ages[[generation]][["predicted"]][[1]]))
    }
}
write.csv(do.call(rbind, results), output, row.names=FALSE, na="NA")
cat(R.version.string, "\n", nrow(do.call(rbind,results)), "predictions\n")
