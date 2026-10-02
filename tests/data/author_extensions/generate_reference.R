# Run with original source directories and an output directory. No aggregator inputs.
# Rscript generate_reference.R <prcPhenoAge checkout> <IntrinClock original directory> <output>
args <- commandArgs(trailingOnly=TRUE)
prc_source <- args[1]
intrin_source <- args[2]
out <- args[3]
dir.create(out, recursive=TRUE, showWarnings=FALSE)
# Resolve data() to the exact .rda bundled by the authors, without installing readr.
data <- function(name) load(file.path(prc_source, 'data', paste0(name, '.rda')), envir=parent.frame())
source(file.path(prc_source, 'R/calcPRCPhenoAge.R'))
source(file.path(prc_source, 'R/calcnonPRCPhenoAge.R'))
all_weights <- read.csv(file.path(prc_source, 'data-raw/Clock_Levine_PRC.csv'))
n <- nrow(all_weights)
# Deterministic synthetic beta values, including extremes and supplied NA values.
x <- rbind(rep(0,n),rep(1,n),seq(0,1,length.out=n),seq(1,0,length.out=n),((seq_len(n)*37)%%997)/997)
colnames(x) <- all_weights$CpG
x <- rbind(x,x[5,]); x[6,c(1,16,18)] <- NA
pheno <- data.frame(sample=paste0('synthetic',seq_len(nrow(x))))
y <- calcPRCPhenoAge(x,pheno,imputation=FALSE)
y <- calcnonPRCPhenoAge(x,y,imputation=FALSE)
write.csv(x,file.path(out,'pheno_input.csv'),row.names=FALSE,na='NaN')
write.csv(y,file.path(out,'pheno_expected.csv'),row.names=FALSE)
write.csv(all_weights[c('CpG','Weight','PRC')],file.path(out,'pheno_coefficients.csv'),row.names=FALSE)
# Confirm source-script missing-column handling independently.
y_missing <- calcPRCPhenoAge(x[,-c(1,16,18)],pheno,imputation=FALSE)
y_missing <- calcnonPRCPhenoAge(x[,-c(1,16,18)],y_missing,imputation=FALSE)
write.csv(y_missing,file.path(out,'pheno_missing_expected.csv'),row.names=FALSE)
# Use the author's fitted cv.glmnet and age transform from the original demo.
library(glmnet)
model <- readRDS(file.path(intrin_source,'final_model_small.RData'))
eval(parse(file.path(intrin_source,'demo.R'))[[1]])
features <- rownames(model$glmnet.fit$beta)
n <- length(features)
x <- rbind(rep(0,n),rep(1,n),seq(0,1,length.out=n),seq(1,0,length.out=n),((seq_len(n)*37)%%997)/997)
colnames(x) <- features
predicted <- returnAge(predict(model,x))
write.csv(x,file.path(out,'intrin_input.csv'),row.names=FALSE)
write.csv(data.frame(predicted=as.numeric(predicted)),file.path(out,'intrin_expected.csv'),row.names=FALSE)
co <- as.matrix(coef(model,s='lambda.1se'))
write.csv(data.frame(feature=rownames(co),coefficient=co[,1]),file.path(out,'intrin_coefficients.csv'),row.names=FALSE)
