# Export lossless linear compositions and independent original-author predictions.
# Only base R is needed. Original source files are checksum-verified by the builder.
args <- commandArgs(trailingOnly=TRUE)
source_dir <- args[[1]]
out_dir <- args[[2]]
options(digits=17)

make_inputs <- function(reference) {
  n <- length(reference)
  j <- seq_len(n) - 1
  x <- rbind(reference, rep(0.5,n), rep(0,n), rep(1,n),
             ((j*37) %% 1000)/1000, ((j*91+17) %% 1000)/1000,
             reference, ((j*37) %% 1000)/1000)
  x[7,c(1, floor(n/2)+1, n)] <- c(0.1, 0.9, 0.3)
  missing <- seq(1,n,by=97)
  x[8,missing] <- reference[missing]
  x
}

export <- function(name, features, reference, weight, bias, oracle) {
  stopifnot(length(features)==length(weight), length(reference)==length(weight),
            !anyDuplicated(features), all(is.finite(weight)),
            all(is.finite(reference[!features %in% c("female","age")])))
  table <- data.frame(feature=features, reference=sprintf('%.17g',reference),
                      weight=sprintf('%.17g',weight))
  con <- gzfile(file.path(out_dir,paste0(name,'.tsv.gz')),'wt')
  write.table(table,con,sep='\t',quote=FALSE,row.names=FALSE)
  close(con)
  writeLines(sprintf('%.17g',bias), file.path(out_dir,paste0(name,'.bias')))
  writeLines(sprintf('%.17g',oracle), file.path(out_dir,paste0(name,'.oracle')))
}

# Original PCBrainAge package weights and calculation function, unmodified.
brain <- new.env()
for (p in list.files(file.path(source_dir,'calcPCBrainAge/data'),full.names=TRUE)) load(p,brain)
sys.source(file.path(source_dir,'calcPCBrainAge/R/calcPCBrainAge.R'),brain)
sys.source(file.path(source_dir,'calcPCBrainAge/R/meanImpute.R'),brain)
rotation <- do.call(rbind,mget(c('rotation1','rotation2','rotation3'),brain))
center <- c(brain$centering1,brain$centering2)
features <- rownames(rotation)
stopifnot(identical(names(center),features))
reference <- brain$imputeMissingBrainCpGs[features]
input <- make_inputs(reference)
colnames(input) <- features
oracle <- brain$calcPCBrainAge(input)
weight <- as.numeric(rotation %*% brain$modelFit$coefs)
bias <- as.numeric(brain$modelFit$intercept - sum(center*weight))
export('pcbrainage',features,reference,weight,bias,oracle)
cat('PCBrainAge:',length(features),'methylation probes,',ncol(rotation),'PCs\n')
rm(brain,rotation,center,reference,input,weight,bias,oracle)
gc()

# Load the original Yale Box artifact, retaining only the PCGrimAge objects.
gr <- new.env()
load(file.path(source_dir,'CalcAllPCClocks.RData'),gr)
rm(list=setdiff(ls(gr),c('CalcPCGrimAge','CpGs','imputeMissingCpGs')),envir=gr)
gc()
reference <- gr$imputeMissingCpGs[gr$CpGs]
gr$datMeth <- make_inputs(reference)
colnames(gr$datMeth) <- gr$CpGs
gr$DNAmAge <- data.frame(Female=c(0,1,0,1,0,1,0,1),Age=c(20,35,50,65,80,95,45,70))
# Evaluate the exact PCA and component lines from the pinned authors' function.
code <- readLines(file.path(source_dir,'PC-Clocks/run_calcPCClocks.R'))
code <- code[grepl('^  temp <- cbind|^  DNAmAge\\$PC(PACKYRS|ADM|B2M|CystatinC|GDF15|Leptin|PAI1|TIMP1) <-',code)]
stopifnot(length(code)==9)
eval(parse(text=code),envir=gr)
clock <- gr$CalcPCGrimAge
pcs <- colnames(clock$rotation)
if (!is.null(rownames(clock$rotation))) stopifnot(identical(rownames(clock$rotation),gr$CpGs))
if (!is.null(names(clock$center))) stopifnot(identical(names(clock$center),gr$CpGs))
for (component in c('PCPACKYRS','PCADM','PCB2M','PCCystatinC','PCGDF15','PCLeptin','PCPAI1','PCTIMP1')) {
  coefs <- clock[[paste0(component,'.model')]]
  stopifnot(all(names(coefs) %in% c(pcs,'Female','Age')))
  pc_coefs <- setNames(rep(0,length(pcs)),pcs)
  selected <- intersect(names(coefs),pcs)
  pc_coefs[selected] <- coefs[selected]
  weight <- as.numeric(clock$rotation %*% pc_coefs)
  bias <- as.numeric(clock[[paste0(component,'.intercept')]] - sum(clock$center*weight))
  covariates <- intersect(c('Female','Age'),names(coefs))
  features <- c(gr$CpGs,tolower(covariates))
  weight <- c(weight,unname(coefs[covariates]))
  references <- c(reference, rep(NA_real_,length(covariates)))
  name <- paste0('pcgrimage',tolower(sub('^PC','',component)))
  export(name,features,references,weight,bias,gr$DNAmAge[[component]])
  cat(name,':',length(features),'features; covariates',paste(covariates,collapse=','),'\n')
}
