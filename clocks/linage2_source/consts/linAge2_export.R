## Load libraries
library(survival) ## For survival analysis
## ggplot2 removed  ## To make figures
    
## Clear workspace
rm(list=ls())

## Fix seed - maintain direction of PCs    
set.seed(111222333)
######################################   FUNCTION DEFINITIONS   ############################################
## FUNCTIONS START >>> 

###############################   < READ, WRITE AND CLEAN DATA >  ##########################################


markIncsFromCodeBook<-function(codeBook) {

    ## Selects all columns marked as data ("1" in Data column) in codebook for inclusion in master dataset
    incFlags<-codeBook[,"Data"]   ## This column is "1" for include in dataset, 0 for not (comments or demo data) 
    incNames<-as.character(codeBook[,1]) 
    incList<-cbind(incNames,incFlags)
    return(incList)
}


dropCols<-function(dataSet,incList) {

    ## Simply drops everything not flagged with 1 in incList (Data)
    incInds<-which(incList[,2]==1)
    incTerms<-incList[,1][incInds]
    nTerms<-length(incTerms)
    
    ## The first column is always the subject number (SEQN) - add that back
    outData<-dataSet[,1]

    for (i in 2:nTerms) {
        
        ## loop over terms that have a "1" in column 2 of the incList, find those in
        ## dataSet and include in output dataSet
        nexTerm<-incTerms[i]
        nextCol<-which(colnames(dataSet)==nexTerm)
        colData<-dataSet[,nextCol]
        outData<-cbind(outData,colData)
        colnames(outData)[i]<-nexTerm
    }

    ## Name first column appropriately and return resulting dataset
    colnames(outData)[1]<-"SEQN"
    return(outData)
}


dropNAcolumns<-function(dataSet,pNAcut,incSwitch,verbose) {
        
    ## This takes a single cutoff fraction and drops all columns (features)
    ## that contain more NAs than allowed by the cutoff.
    ## However, if force include flag is set (==1) for a column, we will force the inclusion of feature

    nRows<-dim(dataSet)[1]
    nCols<-dim(dataSet)[2]
    forceFlags<-rep(0,nCols)

    ## If incSwitch IS set, we read them from the codebook
    if (incSwitch==1) {
        ## Read include flags from codebook
        codeBookFlags<-codeBook$ForceInc       
        ## Identify column terms that we cannot drop
        nForced<-sum(codeBookFlags) 
        forceIncTerms<-codeBook$Var[codeBookFlags==1]
        ## Now identify columns in dataset that need to be retained
        for (i in 1:nForced) {
            nextTerm<-forceIncTerms[i]
            forceThis<-which(colnames(dataSet)==nextTerm)
            ## Flip the respective forceFlag to 1 - this column cannot be dropped
            nrOfNAs<-sum(is.na(dataSet[,forceThis]))
            if (verbose) {
                print(paste("Applying force flag to:",nextTerm))
                cat("\t - this will include:\t",nrOfNAs,"\t NAs in:\t",nRows," ",round(nrOfNAs/nRows*100,2),"%\n")
            }
            forceFlags[forceThis]<-1            
        }
    }
    ## Now drop all columns with too many NAs
    naColSum<-colSums(is.na(dataSet))
    naColP<-naColSum/nRows

    ## Keep only those columns (features) for which naColP (number of NAs) is smaller than pNAcut
    keepCols<-naColP<pNAcut

    ## Finally, recover all columns that we decided to force (retain) 
    ## Merge keepCols (columns that will be kept due to cutoff) and forceFlags
    keepCols<-keepCols | (forceFlags == 1)
    dataSet<-dataSet[,keepCols]

    ## Print dimension of surviving matrix and list of surviving variables
    nrows<-dim(dataSet)[1]
    ncols<-dim(dataSet)[2]
    varNames<-colnames(dataSet)
    humNames<-varNames
    
    for (i in 2:length(varNames)) {
        varName<-varNames[i]
    }
    return(dataSet)
}
    
 
qDataMatGen<-function(masterData,incList) {

    ## Loop over masterData and keep any column that has a zero in the "data" column
    allTerms<-colnames(masterData)
    nTerms<-dim(masterData)[2]   ## number of total terms (columns) in masterData
    nIncFlags<-dim(incList)[1]   ## number of terms in codebook - for which we know include flags
    
    ## The first column of the qDataMatrix has to be SEQN number - add these first
    qDataNames<-"SEQN"
    qDataMatrix <- masterData[,1]
    
    ## Loop over all terms (columns) in masterData - extract one term (column) at a time
    for (i in 2:nTerms) {    
        ## look at the next term in the data and get the respective flag from the incList
        nextTerm<-allTerms[i]

        ## Now loop over all terms in the incList and get the flag for the current term
        flag<-0   ## Graceful default is 0 - not include
        for (j in 1:nIncFlags) {
            if (incList[j,1]==nextTerm) { 
                ## Read the inc flag (second entry of that column) and return it
                flag <- incList[j,2]
            }
        }
        if (flag == 0) {
            ## If include == 0, we will include that parameter in the qDataMatrix 
            qDataColumn <- masterData[,i]    ## Keep the current column for inclusion to qDataMatrix 
            qDataMatrix<-cbind(qDataMatrix,qDataColumn)  ## Add current column to qDataMatrix 
            qDataNames<-c(qDataNames,nextTerm)  ## Also keep the current column name (nextTerm) as column name 
        }
    }
    colnames(qDataMatrix) <- qDataNames  ## Update all column names  
    return(qDataMatrix)  ## Return the matrix 
}


getNonNARows<-function(dataSet) {

    ## Identify rows that contain NAs and drop them by only retaining those that do not
    ## sums over NAs are NA so only rows with no (zero) NAs return !is.na
    keepRows<-(rowSums(is.na(dataSet))==0)
    return(keepRows)
}



#########################  < CALCULATING DERIVED FEATURES FROM DATA >  #####################################

popPCFIfs1 <- function(qDataMat) {
        ## This will calculate our frailty index / disease and comorbidity index for each subject
        ## and populate the matrix

        ## NOTE: we will allow NAs here - so check that the variables are all there
        BPQ020 <- qDataMat[,"BPQ020"]
        DIQ010<- qDataMat[,"DIQ010"]
        HUQ010 <- qDataMat[,"HUQ010"]
        HUQ020 <- qDataMat[,"HUQ020"]
        HUQ050  <- qDataMat[,"HUQ050"]
        HUQ070 <- qDataMat[,"HUQ070"]
        KIQ020  <- qDataMat[,"KIQ020"]
        MCQ010 <- qDataMat[,"MCQ010"]
        MCQ053  <- qDataMat[,"MCQ053"]
        MCQ160A <- qDataMat[,"MCQ160A"]
        MCQ160B  <- qDataMat[,"MCQ160B"]
        MCQ160C <- qDataMat[,"MCQ160C"]
        MCQ160D  <- qDataMat[,"MCQ160D"]              
        MCQ160E  <- qDataMat[,"MCQ160E"] 
        MCQ160F <- qDataMat[,"MCQ160F"]
        MCQ160G <- qDataMat[,"MCQ160G"]
        MCQ160I <- qDataMat[,"MCQ160I"]
        MCQ160J <- qDataMat[,"MCQ160J"]
        MCQ160K <- qDataMat[,"MCQ160K"]
        MCQ160L <- qDataMat[,"MCQ160L"]
        MCQ220 <- qDataMat[,"MCQ220"]
        OSQ010A <- qDataMat[,"OSQ010A"]
        OSQ010B <- qDataMat[,"OSQ010B"]
        OSQ010C <- qDataMat[,"OSQ010C"]
        OSQ060 <- qDataMat[,"OSQ060"]
        PFQ056 <- qDataMat[,"PFQ056"]

        ## Give "safe" value to all NAs ... 
        BPQ020[is.na(BPQ020)] <- 2
        DIQ010[is.na(DIQ010)] <- 2
        HUQ010[is.na(HUQ010)] <- 3
        HUQ020[is.na(HUQ020)] <- 3
        HUQ050[is.na(HUQ050)] <- 0
        HUQ070[is.na(HUQ070)] <- 2
        KIQ020[is.na(KIQ020)] <- 2 
        MCQ010[is.na(MCQ010)] <- 2
        MCQ053[is.na(MCQ053)] <- 2
        MCQ160A[is.na(MCQ160A)] <- 2
        MCQ160B[is.na(MCQ160B)]  <- 2
        MCQ160C[is.na(MCQ160C)] <-  2
        MCQ160D[is.na(MCQ160D)]  <- 2
        MCQ160E[is.na(MCQ160E)]  <- 2
        MCQ160F[is.na(MCQ160F)] <-  2
        MCQ160G[is.na(MCQ160G)] <- 2
        MCQ160I[is.na(MCQ160I)] <- 2
        MCQ160J[is.na(MCQ160J)] <- 2
        MCQ160K[is.na(MCQ160K)] <- 2
        MCQ160L[is.na(MCQ160L)] <- 2
        MCQ220[is.na(MCQ220)] <- 2
        OSQ010A[is.na(OSQ010A)] <- 2
        OSQ010B[is.na(OSQ010B)] <- 2
        OSQ010C[is.na(OSQ010C)] <- 2
        OSQ060[is.na(OSQ060)] <- 2
        PFQ056[is.na(PFQ056)] <- 2

        ## Binary yes/no decision vector 
        binVec <- cbind((BPQ020==1),((DIQ010==1) | (DIQ010==3)),(KIQ020==1),(MCQ010==1),(MCQ053==1),(MCQ160A==1),(MCQ160C==1),(MCQ160D==1),(MCQ160E==1),(MCQ160F==1),(MCQ160G==1),(MCQ160I==1),(MCQ160J==1),(MCQ160K==1),(MCQ160L==1),(MCQ220==1),(OSQ010A==1),(OSQ010B==1),(OSQ010C==1),(OSQ060==1),(PFQ056==1),(HUQ070==1))

    sumOverBinVec <- rowSums(binVec)/22 
    return(sumOverBinVec)
}


popPCFIfs2 <- function(qDataMat) {
        
    HUQ010 <- qDataMat[,"HUQ010"]
    HUQ020 <- qDataMat[,"HUQ020"]
    HUQ010[is.na(HUQ010)] <- 3
    HUQ020[is.na(HUQ020)] <- 3

    ## If sick/feeling bad, get score of 2 to 4 - if getting worse -> get 2x modifier
    ## if getting better -> 1/2 modifier 
    aVec <- ((HUQ010==4)*2+(HUQ010==5)*4)
    dVec <- (1-(HUQ020==1)*0.5+(HUQ020==2))
    fScore <- aVec*dVec
        
    return(fScore)
}


popPCFIfs3 <- function(qDataMat) {

    ## This basically codes NHANES HUQ050: "Number times received healthcare over past year"
    HUQ050  <- qDataMat[,"HUQ050"]
    HUQ050[is.na(HUQ050)] <- 0
    HUQ050[(HUQ050==77)] <- 0 ## Comment codes ("Refused")
    HUQ050[(HUQ050==99)] <- 0 ## Comment codes ("Do not know")
    return(HUQ050)
}


populateLDL <- function (dataMat,qDataMat) {                                                                           
                                                                                                                            
        ## This function will calculate LDL and adds it to the dataMatrix
        ## LDL - calculated from:
        ##    Variable: LBDTCSI	        Total Cholesterol (mmol/L)
        ##    Variable: LBDHDLSI	HDL (mmol/L)
        ##    Variable: LBDSTRSI	Triglycerides (mmol/L)
        ## Formula:  LDL-C=(TC)–(triglycerides/5)– (HDL-C). 
        ## NOTES: Can be inaccurate if triglycerides are very high (above 150 mg/dL)
        
        nSubs <- dim(dataMat)[1]
        
        ## Extract all relevant variables from data matrix 
        totCv <- dataMat[,"LBDTCSI"]
        HDLv <- dataMat[,"LBDHDLSI"]
        triGv <- dataMat[,"LBDSTRSI"]
        seqVec <- dataMat[,"SEQN"]
        LDLvec <- rep(0,nSubs)
        
        ## Loop over all subjects and update LDL
        for (i in 1:nSubs) {
                    
            totC <- totCv[i]
            HDL <- HDLv[i] 
            TG <- triGv[i] 
            LDL <- 0
            
            ## Check that we do not have any NAs here
            if (!is.na(totC)*!is.na(HDL)*!is.na(TG)) {

                ## Calculate LDL from triglycerides and total cholesterol 
                LDL <- (totC - (TG/5) - (HDL))
               
            }
            LDLvec[i] <- LDL
        }
       
        return(LDLvec)
    }

                            
#############################  < DATA SELECTION - ROWS / SUBJECTS >  ####################################

selectAgeBracket<-function(qMat,ageCutLower,ageCutUpper) {

    ## Apply a age bracket to dataset - only retain samples between upper and lower age limit
    keepRows<-((qMat[,"RIDAGEYR"]>=ageCutLower) & (qMat[,"RIDAGEYR"]<=ageCutUpper))
    return(keepRows)
}


nonAccidDeathFlags <-function(qMat) {

    ## Here we will return keep flags for all subjects who die of non-accidental deaths
    ## The cause of death (leading) is recorded (if known) in the questionnaire data matrix
    ## qDatMat in the "UCOD_LEADING" column
    ## Possible values in "UCOD_LEADING" are:
    ## 001 = Disease of the heart
    ## 002 = Malignant neoplasm
    ## 003 = Chronic lower respiratory disease
    ## 004 = Accidents and unintentional injuries
    ## 005 = Cerebrovascular disease
    ## 007 = Diabetes
    ## 008 = Influenza and pneumonia
    ## 009 = Nephritis, kidney issues
    ## 010 = All other causes (residuals)
    ## NA  = no info (the vast majority of cases)

    ## Extract cause of deaths
    causeOfDeath <- qMat[,"UCOD_LEADING"]
    ## Then drop NAs (turn into zeros) 
    causeOfDeath[is.na(causeOfDeath)]<-0
    keepFlags <- causeOfDeath!=4
    
    return(keepFlags) 
                
}


foldOutliers <- function(dataMatNorm, zScoreMax) {

    ## Fold in outlier z-scores
    
    cat("> Folding in outliers at maximum total zScore:",zScoreMax)            
    ## Now truncate / fold outliers and show boxplots    
    
    dataMatNorm_folded <- dataMatNorm    
    allTerms <- colnames(dataMatNorm)[-1]
        
    for (nextTerm in allTerms) {
        colVals <- dataMatNorm[,nextTerm]
        if(sum(is.infinite(colVals))) { print(paste("Infinite value in:",nextTerm)) } 
        ## boxplot(colVals,main=paste(nextTerm,"-before"))
        foldThese <- abs(colVals) > zScoreMax
        colVals[foldThese] <- sign(colVals[foldThese])*zScoreMax
        ## boxplot(colVals,main=paste(nextTerm,"-after"))
        ## readline()
        dataMatNorm_folded[,nextTerm] <- colVals
    }
    cat(" ... Done\n")
    return(dataMatNorm_folded)
}
    

digiCot <- function(dataMat) {

    ## Digitize continine to turn into smoking intensity
    ## Most clinics do not routinely measure cotinine - so here we will     
    ## bin cot as follows:
    ## 0  <= cot < 10 are non smokers (0)
    ## 10 >= cot < 100 are light smokers (1)
    ## 100 >= cot < 200 are moderate smokers (2)
    ## anything above 200 is a heavy smoker (3)
    
    cat("> Digitizing cotinine data ... ")                         
    cot <- dataMat[,"LBXCOT"]
    dataMat[,"LBXCOT"][cot < 10]<-0
    dataMat[,"LBXCOT"][(cot >= 10) & (cot < 100)]<- 1
    dataMat[,"LBXCOT"][(cot >= 100) & (cot < 200)]<- 2
    dataMat[,"LBXCOT"][(cot >= 200)]<- 3 
    cat("Done\n\n")
    return(dataMat)
}
    

#############################    MATH AND ANALYSIS FUNCTIONS    #######################################
normAsZscores_99_young_mf<-function(dataSet,qDataMat,dataSet_ref,qDataMat_ref) {

    ## Normalize by training set (00/99) only
    seqSel <- qDataMat_ref[,"yearsNHANES"]=="9900"
    
    ## Select age cutoff
    ageVec <- qDataMat_ref[,"RIDAGEYR"]
    ageSel <- ageVec <= 50

    ## Combine selections - all in reference data only 
    selVec <- ageSel & seqSel

    ## extract data matrix 
    dataSet_temp <- dataSet_ref[selVec,]
    ## Sex selection vector - true for males
    sexSel_temp <- qDataMat_ref[selVec,"RIAGENDR"]==1  ## Sex selection vector for 1999 to 2000 data only 
    sexSel <- qDataMat[,"RIAGENDR"]==1   ## Sex selection vector for dataset to be normalized 

    ## Normalize data by column average for each column independently
    nRows<-dim(dataSet)[1]
    nCols<-dim(dataSet)[2]

    ## Make normalized matrix by turning each value into z-score
    #dataMatN<-matrix(0,nRows,nCols)
    dataMatN<-dataSet
    ## There is certainly a more elegant way of doing this - but for hackability, lets just do a
    ## simple loop for now - column 1 is still the subject seq number - will not be normalized
    dataMatN[,1]<-dataSet[,1]
    colnames(dataMatN)<-colnames(dataSet)

    ## We will not apply normalization to some columns - fs scores in particular
    skipCols <- c("fs1Score","fs2Score","fs3Score","LBXCOT","LBDBANO")
    
    ## Loop over all columns - starting from column 2 (first data col) 
    for (col in 2:nCols) {
        
        if (sum(skipCols==colnames(dataSet)[col])==0) {

            ## Median and MAD - males 
            med_m <- median(dataSet_temp[sexSel_temp,col])
            mad_m <- mad(dataSet_temp[sexSel_temp,col])

            ## Median and MAD - males 
            med_f <- median(dataSet_temp[!sexSel_temp,col])
            mad_f <- mad(dataSet_temp[!sexSel_temp,col])

            ## Dump normalization paras
            ## cat(colnames(dataSet)[col],"Median (m/f):",med_m,med_f,"MAD (m/f):",mad_m,mad_f,"\n")
            
            ## Loop over all rows in current column and normalize each value by column mean
            for (row in 1:nRows) {

                ## Determine sex
                sexNow <- 1*sexSel[row]
                if (sexNow == 1) { # This is a male
                    #cat("SEQ",dataMat[row,"SEQN"],"is male\n")
                    mad <- mad_m
                    med <- med_m
                }
                if (sexNow == 0) { # This is a female
                    #cat("SEQ",dataMat[row,"SEQN"],"is female\n")
                    mad <- mad_f
                    med <- med_f
                }
                ## Now calculate z-score for each row of this column - use sex-specific median and MAD values
                zScore_nonN <- (dataSet[row,col]-med)/mad
                
                ## Store normalized and log2 fold changes (vs. column average) in new matrices
                #dataMatN[row,col]<-zScore
                dataMatN[row,col]<-zScore_nonN
            }
        }
    }
    return(dataMatN)
}


boxCoxTransform <- function(boxCox_lam, dataMat) { 

    ## Apply box cox transforms based on lambda given
    allTerms <- colnames(dataMat)[-1]
    cat("> Applying boxCox transformed  ... ")
    for (nextTerm in allTerms) {
        ## cat("\n",nextTerm,":")        
        ## Get column number
        dataColNr <- which(colnames(dataMat)==nextTerm) 
        lamNr <- which(colnames(boxCox_lam)==nextTerm)     
        ## Get next transformation
        nextLam <- boxCox_lam[,lamNr]
        ## Get next data item (column)
        colVals <- dataMat[,dataColNr]
        ## Selection of transformation is based on lambda value
        if (!is.na(nextLam)) { ## If NA, do nothing
            if (nextLam == 0) {
                colVals <- log(colVals)  ## If the lambda value is zero, we log the data column
                ## cat("log transformed")        
            } else {            
                colVals <- (colVals^nextLam - 1)/nextLam  ## If it is neither NA nor zero - boxCox formula for lambda
            }
        }
        dataMat[,nextTerm] <- colVals
 
    }
    cat("Done\n")
    return(dataMat)
}


projectToSVD <- function (inputMat,svdCoordMat) {

    cat("> Projecting data into PC coordinates  ... ")
    ## Project inputMat data matrix into the same PC coordinates provided by svdCoordMat
    mSamples <- dim(inputMat)[1]
    nSVs <- dim(svdCoordMat)[2]
    pcMat <- matrix(0,mSamples,nSVs)  ## Empty data matrix in PC coordinates
    ## Doing loop to calculate coordinates for samples in terms of PCs - could do matrix mult instead
    for (sample in 1:mSamples) {
        ## Current sample is current row of data (input) matrix
        curSample <- inputMat[sample,]
        
        ## Now loop over all nSVs and determine
        for (pcNr in 1:nSVs) {
            ## current PC vector is the column
            curPC <- svdCoordMat[,pcNr]
            coord <- curSample %*% curPC
            pcMat[sample,pcNr]<-coord

        }
    }
    cat("Done\n")
    return(pcMat)
}
    
    
getSurvTime <- function (qMatrix) {

    ## Function to calculate survival time between enrollment and end of follow up
    ## Get the age (in month) at time of initial screen

    ## For those individuals who died before the cutoff date in 2019, we have
    ## information on time between survey and death - for survivors, the entry is time
    ## between the initial exam and the end of the follow up

    ## NOTE: THIS IS REALLY NOT SURVIVAL TIME BUT TIME TO FOLLOW UP - interpret with eventFlags! 
    survTimes<-qMatrix[,"PERMTH_EXM"]
    return (survTimes)

    }


getEventVec <- function (qMatrix, cause) {

    ## Read qDataMatrix and determine if individual died during study period or was censored
    ## that is, survived beyond the end of the study ... 

    if (cause == 0) {  ## IF cause is 0, we do not care what people died from and report all deaths 
        eventFlags<-qMatrix[,"MORTSTAT"] 
        return (eventFlags)
    }
    if (cause !=0) {  ## If cause is > 0, we will report only specific causes of death
        eventFlags<-qMatrix[,"MORTSTAT"]
        CODFlags <- qMatrix[,"UCOD_LEADING"]
        
        if (cause == 1) { ## Heart disease deaths only
            countThese <- (CODFlags==1)
            eventFlags <- eventFlags*countThese
            return(eventFlags)
        }
        if (cause == 2) { ## Cancer deaths only
            countThese <- (CODFlags==2)
            eventFlags <- eventFlags*countThese
            return(eventFlags)
        }
        if (cause == 3) { ## COPD deaths only 
            countThese <- (CODFlags==3)
            eventFlags <- eventFlags*countThese
            return(eventFlags)
        }
        if (cause == 4) { ## Accident deaths only 
            countThese <- (CODFlags==4)
            eventFlags <- eventFlags*countThese
            return(eventFlags)
        }
        
        if (cause == 5) { ## Stroke deaths only 
            countThese <- (CODFlags==5)
            eventFlags <- eventFlags*countThese
            return(eventFlags)
        }
        if (cause == 6) { ## Deaths directly from AD only 
            countThese <- (CODFlags==6)
            eventFlags <- eventFlags*countThese
            return(eventFlags)
        }
        if (cause == 7) { ## Deaths directly form diabetes only 
            countThese <- (CODFlags==7)
            eventFlags <- eventFlags*countThese
            return(eventFlags)
        }
        if (cause == 8) { ## Deaths from influenza and pneumonia  
            countThese <- (CODFlags==8)
            eventFlags <- eventFlags*countThese
            return(eventFlags)
        }
        if (cause == 9) { ## Deaths from kidney issues  
            countThese <- (CODFlags==9)
            eventFlags <- eventFlags*countThese
            return(eventFlags)
        }
        ## We can also specify some causes that are combinations or exclusions of others
        if (cause == 10) { ## All NON CVD (not MCI, not stroke) deaths only  
            countThese <- ((CODFlags!=1) & (CODFlags!=5)) 
            eventFlags <- eventFlags*countThese
            return(eventFlags)
        }
        
        if (cause == 11) { ## All non accidental deaths only   
            countThese <- ((CODFlags!=4)) 
            eventFlags <- eventFlags*countThese
            return(eventFlags)
        }

        if (cause == 12) { ## All CVD-related deaths - including stroke   
            countThese <- ((CODFlags==1 ) | (CODFlags==5)) 
            eventFlags <- eventFlags*countThese
            return(eventFlags)
        }
    }
}


makeSurvObject <- function (qMatrix,causeOfDeath) {

    ## Take survival times and censor vector and make a survival object 

    ## First, get event flags from qMatrix - subjects that died have 1, those that survived have 0 in here
    eventFlags <- getEventVec(qMatrix,causeOfDeath)

    ## Then, get survival times (either time between exam and death (if dead) or time to end of follow up (alive)
    times <- getSurvTime(qMatrix)

    ## Now make a survival object (survival library)
    survObj<-Surv(times,eventFlags) 
    
    return(survObj) 
    
}

            
calcBioAge <- function (coxModelNew,nullModel,dataTable) {
   
    ## This will take the coxModel plus the input data table (covariates used for the cox model)
    ## it will then loop over the data table and calculate the delta ages for each individual    
    ## The cox model assumes that the hazard function hi(t) for each subject i
    ## can be broken down into log additive terms according to the linear (lm) model
    ## plus the universal time-dependent (follow up time) term  h0(t)             
    ## e.g. here: hi(t) = h0(t)*exp(beta1*startAge + beta2*x2 + beta3*x3 + beta4*x4)
    
    ## First extract maximum likelihood betas from full coxModel
    betasCOX <- coxModelNew$coefficients
    betasNull <- nullModel$coefficients
    
    ## We know that h is the mortality function according to gompertz - we can infer that:             
    ##     beta1*ageStart == ln(2)/MRDT*ageStart
    ## <=> beta1 == ln(2)/MRDT
    ## <=> MRDT == ln(2)/beta1
    ##
    betaOne <- betasNull[1]
    MRDTfit <- round(log(2)/betaOne,2)
    riskMod <- predict(coxModelNew,newdata=dataTable,type="risk")
    riskNull <- predict(nullModel,newdata=dataTable,type="risk")
    
    logRiskRatio <- log(riskMod/riskNull) 
    ageBioDelta <- logRiskRatio/log(2)*MRDTfit   
    return(ageBioDelta)           
}


drawScree <- function(fileName,svCutP,svCut,screeDat) {
    
    ## Draw scree plot (pdf)
    cat("> Writing out scree plot: [",fileName,"] ... ")
    pdf(screeFile)
    words <- paste("Scree Plot \n Cutoff:",svCutP,"% (blue line) at PC",svCut_M, "(red line)")
    plot(scree_M*100,xlab = "PC Nr." ,ylab = "Variance explained (%)", type="l", col="gray", main=words, lwd=2)
    points(screeDat*100, pch=16, col="black")
    abline(v=svCut, lwd=2, lty="dashed", col="red")
    abline(h=screeDat[svCut_M]*100, lty="dashed", lwd=2, col="blue")
    dev.off()
    cat("Done\n")
}
    

userDataOut <- function() {
    
    ## Function to return linAge2 and individual PCs for user supplied data - just a data dump ...
    pcFoldVsYoung <- 1 ## Convert PC coordinates into Z-changes vs NHANES young controls
    dropSanity <- 1 ## Remove NHANES sanity check SEQs before returning user data matrix 
    
    ## 1) Append chronAge, bioAge, sex, deltaBioAge and sex-specific PCs to user data matrix
    chronAge <- round(coxCovs_user[,"chronAge"]/12,2)
    linAge2 <- round(bioAge_user/12,2)
    bioAge_del <- round(linAge2 - chronAge,2)    
    dataMat_user <- cbind(dataMat_user,chronAge,linAge2,bioAge_del)

    ## 2) Sanity check - check that linAge2 values for reference samples are correct
    sanSam <- dataMat_user[,"SEQN"] > 100000       ## ID sanity samples in training set
    sanSEQs <- dataMat_user[sanSam,"SEQN"]-100000 
    refSEQs <- demoTrain[,"SEQN"]                   ## Get SEQs of sanity samples
    refPos <- which(!is.na(match(refSEQs,sanSEQs))) ## Match sanity sample to training data
    sanVals <- linAge2[sanSam]                      ## Get linAge2 for sanity samples from sanity run
    refVals <- (bioAge_train/12)[refPos]            ## Get linAge2 for same samples from training data
    corCof <- round(cor(refVals,sanVals),4)         ## These should be the same (correlation of 1)
   
    if (corCof != 1) {
        cat(" > Sanity check failed \n")
        return(0)

    } else {
        cat(" > Sanity check passed \n")
    }
    
    ## 3) Normalize PCs of user SEQs to those of young reference set from NHANES
    nPCs <- dim(pcDatMat)[2]  ## Use only PCs that have not been dropped by dimensionality reduction
    nPCs_used <- length(coxModelM$coefficients)

    ## Normalize by training set (00/99) only
    seqSel <- qDataMat[,"yearsNHANES"]=="9900"
    ## Select age cutoff
    ageVec <- qDataMat[,"RIDAGEYR"]
    ageSel <- ageVec <= 50
    ## Select male/female separately
    pcSex_sel  <- qDataMat[,"RIAGENDR"] == 1
    
    ## Combine selections - all in reference data only 
    selVec_M <- ageSel & seqSel & pcSex_sel
    selVec_F <- ageSel & seqSel & !pcSex_sel
    
    ## Extract data matrix
    pcMat_temp_M <- pcDatMat[selVec_M,]
    pcMat_temp_F <- pcDatMat[selVec_F,]
    
    ## Get mean PCs for male and females
    meanPCs_M <- colMeans(pcMat_temp_M)
    meanPCs_F <- colMeans(pcMat_temp_F)
    sdPCs_M <- apply(pcMat_temp_M,2,sd)
    sdPCs_F <- apply(pcMat_temp_F,2,sd)
    
    ## Select PCs from pcMatrix, normalise by mean of PC for sex
    sexSel <- coxCovs_user[,"sex_user"]
        
    ## Get Male PCs        
    PCs_M <- coxCovs_user[sexSel==1,2:(nPCs+1)]
    ## Get Female PCs
    PCs_F <- coxCovs_user[sexSel==2,2:(nPCs+1)]           

    if (pcFoldVsYoung == 1) {
        ## Normalize PCs by mean of NHANES young controls - separately for male and female SEQs    
        ## Now turn user data PCs into Z-scores relative to training data
        PCs_M <- sweep(PCs_M,2,meanPCs_M) ## Subtract mean from columns 
        PCs_M <- PCs_M %*% diag(1/sdPCs_M)
        PCs_F <- sweep(PCs_F,2,meanPCs_F) ## Subtract mean from columns 
        PCs_F <- PCs_F %*% diag(1/sdPCs_F)
        
    }
        
    ## 4) Filter PC data for male/female SEQs and drop non-model PCs
    formula_M <- as.character(formM)[3]
    formula_F <- as.character(formF)[3]

    ## Make masks - 1 if used, 0 if not included - for male and female PCs
    ## Males
    mask_M <- rep(0,nPCs)
    for (token in 2:nPCs_used) {
        pcNr <- as.numeric(strsplit(strsplit(formula_M,"+ PC")[[1]][token],"+ ")[[1]][1])
        mask_M[pcNr] <- 1 
    }
    ## Females
    nPCs_used <- length(coxModelF$coefficients)
    mask_F <- rep(0,nPCs)
    for (token in 2:nPCs_used) {
        pcNr <- as.numeric(strsplit(strsplit(formula_F,"+ PC")[[1]][token],"+ ")[[1]][1])
        mask_F[pcNr] <- 1 
    }

    ## Now drop any PCs not actually used in the model 
    PCs_M <- PCs_M[,mask_M==1]
    PCs_F <- PCs_F[,mask_F==1]
    
    ## 5) Make one data matrix combining original parameters, sex-specific PCs used by clock and LinAge2 results
    ## Make combined column names
    mPCs <- paste(colnames(pcDatMat)[mask_M==1],"M",sep="")
    fPCs <- paste(colnames(pcDatMat)[mask_F==1],"F",sep="")
    allCols <- c(colnames(dataMat_user),mPCs,fPCs)
    
    ## Add NA PCsM for females and NA PCsF for males 
    PCs_M_merge <- cbind(PCs_M,matrix(NA,dim(PCs_M)[1],dim(PCs_F)[2]))
    PCs_F_merge <- cbind(matrix(NA,dim(PCs_F)[1],dim(PCs_M)[2]),PCs_F)

    ## Combine PCs with with parameters from dataMat
    outMat_M <- cbind(dataMat_user[sexSel==1,],PCs_M_merge)
    outMat_F <- cbind(dataMat_user[sexSel==2,],PCs_F_merge)
    
    ## Combine male and female SEQs
    nSEQs <- length(sexSel)
    mParas <- length(allCols)
    outMat <- matrix(0,nSEQs,mParas)
    colnames(outMat) <- allCols 
    outMat[sexSel==1,1:mParas] <- outMat_M[,1:mParas]
    outMat[sexSel==2,1:mParas] <- outMat_F[,1:mParas]
    
    ## 5) Remove sanity data from user data matrix
    maxSEQ <- 100000 ## SEQs over 10,000 are sanity data, added to user data to check
    allSEQs <- dataMat_user[,"SEQN"] 
    keep <- allSEQs < maxSEQ
    if (dropSanity==1){
        outMat <- outMat[keep,]    
    }
    return(outMat)   
}

plotBars <- function(outMat_user,userSEQ) {

    ## Turn individual SEQs of user out matrix into bar graph 

    ## First, get sex of SEQ of interest
    rowN <- which(outMat_user[,"SEQN"]==userSEQ)
    userSex <- (!is.na(outMat_user[rowN,"PC1M"]))*1 + 2*(!is.na(outMat_user[rowN,"PC1F"]))

    ## Get boundaries for male and female PCs
    femPCstart <- which(colnames(outMat_user)=="PC1F")
    malPCstart <- which(colnames(outMat_user)=="PC1M")
    endPCs <- dim(outMat_user)[2]

    ## Get PC values 
    if (userSex == 1) { ## For male SEQ, get male PCs
        pcVals <- outMat_user[rowN,malPCstart:(femPCstart-1)] ## Model PCs
        names <- c("01_PC1M", "02_PC2M","03_PC5M","04_PC6M","05_PC8M","06_PC11M","07_PC15M","08_PC16M","09_PC17M","10_PC19M","11_PC24M","12_PC25M","13_PC27M","14_PC31M","15_PC33M","16_PC36M","17_PC42M")

        
    }
    if (userSex == 2) { ## For female SEQ, get female PCs
        pcVals <- outMat_user[rowN,femPCstart:endPCs]
        names <- c("01_PC1F", "02_PC2F","03_PC4F","04_PC6F","05_PC11F","06_PC13F","07_PC20F","08_PC22F","09_PC23F","10_PC24F","11_PC28F","12_PC31F","13_PC32F","14_PC32F","15_PC35F","16_PC38F","17_PC39F")

    }

    barObj <- data.frame(cbind(names,pcVals))

    ## Plot bar graph
    barObj$pcVals <- round(as.numeric(barObj$pcVals),2)
    return(NULL)
    bars <- ggplot(data = barObj,
                   aes(x = names, y = pcVals)) +
        geom_bar(stat = "identity",
                 fill = "lightgrey",
                 width = 0.8) +
        geom_text(aes(label = pcVals),
                  vjust = 0,
                  size = 3) +
        theme_bw() +
        theme(legend.position = "none",
              text = element_text(size = 7, family = "Arial"),
              axis.text = element_text(size = 7, family = "Arial", color = "black"),
              axis.text.x = element_text(angle = 15),
              axis.title.x = element_blank(),
              axis.title.y = element_text(size = 7, family = "Arial"),
              axis.line = element_line(color = "black"),
              panel.grid.minor = element_blank(),
              panel.border = element_blank()) +
        labs(x = "",
             y = "PC values") +
        scale_y_continuous(breaks = c(-5,-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5), limits = c(-5,5)) 

    return(bars)    
}

    
#####################################################################################################################
####                                     >>> MAIN CUSTOM CLOCK SCRIPT <<<                                        #### 
#####################################################################################################################
## Get source file name and report start

###################################
## I) DATA FILES AND DATA IMPORT ##
###################################

## NOTE ON REDUCED FEATURE SET:
## We dropped the following features from the codebook:
## 1) Fibrinogen
## 2) Gamma Glutamyl Transferase (GGT)

## We are also dropping the following features from the data matrix after they have been used to calculate LDL (below)
## 1) Total Cholesterol
## 2) Triglycerides
## 3) HDL
    
cat("\nI) Reading data and configuration files\n")
cat("#######################################\n") 
## First, read parameter file. This file contains some parameters that can be changed.
paraFile <- "paraInit.csv"
cat("> Reading parameter file: [",paraFile,"] ... ")        
paras <- read.csv(paraFile,sep=",",header=TRUE) 
cat("Done\n")

## Now use this to fix some important parameters
## See paraInit.csv for summary of what these mean
cat("> Reading parameters ... \n")        
errLevl <- paras[(paras[,"pName"]=="errorLevel"),"pValue"]   
cat("   errLevl:",errLevl,"\n") 
pNAcut <- paras[(paras[,"pName"]=="NAcutOffpercent"),"pValue"]
cat("   NAcut:",pNAcut,"\n")
svCutP <- paras[(paras[,"pName"]=="PCcutOffpercent"),"pValue"]
ageLower <- paras[(paras[,"pName"]=="lowerAgeLimit"),"pValue"]  
ageUpper <- paras[(paras[,"pName"]=="upperAgeLimit"),"pValue"]
cat("   Age limits: [",ageLower,",",ageUpper,"]","\n")            
useDerived <- paras[(paras[,"pName"]=="derivedFeatFlag"),"pValue"] 
cat("   Use Derived Features:",useDerived," ... ")
    verbose <- 0     ## Sets level of verbosity for some functions - 0 means not very
cat("Done\n\n")
    
## Second, read in csv of total NHANES (continuous) data - here we load the year 1999 with most
## lab data included
dataFileName <- "mergedDataNHANES9902.csv"
cat(paste("> Reading NHANES training data file: [",dataFileName),"] ... ")    
masterData <- read.csv(dataFileName)
cat("Done\n")

## Codebook file
codeBookFile <- "codebook_linAge2.csv"
cat("> Reading codebook file: [",codeBookFile,"] ... ")        
codeBook <- read.csv(codeBookFile)
cat("Done\n")
## The codebook file contains the Variable names in NHANES 99/00 format (Var) and human readable (Human)
## The codebook file  also contains a flag (Demo/Exam...) coding for the type of data - the flags are:
##     DEMO: Demographic data
##     Q        : Questionnaire
##     E        : Medical Examination
##     LAB      : Clinical Laboratory
##     MORTALITY: Mortality / Survival and cause of death linkage
## Finally, the codebook contains a flag indicating  numerical data (Data) and forced inclusion (ForceInc)
## These flags can be 1 (yes) or 0 (no)
## Set both Data and ForceInc to 1 for variables to be used for custom clock...

## Read in user data, append to sanity data set and run through whole script in parallel to NHANES test data    
sanityDataFile <- "userData_sanity.csv"
userDataFile <- "userData.csv"
cat("> Reading user data file: [",userDataFile,"]... ")
userDataMat <- read.csv(userDataFile)
sanityData <- read.csv(sanityDataFile)     
cat("Done\n\n")

## Digitize continine to turn into smoking intensity    
masterData <- digiCot(masterData)
sanityData <- digiCot(sanityData)
## Only do this for user data if not already done by user
digiCotFlag <- "C"
if (digiCotFlag == "C" | digiCotFlag == "c") {    
    userDataMat <- digiCot(userDataMat)
}

## Now bind sanity data to usder data
userDataMat <- rbind(userDataMat,sanityData)        


##############################
## II)  PREPARE DATA MATRIX ##
##############################
    
###############################
## II.i) FILTER INPUT MATRIX ##
###############################
cat("\nII) Selecting and cleaning data\n")
cat("###############################\n")    

cat("> Splitting data matrix ... ")
## Drop non-data columns from master data based on include flags in the codebook
cat(" selecting data ...  ")
incList <- markIncsFromCodeBook(codeBook)
dataMat <- dropCols(masterData,incList)  ## Main data matrix for clock
dataMat_user <- dropCols(userDataMat,incList)  ## User data matrix for clock

## Now we make a questionnaire Data matrix - everything OTHER than the numerical / clinical data
## qDataMat will include anything that is NOT flagged as "data" in the codebook     
cat(" selecting qData ... ")
qDataMat <- qDataMatGen(masterData,incList)   ## NOTE: This is pretty much the same as dropCols for dataMatrix ...
qDataMat_user <- qDataMatGen(userDataMat,incList)   
cat("Done\n")


######################################
## II.ii) POPULATE DERIVED FEATURES ##     
######################################
## Only gets done if useDerived == 1, skipped else
if (useDerived) {
    cat("> Populating derived features ... ")
    cat(" fs scores ...")

    ######### FS scores
    ## NHANES DATA
    fs1Score <- popPCFIfs1(qDataMat)
    fs2Score <- popPCFIfs2(qDataMat) 
    fs3Score <- popPCFIfs3(qDataMat)
    dataMat <- cbind(dataMat,fs1Score,fs2Score,fs3Score)

    ## USER DATA
    fs1Score <- popPCFIfs1(qDataMat_user)
    fs2Score <- popPCFIfs2(qDataMat_user)
    fs3Score <- popPCFIfs3(qDataMat_user)
    dataMat_user <- cbind(dataMat_user,fs1Score,fs2Score,fs3Score)

    ######### LDL scores
    ## LDL values     
    cat(" LDLV ...")
    LDLV <- populateLDL(dataMat,qDataMat)
    dataMat <- cbind(dataMat,LDLV)

    ## USER DATA
    LDLV <- populateLDL(dataMat_user,qDataMat_user)
    dataMat_user <- cbind(dataMat_user,LDLV)

    ######### Urine albumin to creatinine ratio
    ## Urine Albumin Creatinine ratio
    cat(" Albumin Creatinine ratio ... ")
    creaVals <- dataMat[,"URXUCRSI"]
    albuVals <- dataMat[,"URXUMASI"]
    crAlbRat <- albuVals/(creaVals*1.1312*10^-4)
    dataMat<-cbind(dataMat,crAlbRat)

    ## USER DATA
    creaVals <- dataMat_user[,"URXUCRSI"]
    albuVals <- dataMat_user[,"URXUMASI"]
    crAlbRat <- albuVals/(creaVals*1.1312*10^-4)
    dataMat_user<-cbind(dataMat_user,crAlbRat)
    
    cat("Done\n")
}

## Now drop all columns no longer needed - because paras were used to derive features ##     
drop<-which(colnames(dataMat)=="LBDHDLSI")    
dataMat<-dataMat[,-drop]
dataMat_user<-dataMat_user[,-drop]

drop<-which(colnames(dataMat)=="LBDSTRSI")
dataMat<-dataMat[,-drop]
dataMat_user<-dataMat_user[,-drop]

drop<-which(colnames(dataMat)=="LBDTCSI")
dataMat<-dataMat[,-drop]
dataMat_user<-dataMat_user[,-drop]
    
## We also need to drop all subjects for which we have no information on age
cat("> Removing subjects with missing age data ... ")
subSansAge <- which(is.na(qDataMat[,"RIDAGEEX"]))
dataMat <- dataMat[-subSansAge,]
qDataMat <-qDataMat[-subSansAge,]
cat("Done \n")
    
    
####################################################################
## II.iii) REFINE COHORT BY FURTHER DEMOGRAPHIC AND LAB CRITERIA  ##           
####################################################################
## Drop all accidental death cases
cat("> Removing accidental deaths ... ")
keepRows <- nonAccidDeathFlags(qDataMat) 
dataMat <- dataMat[keepRows,]
qDataMat <- qDataMat[keepRows,]
cat("Done\n")

## Remove individuals below the age of ageCut - also remove individuals over 84 
## as the age data is top-coded at 84 (e.g. 100 is recorded as 85)
cat("> Applying age filter: [",ageLower,",",ageUpper,"] years  ... ")
keepRows <- selectAgeBracket(qDataMat,ageLower,ageUpper)
dataMat <- dataMat[keepRows,]
## NOTE: Any time that we drop rows (subject), we have to also drop the same rows
## from demographic data and update the sequence data:
qDataMat <- qDataMat[keepRows,]
cat("Done\n")

## Next, we need to remove columns (features) with excessive number of missing values
cat("> NA percentage threshold for dropping feature is set to:",pNAcut*100,"%\n")
cat("> Dropping features with more NAs than threshold ... ")
dataMat<-dropNAcolumns(dataMat,pNAcut,1,verbose)
cat("Done\n")
cat("> Dropping subjects with NAs  ... ")
## Drop all subjects with missing values from the dataset
keepRows <- getNonNARows(dataMat) 
dataMat <- dataMat[keepRows,]
## Also need to again update the demographic matrix to remove the same people
qDataMat <- qDataMat[keepRows,]
cat("Done\n\n")


#########################
## III) NORMALIZATION  ##
#########################
cat("\nIII) Normalization and parameter transformation \n")
cat("#################################################\n") 

## Box-cox / log transforms of specfic features here ##  
cat("> Loading transformation options for distributions - log transforms or not only")        
boxCox_lam <- read.csv("logNoLog.csv")[2,]
cat("Done\n")

cat("> Applying transformations:\n")            
## Loop over all data items, look up appropriate transformation, then apply that        
dataMat_trans <- boxCoxTransform(boxCox_lam,dataMat)
dataMat_trans_user <- boxCoxTransform(boxCox_lam,dataMat_user)
    
        
############## Turn parameter values into Z-scores #########################
############################################################################    
## PICK NORMALIZATION OPTION HERE BY CALLING FUNCTION
cat("> Normalizing as z-score  ... by 9900 cohort young individuals ... ")
dataMatNorm <- normAsZscores_99_young_mf(dataMat_trans,qDataMat,dataMat_trans,qDataMat)  ## Male Female stratified normalizer - normalize NHANES data
dataMatNorm_user <- normAsZscores_99_young_mf(dataMat_trans_user,qDataMat_user,dataMat_trans,qDataMat)  ## Male Female stratified normalizer - normalize user data     
cat("Done\n")

    
############# Fold in Z score outliers - set max level and move anything above +/- to that limit 
################################################################################################    
zScoreMax <- 6    
cat("> Folding outliers - cutOff level: ",zScoreMax," ... \n")
cat("> NHANES data: \n")
dataMatNorm_folded <- foldOutliers(dataMatNorm,zScoreMax)
cat("> User data: \n")
dataMatUser_folded <- foldOutliers(dataMatNorm_user,zScoreMax)    

    
## Training and testing data split    
cat("> Splitting data into training (99/00 wave) and testing (01/02 wave) subsets ... ")
nCols <- dim(dataMat)[2]
inputMat <- dataMatNorm_folded[,2:nCols] ## Drop SEQn (subject number) from input data for PCA/SVD
inputMat99 <- inputMat[(qDataMat[,"yearsNHANES"]==9900),]  ## Only use 99/00 rows - all columns
inputMat01 <- inputMat[(qDataMat[,"yearsNHANES"]==102),]  ## Only use 01/02 rows - all columns
inputMat_user <- dataMatUser_folded[,2:nCols]
    
## Get sexSel vector for the 99 (training) data matrix only     
sexSel99 <- qDataMat[(qDataMat[,"yearsNHANES"]==9900),"RIAGENDR"]  
sexSel01 <- qDataMat[(qDataMat[,"yearsNHANES"]==102),"RIAGENDR"]  
sexSel_user <- qDataMat_user[,"RIAGENDR"]
cat("Done\n")


###############################################################
## IV)  DIMENSIONALITY REDUCTION / COORDINATE TRANSFORMATION ##
###############################################################
cat("\nIV) SVD and dimensionality reduction \n")
cat("########################################\n") 
#################################
## IV.i)  Do the basic PCA/SVD ##
#################################
## Now do the SVD ONLY for the 99/00 cohort - male/female separately     
inputMat99_M <- inputMat99[sexSel99==1,]
inputMat99_F <- inputMat99[sexSel99==2,]

## Now matrix to project into SVD coordinates - male/female separately     
inputMat01_M <- inputMat01[sexSel01==1,]
inputMat01_F <- inputMat01[sexSel01==2,]
   
## Now matrix to project into SVD coordinates - male/female separately     
inputMat_user_M <- inputMat_user[sexSel_user==1,]
inputMat_user_F <- inputMat_user[sexSel_user==2,]
                                    
    
## Read pre-calculated left and right singular matrices for training datset with correct directionality
cat("> Reading PC coordinate system (SVD) for 99/00 NHANES wave ... ")
## right singular vectors     
vMatDat99_F <- as.matrix(read.csv("vMatDat99_F_pre.csv"))
vMatDat99_M <- as.matrix(read.csv("vMatDat99_M_pre.csv"))

## left singular vectors    
uMatDat99_F <- as.matrix(read.csv("uMatDat99_F_pre.csv"))
uMatDat99_M <- as.matrix(read.csv("uMatDat99_M_pre.csv"))

## singular values     
diagDat99_M <- as.matrix(read.csv("diagDat99_M_pre.csv"))
diagDat99_F <- as.matrix(read.csv("diagDat99_F_pre.csv"))
cat("Done\n")
    
## uMat (left singular vector) is of dimension nrOfSamples x nrOfSVDs
mSamples99_M <- dim(uMatDat99_M)[1]
mSamples99_F <- dim(uMatDat99_F)[1]    

nSVs99_M <- dim(uMatDat99_M)[2]
nSVs99_F <- dim(uMatDat99_F)[2]
    
## Make data matrix of training set in PC coordinates derived from training data only 
cat("> Determining PC coordinates for 99/00 NHANES wave ... ")
pcMat99_M <- uMatDat99_M %*% diagDat99_M
pcMat99_F <- uMatDat99_F %*% diagDat99_F    
cat("Done\n")

## We want PCs to increase with age - if they are age-dependent. Get current direction
## for male and female PCs, then switch them (and the coordinates for all subjects) if
## the direction is negative (going down with age)     

cat("> Determining PC coordinates for 01/02 NHANES wave and user data ... ")
## pcMat99 are the SVD coordinates for the 9900 cohort - in SVD coordinates from 9900 cohort only
## pcMat01 are the SVD coordinates for the 0102 cohort - in SVD coordinates from 9900 cohort only
## Merge dataset - express BOTH 9900 and 0102 cohorts in SVD coordinates from 9900 cohort       
pcMat01_M <- projectToSVD(inputMat01_M,vMatDat99_M)
pcMat01_F <- projectToSVD(inputMat01_F,vMatDat99_F)
pcMat_user_M <- projectToSVD(inputMat_user_M,vMatDat99_M)
pcMat_user_F <- projectToSVD(inputMat_user_F,vMatDat99_F)
cat("Done\n")

    
cat("> Merging PC data for both training and testing data ... ")
## First, reconstitute the pcMat99 and pcMat01 by merging genders     
## Make dummy pcDatMat99 and pcDatMat01 - as many rows as male + female samples, cols = nSVs  
rowsAll99 <- dim(pcMat99_M)[1] + dim(pcMat99_F)[1]
rowsAll01 <- dim(pcMat01_M)[1] + dim(pcMat01_F)[1]    
colsAll <- nSVs99_M ## This should be the same for male and female     
pcMat99 <- matrix(0,nrow=rowsAll99,ncol=colsAll)
pcMat01 <- matrix(0,nrow=rowsAll01,ncol=colsAll)

## Do the same for user-provided data matrix
rowsAll_user <- dim(pcMat_user_M) + dim(pcMat_user_F)
pcMat_user <- matrix(0,nrow=rowsAll_user,ncol=colsAll)    

    
## Then sort male and female PC coordinates for 99 and 00 back into these matrices 
pcMat99[sexSel99==1,] <- pcMat99_M
pcMat99[sexSel99==2,] <- pcMat99_F     

pcMat01[sexSel01==1,] <- pcMat01_M
pcMat01[sexSel01==2,] <- pcMat01_F     

pcMat_user[sexSel_user==1,] <- pcMat_user_M
pcMat_user[sexSel_user==2,] <- pcMat_user_F     
colnames(pcMat_user) <- paste("PC",1:nSVs99_M,sep="")    
    
## Now merge pcDatMat by merging 99 and 01 matrices     
pcDatMat <- rbind(pcMat99,pcMat01)    
colnames(pcDatMat) <- paste("PC",1:nSVs99_M,sep="")
cat("Done\n")


#######################################
## IV.ii)   DIMENSIONALITY REDUCTION ##
#######################################
## scree[n] * 100 is the percent explained by the nth singular vector - use this to truncate data
## at the point where the nth SV explains less than svCutP % of total variance - that is, where:
## scree[n] becomes less than svCutP/100
cat("> Calculating scree plot  ... males  ")
scree_M <- diag(diagDat99_M)^2/sum(diag(diagDat99_M^2))
cat(" determining PCA cutoff ... males  ")
svCut_M <- which(scree_M<svCutP/100)
svCut_M <- min(svCut_M,nSVs99_M)  ## If no cutoff, use all SVs
cat("Done\n")

screeFile <- "scree_M.pdf"
drawScree(screeFile,svCutP,svCut_M,scree_M)
    
## Females    
cat("> Calculating scree plot  ... females  ")
scree_F <- diag(diagDat99_F)^2/sum(diag(diagDat99_F^2))
cat(" determining PCA cutoff ... females  ")
svCut_F <- which(scree_F<svCutP/100)
svCut_F <- min(svCut_F,nSVs99_F)  ## If no cutoff, use all SVs
cat("Done\n")
    
## Draw scree plot (pdf)
screeFile <- "scree_F.pdf"
drawScree(screeFile,svCutP,svCut_M,scree_F)

## Get consensus PC cutoff - max for male/female (if in doubt, keep)    
svCut <- max(svCut_M,svCut_F)    
   
cat("> Reducing dimensionality by dropping dimensions (PCs) explaining less than",svCutP,"% of variance. \n")
## Truncate the dataMatrix at this point - dropping all higher SVs / PCs
pcDatMat<-pcDatMat[,1:svCut[1]]
maxPC <- svCut[1]
cat("> Dropped PCs beyond PC Nr.",maxPC," ... ")
cat("Done\n")


############################
## V)  CLOCK CONSTRUCTION ##
############################
cat("\nV) Building clock based on 99/00 wave\n")
cat("#####################################\n")    

##############################################
## V.i)  EXTRACT DEMO AND SEX OF INPUT DATA ##
##############################################
## We are using 99/00 NHANES wave as training set and 01/02 as testing est
trainSam <- (qDataMat[,"yearsNHANES"]==9900)
testSam <- (qDataMat[,"yearsNHANES"]==102)

## Extract demographics (qDataMat) for training and testing set 
demoTest <- qDataMat[testSam,]
demoTrain <- qDataMat[trainSam,]

## Then get age at time of examination  - this is always the first covariate
initAgeTrain <- demoTrain[,"RIDAGEEX"] 
initAgeTest <- demoTest[,"RIDAGEEX"] 

## Extract sex flag - 1 male, 2 female for testing and training 
sexTest <- qDataMat[testSam,"RIAGENDR"]
sexTrain <- qDataMat[trainSam,"RIAGENDR"]

## Extract ID of all training set subjects
selTrain <-demoTrain[,1]    ## Here we could drop columns with NAs for instance     
selTest <- demoTest[,1]    

## Split the PCA matrix into test and train matrices 
xTrainPCA <- pcDatMat[trainSam,]  
xTestPCA <- pcDatMat[testSam,]

############################################################
## V.ii) MAKE COVARIATES FOR TRAIN/TEST COX PH MODELs     ##
############################################################
## Training
coxCovsTrain <- cbind(initAgeTrain,xTrainPCA,sexTrain)
colnames(coxCovsTrain)[1] <- "chronAge"  ## The variable names need to be fixed for the cox model function 
colnames(coxCovsTrain)[(maxPC+2)] <- "sex"
coxCovsTrain <- as.data.frame(coxCovsTrain)

## Testing
coxCovsTest <- cbind(initAgeTest,xTestPCA,sexTest) 
colnames(coxCovsTest)[1] <- "chronAge"
colnames(coxCovsTest)[(maxPC+2)] <- "sex"
coxCovsTest <- as.data.frame(coxCovsTest)

## User covariate matrix
sex_user <- qDataMat_user[,"RIAGENDR"]
initAge_user <- qDataMat_user[,"RIDAGEEX"]
coxCovs_user <- cbind(initAge_user,pcMat_user,sex_user)    
colnames(coxCovs_user)[1] <- "chronAge"
colnames(coxCovs_user)[(maxPC+2)] <- "sex"
    
## Split back into male / female to apply separate models     
coxCovs_user_M <- coxCovs_user[sex_user==1,]
coxCovs_user_F <- coxCovs_user[sex_user==2,]

    
## SPLIT INTO MALE AND FEMALE SETS ##
## Females
##########
testUseF <- demoTest[,"RIAGENDR"]==2
trainUseF <- demoTrain[,"RIAGENDR"]==2
## Female COX PH covariates
coxCovsTrainF <- coxCovsTrain[trainUseF,]
coxCovsTestF <- coxCovsTest[testUseF,]
## Female survival objects
survObjTrainF <- makeSurvObject(demoTrain,0)[(demoTrain[,"RIAGENDR"]==2)] 
    
## Female survival object for testing set
survObjTestF <- makeSurvObject(demoTest,0)[(demoTest[,"RIAGENDR"]==2)]


## Males
########
testUseM <- demoTest[,"RIAGENDR"]==1
trainUseM <- demoTrain[,"RIAGENDR"]==1
## Male COX PH covariates
coxCovsTrainM<-coxCovsTrain[trainUseM,]
coxCovsTestM<-coxCovsTest[testUseM,]
## Male survival objects
survObjTrainM <- makeSurvObject(demoTrain,0)[(demoTrain[,"RIAGENDR"]==1)] 

## Male survival object for testing set
survObjTestM <- makeSurvObject(demoTest,0)[(demoTest[,"RIAGENDR"]==1)]


#############################################
## V.iii) Use optimal model - post GLMNET  ##
#############################################
cat("> Defining models ... ")
## FINAL MALE MODEL
formM <- as.formula("survObjTrainM ~ chronAge + PC1 + PC2 + PC5 + PC6 + PC8 + PC11 + PC15 + PC16 + PC17 + PC19 + PC24 + PC25 + PC27 + PC31 + PC33 + PC36 + PC42")
## FINAL FEMALE MODEL
formF <- as.formula("survObjTrainF ~ chronAge + PC1 + PC2 + PC4 + PC6 + PC11 + PC13 + PC20 + PC22 + PC23 + PC24 + PC28 + PC31 + PC32 + PC35 + PC37 + PC38 + PC39")    
cat("Done\n")
    
## Now make coxph models for prediction
cat("> Fitting final models ... ")
## Females 
cat("Females ... ")
nullModelF <- coxph(survObjTrainF ~ chronAge, data=coxCovsTrainF)   
coxModelF <- coxph(formF, data=coxCovsTrainF)   
## Males
cat("Males ... ")
nullModelM <- coxph(survObjTrainM ~ chronAge, data=coxCovsTrainM)   
coxModelM <- coxph(formM, data=coxCovsTrainM)   
cat("Done\n")

########################################################
## V.iv) FIT FINAL MODEL - TRUNCATED AT ONLY SIG PCs  ##
########################################################
pValCoxCut <- 1  #<- no selection - keep all PCs from this step 0.05
useParF <- summary(coxModelF)$coefficients[,5]<pValCoxCut    
formF <- as.formula(paste("survObjTrainF ~",paste(factor(names(useParF)[useParF]),collapse=" + ")))
coxModelF <- coxph(formF, data=coxCovsTrainF)   
    
useParM <- summary(coxModelM)$coefficients[,5]<pValCoxCut
formM <- as.formula(paste("survObjTrainM ~",paste(factor(names(useParM)[useParM]),collapse=" + ")))
coxModelM <- coxph(formM, data=coxCovsTrainM)   

    

############################################################################
## VI) CALCULATE BIOAGES FOR TESTING AND TRAINING SET AND EVALUATE CLOCK  ##
############################################################################
cat("\nVI) Populating BioAges for male / female SEQs\n")
cat("###############################################\n")    
cat("> Calculating BioAges for test data based on LinAge2 ...")
## BioAge deltas and BioAge for testing set
cat("Females ... ")
delBioAgeTestF <- calcBioAge(coxModelF,nullModelF,coxCovsTestF)
bioAgeTestF <- coxCovsTestF[,"chronAge"] + delBioAgeTestF
cat("Males ... ")
delBioAgeTestM <- calcBioAge(coxModelM,nullModelM,coxCovsTestM)
bioAgeTestM <- coxCovsTestM[,"chronAge"] + delBioAgeTestM
cat("Done\n")

cat("> Calculating BioAges for training data based on LinAge2 ... ")
## BioAge deltas and BioAge for testing set
cat("Females ... ")
delBioAgeTrainF <- calcBioAge(coxModelF,nullModelF,coxCovsTrainF)
bioAgeTrainF <- coxCovsTrainF[,"chronAge"] + delBioAgeTrainF
cat("Males ... ")
delBioAgeTrainM <- calcBioAge(coxModelM,nullModelM,coxCovsTrainM)
bioAgeTrainM <- coxCovsTrainM[,"chronAge"] + delBioAgeTrainM
cat("Done\n")

cat("> Calculating BioAges for user data based on LinAge2 ... ")
## BioAge deltas and BioAge for user data set
cat("Females ... ")
coxCovs_user_F <- data.frame(coxCovs_user_F)
delBioAge_user_F <- calcBioAge(coxModelF,nullModelF,coxCovs_user_F)
bioAge_user_F <- coxCovs_user_F[,"chronAge"] + delBioAge_user_F

cat("Males ... ")
coxCovs_user_M <- data.frame(coxCovs_user_M)
delBioAge_user_M <- calcBioAge(coxModelM,nullModelM,coxCovs_user_M)
bioAge_user_M <- coxCovs_user_M[,"chronAge"] + delBioAge_user_M
cat("Done\n")
    
## Sort BA estimates back into the (mixed sex) testing data matrix (testing data)
cat("> Sort BA into testing matrix ... for both sexes")
nTest <- dim(demoTest)[1]
bioAge <- rep(0,nTest)
chrAge <- demoTest[,"RIDAGEEX"]


## Sort by SEQN
SEQnF <- demoTest[testUseF,"SEQN"]
SEQnM <- demoTest[testUseM,"SEQN"]
bioAge[!is.na(match(demoTest[,"SEQN"],SEQnF))] <- bioAgeTestF
bioAge[!is.na(match(demoTest[,"SEQN"],SEQnM))] <- bioAgeTestM
cat("Done\n")

## Sort BA estimates back into the (mixed sex) testing data matrix (training data)
cat("> Sort BA into training matrix ... for both sexes")
nTrain <- dim(demoTrain)[1]
bioAge_train <- rep(0,nTrain)
chrAge_train <- demoTrain[,"RIDAGEEX"]


## Sort by SEQN
SEQnF_train <- demoTrain[trainUseF,"SEQN"]
SEQnM_train <- demoTrain[trainUseM,"SEQN"]
bioAge_train[!is.na(match(demoTrain[,"SEQN"],SEQnF_train))] <- bioAgeTrainF
bioAge_train[!is.na(match(demoTrain[,"SEQN"],SEQnM_train))] <- bioAgeTrainM
cat("Done\n")

## Sort user data by SEQN
cat("> Sort BA into user matrix ... for both sexes ... ")
bioAge_user <- rep(0,dim(userDataMat)[1])
SEQnF_user <- qDataMat_user[qDataMat_user[,"RIAGENDR"]==2,"SEQN"]
SEQnM_user <- qDataMat_user[qDataMat_user[,"RIAGENDR"]==1,"SEQN"]
cat("Done\n")
    
cat("> Adding PCs and LinAge2 data to user data matrix ... ")        
bioAge_user[!is.na(match(qDataMat_user[,"SEQN"],SEQnF_user))] <- bioAge_user_F
bioAge_user[!is.na(match(qDataMat_user[,"SEQN"],SEQnM_user))] <- bioAge_user_M
outMat <- userDataOut()
cat("Done\n")
    
## All done
cat("#################################################################################\n") 

## Have a quick look at the output data and investigate individual SEQ
userSEQs <- outMat[,"SEQN"]
cat("> Data for SEQs:\n\n  ")
cat(userSEQs)
cat("\n\n  added to the data matrix\n")

## Write out the user data matrix with added bioAge and sex-specific PC data
cat("> Writing updated user data matrix ... <userData_out.csv>\n")
write.csv(outMat,"userData_out.csv")
cat("Done\n")



cat("<<< \n\n")

############ DIAGNOSTIC DUMP ############
sink("diag.txt")
cat("== dataMat colnames (NHANES, after all drops) ==\n"); print(colnames(dataMat))
cat("ncol dataMat:", dim(dataMat)[2], " ncol dataMat_user:", dim(dataMat_user)[2], "\n")
cat("== dataMat_user colnames ==\n"); print(colnames(dataMat_user))
cat("identical colnames:", identical(colnames(dataMat), colnames(dataMat_user)), "\n")
cat("== svCut_M:", svCut_M, " svCut_F:", svCut_F, " maxPC:", maxPC, "\n")
cat("== raw dataMat_user ==\n"); print(dataMat_user[,1:min(60,ncol(dataMat_user))])
cat("== dataMat_trans_user (post log) ==\n"); print(dataMat_trans_user)
cat("== dataMatNorm_user (post z) ==\n"); print(dataMatNorm_user)
cat("== dataMatUser_folded ==\n"); print(dataMatUser_folded)
cat("== pcMat_user (first 2 rows, all PCs) ==\n"); print(pcMat_user[1:2,])
cat("== coxModelM coefficients ==\n"); print(coxModelM$coefficients)
cat("== coxModelF coefficients ==\n"); print(coxModelF$coefficients)
cat("== nullModelM coef ==\n"); print(nullModelM$coefficients)
cat("== nullModelF coef ==\n"); print(nullModelF$coefficients)
cat("== MRDT M:", log(2)/nullModelM$coefficients[1], " F:", log(2)/nullModelF$coefficients[1], "\n")
cat("== coxModelM means ==\n"); print(coxModelM$means)
cat("== coxModelF means ==\n"); print(coxModelF$means)
cat("== nullModelM means ==\n"); print(nullModelM$means)
cat("== nullModelF means ==\n"); print(nullModelF$means)
cat("== user bioAge (months) ==\n"); print(bioAge_user[1:2])
cat("== user chronAge months ==\n"); print(coxCovs_user[1:2,"chronAge"])
cat("== linAge2 years ==\n"); print(round(bioAge_user[1:2]/12,2))
cat("== formM ==\n"); print(formM); cat("== formF ==\n"); print(formF)
# normalization constants
seqSel <- qDataMat[,"yearsNHANES"]=="9900"; ageSel <- qDataMat[,"RIDAGEYR"]<=50; selVec<-ageSel&seqSel
dst <- dataMat_trans[selVec,]; ss <- qDataMat[selVec,"RIAGENDR"]==1
cat("== median/MAD reference (n_M=",sum(ss)," n_F=",sum(!ss),") ==\n")
for (col in 2:ncol(dataMat_trans)) cat(colnames(dataMat_trans)[col], median(dst[ss,col]), mad(dst[ss,col]), median(dst[!ss,col]), mad(dst[!ss,col]), "\n")
cat("== PC mean/sd young ref ==\n")
pcs<-qDataMat[,"RIAGENDR"]==1
cat("nM",sum(selVec&pcs),"nF",sum(selVec&!pcs),"\n")
print(rbind(colMeans(pcDatMat[selVec&pcs,]),apply(pcDatMat[selVec&pcs,],2,sd),colMeans(pcDatMat[selVec&!pcs,]),apply(pcDatMat[selVec&!pcs,],2,sd)))
sink()

dir.create("consts",showWarnings=FALSE)
feat <- colnames(dataMat)[-1]
write.csv(data.frame(feature=feat, lam=as.numeric(boxCox_lam[1,match(feat,colnames(boxCox_lam))])),"consts/features.csv",row.names=FALSE)
seqSel <- qDataMat[,"yearsNHANES"]=="9900"; ageSel <- qDataMat[,"RIDAGEYR"]<=50; selVec<-ageSel&seqSel
dst <- dataMat_trans[selVec,]; ss <- qDataMat[selVec,"RIAGENDR"]==1
mm<-data.frame(feature=feat,
  med_m=sapply(2:ncol(dataMat_trans),function(c) median(dst[ss,c])),
  mad_m=sapply(2:ncol(dataMat_trans),function(c) mad(dst[ss,c])),
  med_f=sapply(2:ncol(dataMat_trans),function(c) median(dst[!ss,c])),
  mad_f=sapply(2:ncol(dataMat_trans),function(c) mad(dst[!ss,c])),
  skip=feat %in% c("fs1Score","fs2Score","fs3Score","LBXCOT","LBDBANO"))
write.csv(mm,"consts/normstats.csv",row.names=FALSE)
write.csv(vMatDat99_M,"consts/vM.csv",row.names=FALSE)
write.csv(vMatDat99_F,"consts/vF.csv",row.names=FALSE)
write.csv(data.frame(name=names(coxModelM$coefficients),beta=as.numeric(coxModelM$coefficients),mean=as.numeric(coxModelM$means)),"consts/coxM.csv",row.names=FALSE)
write.csv(data.frame(name=names(coxModelF$coefficients),beta=as.numeric(coxModelF$coefficients),mean=as.numeric(coxModelF$means)),"consts/coxF.csv",row.names=FALSE)
write.csv(data.frame(sex=c("M","F"),beta=c(nullModelM$coefficients[1],nullModelF$coefficients[1]),mean=c(nullModelM$means[1],nullModelF$means[1])),"consts/coxnull.csv",row.names=FALSE)
write.csv(pcMat_user[1:2,],"consts/pc_user.csv",row.names=FALSE)
write.csv(dataMatUser_folded[1:2,],"consts/folded_user.csv",row.names=FALSE)
