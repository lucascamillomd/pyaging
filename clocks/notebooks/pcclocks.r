
library(dplyr)
library(tibble)
library(tidyr)
library(jsonlite)

load(file = "CalcAllPCClocks.RData")

print(ls(all.names = TRUE))

CalcPCGrimAge$rotation.names = colnames(CalcPCGrimAge$rotation)

CalcPCGrimAge$PCPACKYRS.model.names = names(CalcPCGrimAge$PCPACKYRS.model)
CalcPCGrimAge$PCADM.model.names = names(CalcPCGrimAge$PCADM.model)
CalcPCGrimAge$PCB2M.model.names = names(CalcPCGrimAge$PCB2M.model)
CalcPCGrimAge$PCCystatinC.model.names = names(CalcPCGrimAge$PCCystatinC.model)
CalcPCGrimAge$PCGDF15.model.names = names(CalcPCGrimAge$PCGDF15.model)
CalcPCGrimAge$PCLeptin.model.names = names(CalcPCGrimAge$PCLeptin.model)
CalcPCGrimAge$PCPAI1.model.names = names(CalcPCGrimAge$PCPAI1.model)
CalcPCGrimAge$PCTIMP1.model.names = names(CalcPCGrimAge$PCTIMP1.model)

write_json(CalcPCGrimAge, "CalcPCGrimAge.json", digits = 10)
write_json(CpGs, "PCGrimAgeCpGs.json")
