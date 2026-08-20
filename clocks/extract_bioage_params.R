#!/usr/bin/env Rscript
# Extract KDM, HD, and refit-PhenoAge parameters from dayoonkwon/BioAge.
#
# All three fits are trained on SI-unit variants of the NHANES columns so the
# parameters land natively in pyaging's unit convention and need no post-hoc
# coefficient conversion. KDM, HD, and PhenoAge are all scale-invariant in their
# inputs (KDM: (x-q)*k/s^2 with q,k,s all scaling with x; HD: z-scored against
# the reference; PhenoAge: gompertz coefficient scales as 1/c), so fitting in SI
# units reproduces BioAge's published numbers exactly rather than approximating
# them.
#
# Unit notes established empirically against the shipped NHANES data:
#   * NHANES3$fev is millilitres (median 2885); NHANES3$fev_1000 is litres
#     (median 2.885). We use fev_1000.
#   * lncrp is log1p(crp in mg/dL), NOT log(crp): exp(lncrp) - crp == 1 exactly
#     across both cohorts.
#
# CRP naming. Internally the fitted column stays `log_crp`, because that is what
# the value is: log1p(CRP in mg/dL), the scale every fit below is trained on. The
# name EMITTED to the JSON is `c_reactive_protein`, because that is what a pyaging
# user supplies: the raw measurement in mg/dL, which each clock log1p's itself in
# preprocess(). The rename is therefore name-only for the parameters — no q, k, s,
# coefficient, mean, sd, or covariance moves — while the reference rows carry the
# raw `crp` column so that feeding them in and letting the clock transform
# reproduces BioAge's own output.
#   * albumin_gL == albumin * 10, glucose_mmol == glucose * 0.0555, and
#     creat_umol == creat * 88.4017 (= 10000 / 113.12) are all exact, with zero
#     deviation across every non-missing row of both cohorts.

suppressPackageStartupMessages({
  library(BioAge)
  library(dplyr)
  library(jsonlite)
})

out_dir <- "clocks/bioage_params"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Conventional -> SI conversion factors (verified against the BioAge columns
# above; totchol and bun have no SI column shipped so we convert them here).
TOTCHOL_MGDL_TO_MMOL <- 0.02586 # cholesterol, 1 / 38.67
BUN_MGDL_TO_MMOL <- 0.357 # urea nitrogen, 1 / 2.8
ALBUMIN_GDL_TO_GL <- 10
CREAT_MGDL_TO_UMOL <- 88.4 # documentation window only; exact factor is 88.4017
GLUCOSE_MGDL_TO_MMOL <- 0.0555 # 1 / 18.0182

# Rename BioAge columns to pyaging feature names, converting units where needed.
# NOTE: this overwrites `albumin`, `creat`, `glucose` etc., so any filtering that
# depends on the conventional-unit columns must happen BEFORE this is applied.
to_pyaging_units <- function(data) {
  data %>% mutate(
    albumin = albumin_gL,
    creatinine = creat_umol,
    glucose = glucose_mmol,
    log_crp = lncrp,
    c_reactive_protein = crp,
    lymphocyte_percent = lymph,
    mean_cell_volume = mcv,
    red_cell_distribution_width = rdw,
    alkaline_phosphatase = alp,
    white_blood_cell_count = wbc,
    total_cholesterol = totchol * TOTCHOL_MGDL_TO_MMOL,
    blood_urea_nitrogen = bun * BUN_MGDL_TO_MMOL,
    hemoglobin_a1c = hba1c,
    systolic_blood_pressure = sbp,
    forced_expiratory_volume = fev_1000,
    female = as.numeric(gender == 2)
  )
}

nhanes3 <- to_pyaging_units(NHANES3)
nhanes4 <- to_pyaging_units(NHANES4)

# The internal fitted column name -> the feature name a pyaging user supplies.
# See the CRP naming note in the header.
to_feature_names <- function(names) {
  replace(names, names == "log_crp", "c_reactive_protein")
}

# ---- KDM -------------------------------------------------------------------
kdm_markers <- c(
  "forced_expiratory_volume", "systolic_blood_pressure", "total_cholesterol",
  "hemoglobin_a1c", "albumin", "creatinine", "log_crp",
  "alkaline_phosphatase", "blood_urea_nitrogen"
)

# Same training window as BioAge::kdm_nhanes().
kdm_train_for <- function(female_value) {
  kdm_calc(
    nhanes3 %>% filter(age >= 30, age <= 75, pregnant == 0, female == female_value),
    biomarkers = kdm_markers, fit = NULL, s_ba2 = NULL
  )
}

kdm_train <- list("0" = kdm_train_for(0), "1" = kdm_train_for(1))

kdm_params_for <- function(train) {
  agev <- train$fit$lm_age
  stopifnot(identical(agev$bm, kdm_markers))
  list(
    biomarkers = to_feature_names(agev$bm),
    q = as.numeric(agev$q),
    k = as.numeric(agev$k),
    s = as.numeric(agev$s),
    s_ba2 = as.numeric(train$fit$s_ba2)
  )
}

write_json(
  list(
    features = to_feature_names(c(kdm_markers, "age", "female")),
    male = kdm_params_for(kdm_train[["0"]]),
    female = kdm_params_for(kdm_train[["1"]])
  ),
  file.path(out_dir, "kdmage.json"),
  digits = 12, auto_unbox = TRUE, pretty = TRUE
)

# ---- Homeostatic dysregulation --------------------------------------------
hd_markers <- c(
  "albumin", "lymphocyte_percent", "mean_cell_volume", "glucose",
  "red_cell_distribution_width", "creatinine", "log_crp",
  "alkaline_phosphatase", "white_blood_cell_count"
)

# Clinically acceptable reference windows. The thresholds are applied to the
# CONVENTIONAL-unit columns exactly as BioAge::hd_nhanes() does, and the NA mask
# is propagated to the SI column, so the reference cohort is bit-identical to
# BioAge's. Applying rounded SI thresholds instead would silently shift cohort
# membership at the boundary (e.g. creat 0.6 mg/dL -> 53.04, not 53, umol/L).
hd_reference_all <- NHANES3 %>%
  filter(age >= 20, age <= 30, pregnant == 0, bmi < 30) %>%
  mutate(
    albumin_gL = ifelse(albumin >= 3.5 & albumin <= 5, albumin_gL, NA),
    alp = ifelse(gender == 2,
      ifelse(alp >= 37 & alp <= 98, alp, NA),
      ifelse(alp >= 45 & alp <= 115, alp, NA)
    ),
    creat_umol = ifelse(gender == 2,
      ifelse(creat >= 0.6 & creat <= 1.1, creat_umol, NA),
      ifelse(creat >= 0.8 & creat <= 1.3, creat_umol, NA)
    ),
    glucose_mmol = ifelse(glucose >= 60 & glucose <= 100, glucose_mmol, NA),
    mcv = ifelse(gender == 2,
      ifelse(mcv >= 78 & mcv <= 101, mcv, NA),
      ifelse(mcv >= 82 & mcv <= 102, mcv, NA)
    ),
    rdw = ifelse(rdw >= 11.5 & rdw <= 14.5, rdw, NA),
    # BioAge filters raw crp < 2 mg/dL; because lncrp == log1p(crp) this is
    # log_crp < log(3), NOT exp(log_crp) < 2.
    lncrp = ifelse(crp < 2, lncrp, NA),
    lymph = ifelse(lymph >= 20 & lymph <= 40, lymph, NA),
    wbc = ifelse(wbc >= 4.5 & wbc <= 11, wbc, NA)
  ) %>%
  to_pyaging_units()

# Documentation only: the same windows expressed in pyaging (SI) units.
hd_reference_window <- function(female_value) {
  list(
    albumin = c(3.5, 5) * ALBUMIN_GDL_TO_GL,
    lymphocyte_percent = c(20, 40),
    mean_cell_volume = if (female_value == 1) c(78, 101) else c(82, 102),
    glucose = c(60, 100) * GLUCOSE_MGDL_TO_MMOL,
    red_cell_distribution_width = c(11.5, 14.5),
    creatinine = if (female_value == 1) {
      c(0.6, 1.1) * CREAT_MGDL_TO_UMOL
    } else {
      c(0.8, 1.3) * CREAT_MGDL_TO_UMOL
    },
    # Expressed in raw mg/dL like the user-facing feature, not on the log1p scale
    # the fit uses: log1p(0) = 0 and log1p(2) = log(3), so this is the exact
    # preimage of the old c(0, log(3)).
    c_reactive_protein = c(0, 2),
    alkaline_phosphatase = if (female_value == 1) c(37, 98) else c(45, 115),
    white_blood_cell_count = c(4.5, 11)
  )
}

# Faithful re-implementation of hd_calc()'s math. Needed because hd_calc returns
# only the cohort-normalized hd/hd_log, never the raw Mahalanobis distance, and
# we must bake the normalizing constant sd(log(distance)) into the exported
# parameters for single-sample prediction to be reproducible.
hd_fit_for <- function(female_value) {
  reference <- hd_reference_all %>% filter(female == female_value)
  ref_matrix <- as.matrix(reference[, hd_markers])
  ref_mean <- colMeans(ref_matrix, na.rm = TRUE)
  ref_sd <- apply(ref_matrix, 2, sd, na.rm = TRUE)

  standardize <- function(matrix_in) {
    sweep(sweep(matrix_in, 2, ref_mean, "-"), 2, ref_sd, "/")
  }
  ref_z <- na.omit(standardize(ref_matrix))
  center <- colMeans(ref_z)
  covariance <- var(ref_z)

  projection <- na.omit(standardize(as.matrix(
    (nhanes4 %>% filter(female == female_value))[, hd_markers]
  )))
  distances <- sqrt(mahalanobis(projection, center, covariance))

  list(
    biomarkers = to_feature_names(hd_markers),
    reference_mean = unname(ref_mean),
    reference_sd = unname(ref_sd),
    reference_window = hd_reference_window(female_value),
    reference_n = nrow(ref_z),
    standardized_center = unname(center),
    standardized_covariance = unname(as.matrix(covariance)),
    log_hd_sd = sd(log(distances), na.rm = TRUE)
  )
}

hd_fit <- list("0" = hd_fit_for(0), "1" = hd_fit_for(1))

write_json(
  list(
    features = to_feature_names(c(hd_markers, "female")),
    male = hd_fit[["0"]], female = hd_fit[["1"]]
  ),
  file.path(out_dir, "homeostaticdysregulation.json"),
  digits = 12, auto_unbox = TRUE, pretty = TRUE
)

# ---- PhenoAge refit without creatinine, albumin, alkaline phosphatase ------
sp_markers <- c(
  "glucose", "log_crp", "lymphocyte_percent",
  "mean_cell_volume", "red_cell_distribution_width", "white_blood_cell_count"
)

sp_train <- phenoage_calc(
  nhanes3 %>% filter(age >= 20, age <= 84),
  biomarkers = sp_markers, fit = NULL
)

# flexsurvreg's gompertz coefficient table is rownamed
# c("shape", "rate", <covariates...>); "rate" is the linear-predictor intercept.
sp_coef <- sp_train$fit$coef
stopifnot(identical(rownames(sp_coef), c("shape", "rate", sp_markers, "age")))

write_json(
  list(
    features = to_feature_names(c(sp_markers, "age")),
    coefficients = as.numeric(sp_coef[c(sp_markers, "age"), "coef"]),
    intercept = as.numeric(sp_coef["rate", "coef"]),
    m_n = as.numeric(sp_train$fit$m_n),
    m_d = as.numeric(sp_train$fit$m_d),
    ba_n = as.numeric(sp_train$fit$BA_n),
    ba_d = as.numeric(sp_train$fit$BA_d),
    ba_i = as.numeric(sp_train$fit$BA_i)
  ),
  file.path(out_dir, "phenoagesaopaulo.json"),
  digits = 12, auto_unbox = TRUE, pretty = TRUE
)

# ---- Reference predictions for downstream parity tests ---------------------
# 20 fixed NHANES IV subjects: complete cases across the union of every
# biomarker used by the three clocks, sorted by sampleID (C-locale byte order),
# first 20. `expected` values come from BioAge's own functions, never from a
# re-implementation.
ref_features <- c(unique(c(kdm_markers, hd_markers, sp_markers)), "age", "female")

kdm_projected <- bind_rows(lapply(c(0, 1), function(female_value) {
  kdm_calc(
    nhanes4 %>% filter(female == female_value),
    biomarkers = kdm_markers,
    fit = kdm_train[[as.character(female_value)]]$fit,
    s_ba2 = kdm_train[[as.character(female_value)]]$fit$s_ba2
  )$data %>% select(sampleID, kdm)
}))

hd_projected <- bind_rows(lapply(c(0, 1), function(female_value) {
  hd_calc(
    data = nhanes4 %>% filter(female == female_value),
    reference = hd_reference_all %>% filter(female == female_value),
    biomarkers = hd_markers
  )$data %>% select(sampleID, hd_log)
}))

sp_projected <- phenoage_calc(
  nhanes4 %>% filter(age >= 20),
  biomarkers = sp_markers, fit = sp_train$fit
)$data %>% select(sampleID, phenoage)

# Sanity check: our re-derived standardized distances must reproduce BioAge's
# own hd_log for the same cohort, given log_hd_sd as the normalizing constant.
for (female_value in c(0, 1)) {
  fit <- hd_fit[[as.character(female_value)]]
  cohort <- nhanes4 %>% filter(female == female_value)
  z <- sweep(sweep(as.matrix(cohort[, hd_markers]), 2, fit$reference_mean, "-"),
    2, fit$reference_sd, "/")
  keep <- stats::complete.cases(z)
  own <- log(sqrt(mahalanobis(
    z[keep, ], fit$standardized_center, fit$standardized_covariance
  ))) / fit$log_hd_sd
  theirs <- hd_calc(
    data = cohort,
    reference = hd_reference_all %>% filter(female == female_value),
    biomarkers = hd_markers
  )$data$hd_log
  theirs <- theirs[!is.na(theirs)]
  cat(sprintf(
    "hd_log parity (female=%d): n=%d max|diff|=%.3g\n",
    female_value, length(own), max(abs(own - theirs))
  ))
  stopifnot(max(abs(own - theirs)) < 1e-10)
}

# Subject selection runs over the fitted columns, exactly as before, so that
# carrying the raw CRP column cannot shift cohort membership and move `expected`.
# The raw column is joined back only afterwards, for emission.
selected <- nhanes4 %>%
  select(sampleID, all_of(ref_features)) %>%
  filter(stats::complete.cases(.)) %>%
  arrange(sampleID) %>%
  head(20) %>%
  left_join(nhanes4 %>% select(sampleID, c_reactive_protein), by = "sampleID") %>%
  left_join(kdm_projected, by = "sampleID") %>%
  left_join(hd_projected, by = "sampleID") %>%
  left_join(sp_projected, by = "sampleID")

stopifnot(nrow(selected) == 20, !anyNA(selected))

# The emitted rows carry raw CRP where the fits carry log1p(CRP); the clocks close
# that gap in preprocess(). If this ever fails, the two have drifted apart.
stopifnot(max(abs(log1p(selected$c_reactive_protein) - selected$log_crp)) < 1e-12)

emit_features <- to_feature_names(ref_features)

write_json(
  list(
    features = emit_features,
    sample_ids = selected$sampleID,
    rows = selected %>% select(all_of(emit_features)),
    expected = list(
      kdmage = selected$kdm,
      homeostaticdysregulation = selected$hd_log,
      phenoagesaopaulo = selected$phenoage
    )
  ),
  file.path(out_dir, "reference_predictions.json"),
  digits = 12, auto_unbox = TRUE, pretty = TRUE
)

cat("wrote parameters to", out_dir, "\n")
