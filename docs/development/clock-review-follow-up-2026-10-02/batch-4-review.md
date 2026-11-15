# Batch 4 clock review

Reviewed all 40 assigned records against the current metadata, evidence ledger, methods audit, citation audit, and original-author sources. No correction was justified. The OrganAge clocks share the correct Cell Metabolism paper and DOI (128 citations in the retained 2026-10-02 snapshot); PAC maps to the correct Aging Cell paper and DOI (68 citations). The original PAC R file and the OrganAge author README were checked live.

The OrganAge identities retain the target distinction: chronological variants predict age in years; mortality variants return relative natural-log mortality hazard. The author README recommends full models with fold 1 for Olink Explore 3072, matching the pinned model provenance and input/platform metadata. PAC combines a LASSO Cox-selected protein predictor with Gompertz age calibration and returns mortality-equivalent years; it does not return an age-gap residual.

The accessible OrganAge paper summary and README do not establish the precise age interval or fitted sample count for each individual coefficient vector. The metadata's broad adult label is retained without inferring a narrower range. The full paper/supplement coefficient tables were not extracted in this bounded pass. Exact facts, source locators, and that limit are in `batch-4-review.json`.
