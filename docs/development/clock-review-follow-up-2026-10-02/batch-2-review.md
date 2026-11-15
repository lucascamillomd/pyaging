# Batch 2 clock metadata review

Reviewed all 40 assigned clocks against the evidence ledger, completed methods audit, and DOI-mapped citation audit. All existing bibliographic and scientific metadata matched the evidence ledger, except the nine stale GrimAge2 component `postprocess` labels listed below. Every candidate DOI maps to the same clock name in the citation audit, and the citation count and 2026-10-02 observation date match.

## Confirmed corrections

The GrimAge2 component notebook code sets `model.postprocess_name = None`; serialized notebook/artifact metadata still says `cox_to_years`. Correct `postprocess` to `null` (identity/no transformation) for these nine outputs: `grimage2adm`, `grimage2b2m`, `grimage2cystatinc`, `grimage2gdf15`, `grimage2leptin`, `grimage2loga1c`, `grimage2logcrp`, `grimage2packyrs`, and `grimage2timp1`. Each exact notebook and code-cell pointer is recorded in the JSON. The main `grimage2` score remains `cox_to_years`; `grimage2pai1` already has no postprocess.

## Retained scientific details

- HypoClock’s underlying HypoScore is the mean beta value across 678 solo-WCGWs. Teschendorff’s 2020 paper explicitly uses `1 - HypoScore` as the anti-correlated division-burden measure, so pyaging’s `one_minus` result is intentional. The 2018 Zhou paper is a biological precursor; the named 678-site implementation is from 2020.
- Keep GrimAge v1 cited to Lu et al. (2019) and GrimAge2 cited to Lu et al. (2022). The current GrimAge2 paper reports 1,833 FHS training participants and 711 test participants, ages 40–92 (training median age 65). Component records inherit the 2022 version-2 citation.
- `training_details` in the JSON captures source-backed cohort, sample-count, age, and health-selection facts found in the evidence ledger for each assigned clock.

No unresolved fields were found. No model scores were recalculated.

Coordinator check: the raw HypoScore is inverse-oriented; its complement is positive-oriented according to the paper equation. Proxy objects also carry the stale attribute, so both the attribute and metadata key need correction.
