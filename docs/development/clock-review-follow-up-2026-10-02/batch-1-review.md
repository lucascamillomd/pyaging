# Batch 1 clock review

Reviewed all 40 assigned clocks against the current metadata, evidence ledger, methods audit, and citation audit. The supplied metadata agrees with the audited ledger for all reviewed fields. Exact-DOI citation counts and their `2026-10-02` date match the citation audit. I found no correction to make and made no numerical changes.

I checked Crossref bibliographic records live and requested the primary paper sources live. Most primary pages responded; several sources blocked direct access with HTTP 403 or 429. Those limits are recorded in the JSON. For the bioRxiv preprint, Crossref omits the container-title field, but the DOI record's title, authors, and year agree with the supplied citation and venue.

Four evidence gaps remain:

- **DepressionBarbu:** the published paper and live supplementary workbook do not expose the origin of the packaged intercept `12.2169841`. The notebook hard-codes it under a calibration-intercept comment, but does not cite its provenance. The workbook contains the CpG weights only. No numerical inference was made.
- **CpGPTGrimAge3 and CpGPTPCGrimAge3:** the preprint masks mortality-cohort identities and does not give their age range or exact array mix. Its current endpoint returned HTTP 429. The named composite methods are supported by the author tutorial/code, with the manuscript limitation retained.
- **CVDWesterman:** the publication’s final cross-study learner and pyaging’s packaged 235-feature sigmoid score differ; the source mapping between them remains undocumented.

The older evidence-ledger access issue for **Bohlin** appears stale: the current official GAprediction source exposes both `lambda.1se` and `lambda.min`, and the supplied note correctly identifies pyaging’s 251-CpG model as the latter. This batch review retains the distinction from the 96-CpG default model.

`batch-1-review.json` contains the complete per-clock retained-source entries and source-backed training details.
