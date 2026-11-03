# DNAm median fallback for pyaging 0.6.0

Use a cohort-derived beta value for absent CpG columns when a human DNA
methylation clock has no model-supplied reference vector.

## Reference table

Compute per-CpG medians separately for the Hannum 450K cohort
([GSE40279](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE40279))
and the Biomarkers of Aging EPIC v2 cohort
([GSE246337](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE246337)).
Collapse EPIC v2 probe suffixes within each sample using pyaging's existing
aggregation before computing medians across samples. Use finite beta values
within [0, 1].

For a CpG with two cohort medians, take their arithmetic mean with equal
weight per cohort. If only one cohort has a median, use that median.
Hannum-first would discard the BoA estimate for shared probes. An array-specific
lookup would require knowing the input array. The agreed mean uses both
estimates and works without array metadata.

Store the resulting lookup separately from feature ranges. Record the input
files, checksums, sample and probe counts, aggregation, and coverage by cohort
so the lookup can be regenerated and reviewed.

## Prediction and verification

For an absent column, use the clock's existing reference vector if it has
one. Otherwise, look up the CpG in the shared table for human clocks that
accept methylation beta values, then use zero for uncovered features. Keep
model references separate from the fallback because models may use their
own references during preprocessing. Preserve the existing reporting of
absent features and the existing rejection of entirely missing input.

Verify the overlap and single-cohort rules, reference precedence, zero
fallback, and unchanged predictions for complete inputs. Compare median
fill with zero fill on held-out samples with masked probes and report
coverage and prediction changes. The reference is blood-derived, so
report results by tissue where held-out data permit.

Bump the pyaging package version to 0.6.0 as part of the implementation.
