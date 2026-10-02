# 0.5.4 source and validation record

This release adds 104 scalar predictors to the existing 177-entry catalog. The
source implementations and numerical oracles come from the original authors.
Biolearn and methylCIPHER are secondary comparisons only.

| Family | Added entries | Original implementation |
|---|---:|---|
| OrganAge | 90 | `ludgergoeminne/organAging`, commit `5147b0301ec7f4abdb10ef650d04f47454ddc8fd`; author-recommended fold-one CSVs |
| PCGrimAge components | 8 | `MorganLevineLab/PC-Clocks`, commit `5e65bce1eeb600949fce0853e46ed2aa6210051d`; author-distributed `CalcAllPCClocks.RData` |
| PCBrainAge | 1 | `MorganLevineLab/calcPCBrainAge`, commit `8a4a0bf93fe715e48d495af46edf17f10d988afa` |
| CellDRIFT | 1 | `MorganLevineLab/CellDRIFT`, commit `066b3e816795f3c0b372ff8d33f59185a490bd8b` |
| MiAge | 1 | Authors' `function_library.r` and `site_specific_parameters.Rdata`, preserved in an archived copy; see the replication fixture provenance |
| PhenoAge partitions | 2 | `MorganLevineLab/prcPhenoAge`, commit `0af213c60035d8dd568d9d9ba2df90cfbca19596` |
| IntrinClock370 | 1 | Original Zenodo `10426597`, `final_model_small.RData` at `lambda.1se`, with the age transform from `demo.R` |

The four `clocks/build_*.py` scripts provide reproducible conversion. Source
digests, native R reference generators, synthetic fixtures, and numerical
comparisons are under `tests/data/{organage,pc_extensions,replication,author_extensions}`.
The temporary reference runtime was R 4.5.3 with glmnet 5.1. It is a validation
tool only; pyaging does not require R at runtime.

## Numerical representation

PCBrainAge, PCGrimAge components, and CellDRIFT compose the original affine
PCA/regression stages into double-precision linear coefficients. This avoids
publishing repeated large rotation matrices. The oracle is evaluated through
the original staged R equations, independently of the composed Python weights.
PCBrainAge preserves all 357,852 probes, including 1,483 non-CpG `ch.*` probes.

MiAge preserves the original five starting values, bounds, objective and
analytic gradient using SciPy's L-BFGS-B solver on the CPU. Its output scale is
relative mitotic age, not independently calibrated years or an absolute count of
cell divisions. This implementation intentionally retains the author's
all-missing tie behavior; such a result is not a biological estimate.

PhenoAge partitions have zero intercept. IntrinClock370 extracts the original
370 nonzero coefficients at the exact saved `lambda.1se` value; the existing
380-CpG implementation is unchanged. OrganAge accepts original protein symbols
and normalized Olink NPX without additional cohort standardization.

## Validation and publication

Tests exercise author-reference predictions, feature ordering, omitted columns,
supplied NaNs, batches, serialization, and the public prediction pipeline.
The complete local default test suite passed 607 tests; 81 tests were skipped
because older optional artifacts or CUDA were unavailable. All 104 new artifacts
passed notebook, metadata, and weight consistency validation. The 177 existing
registry entries and aggregate records were verified unchanged.

Expected predictions were generated in native R; the input fixtures are
synthetic and contain no patient-level data. The metadata ledger identifies
the evidence for every new catalog field.

OrganAge's 1,080 reference predictions match within `1e-10`; IntrinClock370's
five cases differ by at most `8.53e-14` years. PhenoAge-partition differences are
below `6e-13`, and PCBrainAge's eight cases differ by at most `2.79e-12` years.
Across eight cases for each PCGrimAge component, protein-proxy errors are at
most `1.26e-8` picograms per milliliter; PackYrs differs by at most `1.61e-12`
pack-years. All selected demographic covariates are required and have no
reference defaults: B2M and GDF15 require age, PAI1 requires female, and the
other five components require both age and female.
CellDRIFT differs by less than `4e-13` population doublings and MiAge by at most
`6.10e-5` on the tested relative-division scale. Family-specific fixture
documentation records the comparisons and tolerances. CUDA execution cannot be exercised on the macOS validation host.

The secondary comparison found identical IntrinClock370 and PhenoAge-partition
coefficients in methylCIPHERv2. All 90 OrganAge models' nonzero terms are present
in Biolearn, with coefficient differences below `5e-11` due to decimal rounding.
Neither comparison supplied coefficients or oracle predictions to the builds.

New Hugging Face artifacts include source provenance and any supplied author
license. OrganAge retains its academic noncommercial restriction. Missing
upstream license declarations are left unspecified rather than replaced by the
pyaging software's MIT license. Existing model files and their runtime versions
are preserved when the aggregate catalog is extended.
