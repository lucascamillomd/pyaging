# Package audit for 0.5.3

Audit date: 2026-10-02. Baseline: `e7267e9`, package version 0.5.2.

## Scope and release boundary

The audit covers prediction, preprocessing, model runtime code, downloads, metadata
helpers, logging, packaging, tests, documentation, and release scripts. The existing
clock coefficients, serialized models, feature-range registry, provenance ledger,
and Hugging Face files are retained. Clock-building notebooks are inspected but
are not re-executed to generate new artifacts.

This is a software audit, not a new scientific validation of every clock against
its source publication. Full-catalog tests require generated weights that are not
part of the Git checkout. Their skips and other validation limits are recorded below.

## Findings and changes

| Area | Reproduced problem | Change |
| --- | --- | --- |
| DataFrame conversion | Numeric IDs lose metadata after AnnData stringifies them; hierarchical IDs fail conversion | Convert sample IDs consistently and preserve metadata order |
| Array ownership | Returned matrices can be read-only or share storage with inputs and snapshots | Create owned arrays and independent original/imputed snapshots |
| Input validation | Invalid strategies pass on complete data; duplicate labels can remove valid columns | Validate strategy and feature names before transformation |
| EPIC v2 | Singleton suffixes prevent feature matching; duplicate labels overweight probes | Normalize all probe names and aggregate positions with grouped means |
| bigWig | Repeated array appends copy accumulated results for every gene | Preallocate output; reuse genomic region tuples |
| Gene metadata | Numeric-only chromosome CSVs lose all autosomes | Parse chromosome identifiers as strings and use vectorized filtering |
| tAge | Missing counts become zeros during grouping; invalid reference flags select samples | Validate finite nonnegative counts and boolean/0-or-1 flags |
| tAge reference rows | Repeated sample labels select extra reference samples | Select references with positional masks |
| Feature matching | Per-clock dictionaries rebuild the index for large matrices | Reuse the pandas index lookup engine |
| Sparse prediction | Sparse slices cannot be assigned directly to dense aligned matrices | Densify only the selected feature slice |
| Batch ownership | In-place model transforms alter retained inputs | Give every batch owned tensor storage |
| Rank clocks | Missing-value medians depend on batch size | Compute the existing cohort-wide lower median once and pass it to each batch |
| Rank computation | Per-feature Python loops synchronize CUDA tensors repeatedly | Use tensor operations for tied ranks and share the implementation |
| PCGrimAge | Forward bypasses its reference-value imputer | Apply the existing preprocessing before its PCA |
| Mitotic clocks | All-missing float64 input creates float32 fallback and fails matrix multiplication | Create NaN fallback on the input dtype and device |
| PhenoAge links | Intermediate mortality rounds to 0 or 1, yielding artificial infinities | Cancel the nested exp/log terms algebraically and evaluate in log space |
| Device memory | CuPy availability causes full-cohort GPU allocation | Keep aligned inputs in host memory and transfer batches |
| Loading across datasets | Every prediction repeats Hub resolution and model preparation | Add explicit bounded LRU `ClockCache` with revision/device isolation |
| Metadata ownership | Cached-model metadata could be mutated through a dataset | Deep-copy metadata when attaching it to each dataset |
| Download integrity | Interrupted files are reused as completed downloads | Write temporary sibling files and atomically publish completed files |
| Revision fallback | Existing data tags absent from per-clock repos fail without fallback | Retry the same revision in the shared data repo; retain missing-tag diagnostics |
| Logging | Global logger and HF progress changes affect unrelated code | Use pyaging logger names and per-context download progress control |
| Release commands | Default version downgrades the package; failed notebooks do not fail the release | Derive version from source, use portable updates, and stop on notebook errors |
| HF sync script | Missing local catalog reports a successful zero-clock release | Reject empty catalogs before creating a Hub client |

## Efficiency measurements

These are local component measurements, not claims about total pipeline speed.

| Operation | Workload | Before | After | Numerical check |
| --- | --- | --- | --- | --- |
| EPIC aggregation | 40 samples × 20,000 probes, including missing values | 1.826 s | 0.0142 s | Maximum absolute difference 0 for valid uniquely labeled inputs |
| Feature lookup | 500,000 source features and 20,000 clock features, warmed index, median of 5 runs | 104.3 ms | 1.13 ms | Same selected feature positions |
| Quantile helper | 200 × 20,000 float64, median of 5 runs | Relative baseline 1 | 1.81× faster | Bitwise equal float64 output |
| Tied ranks | 8,113 genes, CPU, median of 7 runs | 10.869 ms | 0.370 ms | Bitwise equal ranks; independent SciPy parity tests |

The clock cache avoids repeated model loads while retaining only the configured
number of models. Its capacity is a model count, not a byte limit. The prediction
guide explains dataset ordering, eviction, revision changes, and memory tradeoffs.

## Validation

The unmodified baseline passed 412 tests, with 80 skips for unavailable local
artifacts and 361 tests deselected by the configured offline test policy. Each
bug fix has a regression that failed before its implementation.

- Final local suite: 556 passed, 82 skipped, 361 deselected. The skips are 80
  unavailable local-artifact cases and two CUDA cases. The 17 warnings come from
  existing tAge warning tests. Lint, formatting, and whitespace checks passed.
- Downloaded unchanged HF weights for Horvath2013, AltumAge, CpGPTGrimAge3,
  DunedinPACE, PhenoAge, PhenoAgeSaoPaulo, and LinAge2. Compared 0.5.2 and 0.5.3
  on identical inputs at batch size 1 and whole-cohort size. Maximum release
  difference was 1.42e-13, and maximum within-version batch difference was 2.85e-14.
- All seven clocks passed their boundary golden checks using those same weights.
  Only the two artificial PhenoAge infinities have updated expectations, derived
  from the same coefficients and algebraically equivalent log-space calculation.
- LinAge2 reproduces its two published example subjects and their batch invariance.
- Real temporary bigWig files verify signal means, arcsinh transforms, and missing regions.
- Wheel and source distribution build successfully. Release CI independently
  verifies the tagged version, tests, build, and package metadata before publishing.

## Follow-up recommendations

- Move pickled model metadata to JSON and tensor weights to a format such as
  safetensors in a separate artifact migration. Existing pickles depend on Python
  class paths; changing their format requires coordinated HF updates and parity checks.
- Add a GPU CI runner to exercise real CUDA/CuPy allocation and batch behavior.
  The current macOS audit can verify CPU behavior and simulated transfer boundaries.
- Consider host-memory streaming for very large datasets. The current pipeline
  still builds a complete aligned matrix per clock even though GPU transfers are batched.
- Prepare SystemsAge's plain tensor assets once per loaded model. They currently
  bypass module dtype/device conversion and can be copied again per batch. This
  needs old-pickle compatibility checks and real SystemsAge prediction parity;
  the new cache already avoids repeated deserialization, but not those asset copies.
- Keep data release tags distinct from package-only releases. Retagging or
  restamping every clock for a software patch is unnecessary and changes HF state.
  Live inspection found `v0.5.2` in the shared repo but only `v0.5.1` for tAge,
  Horvath2013, and AltumAge. Shared `v0.5.2` has the latter two clocks but lacks
  tAge and its mapping sidecar. A complete common release snapshot requires an
  explicitly authorized HF update; this release leaves that state untouched.
- Retain scientific provenance and full-catalog prediction checks when adding or
  rebuilding a clock. Package regression tests cannot replace comparison with the
  original implementation and representative biological data.
