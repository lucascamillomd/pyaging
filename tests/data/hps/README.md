# HPS original-author reference

The expected predictions were produced by sourcing and executing the original
Kuo lab `HPS.R`, pinned at commit
`6eec9ac5a091e34fece69fa4652ebd9af22c1839`, using R 4.5.3. The author code is
downloaded separately by `clocks/build_hps.py`; it is not vendored here.

The ten input rows are synthetic sin/cos profiles, with eight ages spanning
39–70 years and two extreme profiles testing the score's 0/1 limits. No
participant records are included. Cases cover complete data, an explicit NA,
and reversed columns. The generator also verifies that the author function
returns no prediction when a required column is absent.

Regenerate with:

```sh
Rscript tests/data/hps/generate_reference.R ORIGINAL_SOURCE_DIR tests/data/hps
```

`hps_author_parameters.json` records the published fitted numerical parameters
in source order. `reference_provenance.json` records source and fixture hashes.
