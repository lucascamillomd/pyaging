# PCBrainAge and standalone PCGrimAge components

These fixtures and artifacts use the original authors' code and weight files.
Neither methylCIPHER nor Biolearn supplies coefficients or expected predictions.

## Sources

- PCBrainAge: [MorganLevineLab/calcPCBrainAge](https://github.com/MorganLevineLab/calcPCBrainAge/tree/8a4a0bf93fe715e48d495af46edf17f10d988afa), commit `8a4a0bf93fe715e48d495af46edf17f10d988afa`. The package data comprise three PCA rotation partitions, two centering partitions, the 15-PC regression, and brain-reference methylation values. The paper is [Thrush et al., 2022](https://doi.org/10.18632/aging.204196).
- PCGrimAge components: [MorganLevineLab/PC-Clocks](https://github.com/MorganLevineLab/PC-Clocks/tree/5e65bce1eeb600949fce0853e46ed2aa6210051d), commit `5e65bce1eeb600949fce0853e46ed2aa6210051d`; `CalcAllPCClocks.RData` comes from the [authors' linked Yale Box distribution](https://yale.app.box.com/s/kq0b0a7lxckxjvaz7x5n4keaug7tewry), file ID `904007593153`. The paper is [Higgins-Chen et al., 2022](https://doi.org/10.1038/s43587-022-00248-2).

`provenance.json` records source-file SHA-256 digests. The PCBrainAge package
DESCRIPTION declares “MIT License” but the pinned repository has no separate
license text. PC-Clocks does not provide an explicit software or weight license
in the pinned repository; its terms are not inferred from pyaging's license.

## Representation and input contract

The model applies `(betas - center) @ rotation @ coefficients + intercept`.
Because all of these stages are affine, the builder composes them once as
`betas @ (rotation @ coefficients) + (intercept - center @ rotation @ coefficients)`.
The operation preserves the authors' double precision and does not retrain,
round, restandardize, or recalibrate any component. This avoids duplicating
large identical PCA matrices across eight standalone components.

PCBrainAge uses 357,852 methylation probes (356,369 `cg*` CpG probes and
1,483 `ch.*` non-CpG probes) and the released core 15-PC regression. It returns
chronological age in years; it does not calculate age-acceleration residuals or
classify Alzheimer's disease. Its missing-CpG reference is the authors' adult
GSE74193 brain means.

Each PCGrimAge component uses the authors' 78,464-CpG rotation and its own named
PC regression. A component retains only chronological age (`age`, years) and
sex (`female`, 1 for female and 0 for male) selected by its original regression.
Missing CpGs use the authors' GSE40279 blood-reference means. Selected age/sex covariates are required for every sample; absent or non-finite
demographic inputs raise an explicit error, matching the original author input
requirement. CpG-reference filling does not invent demographic values.
`df_to_adata` can impute NaNs before prediction according to its selected
strategy; validate age/sex before conversion if original observation status
must be preserved. The runtime checks the values actually passed to prediction. The eight
scores retain the original DNAm proxy units; the seven protein proxies are not
measured concentrations and the pack-years score is not a recorded smoking
history.

Impute isolated missing beta values deliberately before prediction, as with
other pyaging clocks. Runtime missing values are filled from the stored
reference, independent of batch size. The original authors' optional within-
cohort mean imputation is not silently introduced into the pyaging pipeline.
The author PCBrainAge wrapper contains a `datMeth`/`DNAm` typo in the branch
for absent CpGs; its core predictor is used unchanged on complete and explicitly
reference-filled inputs for the oracle.

## Oracle and rebuild

`export_author_models.R` requires only base R. For PCBrainAge it sources and
calls the authors' unmodified function. For PCGrimAge it evaluates the exact PCA
and eight component-assignment lines extracted from the pinned author script.
The full author wrapper also attempts package installs and interactive input,
which are unrelated to its numerical prediction equations.

Eight deterministic inputs cover reference methylation, beta values of 0,
0.5 and 1, two distinct CpG-index patterns, selective perturbations, and every
97th CpG missing with author-reference substitution. Sex and age vary across
the PCGrimAge inputs. `oracle.json` contains R predictions; `coefficients.npz`
contains the double-precision composed weights, intercepts, references, and
feature names. Expected predictions are never computed from the composed
coefficients.

Rebuild local `.pt` weights without downloads:

```bash
uv run python clocks/build_pc_extensions.py
```

To regenerate the compact source coefficients and R oracle, place the two
pinned repositories and the author RData in one directory, then run:

```bash
uv run python clocks/build_pc_extensions.py \
  --source-dir /tmp/pc-sources --rscript /path/to/Rscript --regenerate
uv run pytest tests/models/test_pc_extensions.py
```

The builder verifies all original source checksums before running R. Tests
compare all nine models with the R oracle, one-sample versus full-batch
prediction, saved/reloaded models, and the full `predict_age` pipeline with
reordered and absent CpGs. Only the file-download boundary is replaced in the
pipeline tests.
