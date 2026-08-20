# LinAge2 source material

## Citation

Fong S., Denisov K.A., Nefedova A.A., Kennedy B.K., Gruber J.
*LinAge2: providing actionable insights and benchmarking with epigenetic clocks.*
npj Aging **11**, 29 (2025). PMID 40268972, PMC12019333.
https://doi.org/10.1038/s41514-025-00221-4

Licence: CC BY 4.0. The archive's files are redistributed here unmodified under that licence.

## What is vendored here

The published code archive, minus the three files listed under "What is deliberately not vendored":

| file | role |
|---|---|
| `linAge2.R` | the reference implementation, unmodified |
| `codebook_linAge2.csv` | variable descriptions and units; the authority for the pyaging feature names |
| `logNoLog.csv` | the Box-Cox lambda per feature (only `0` = natural log and blank = identity occur) |
| `paraInit.csv` | training-pipeline parameters |
| `userData.csv` | the paper's two published example subjects (SEQN 8881, 9106) |
| `userData_sanity.csv` | ten training subjects used by the reference script's internal self-consistency check |
| `vMatDat99_{M,F}_pre.csv` | the 59x59 right-singular-vector matrices `V` — the only saved constant inference needs |
| `diagDat99_{M,F}_pre.csv` | the diagonal matrices `D` of singular values; a training artefact, kept because it is 8 KB and it is what re-derives the scree cutoff |

## What is deliberately not vendored

**`mergedDataNHANES9902.csv` (6.9 MB)** — the merged NHANES 1999-2002 training table. It is data,
not code, and nothing in pyaging reads it at runtime.

It matters because **the reference implementation ships no fitted constants.** `linAge2.R` re-runs
the entire training pipeline over that CSV on every invocation to re-derive the per-sex median and
MAD normalization statistics, the Cox coefficients and their training means, and the null-model
coefficients. Those constants therefore had to be extracted once, and they live in `consts/`.

**`uMatDat99_{M,F}_pre.csv` (1.19 MB and 1.18 MB)** — the left singular vectors `U`, one row per
training subject (1045 males, 1034 females). Nothing reads them: at inference a new subject needs
only `V`, and `U` exists in the reference solely to reconstruct training-subject PC scores while the
Cox models are being fitted. Vendoring them would not make the training path reproducible anyway,
since that needs `mergedDataNHANES9902.csv`, which is not here either. Take them from the original
archive if you need them.

## Regenerating `consts/`

`consts/linAge2_export.R` is `linAge2.R` patched to be scriptable and to dump the constants: the
cotinine prompt is answered `"C"` (the example data holds raw cotinine in ng/mL), the interactive
plotting loop and `ggplot2` are removed, and a diagnostic and export block is appended. Given a
local copy of the full archive:

```bash
ARCHIVE="/path/to/linAge2_code"      # the directory holding mergedDataNHANES9902.csv
WORK=$(mktemp -d)
cp "$ARCHIVE"/*.csv "$WORK"/
cp clocks/linage2_source/consts/linAge2_export.R "$WORK"/
(cd "$WORK" && Rscript linAge2_export.R)     # ~40 s, needs only the `survival` package
cp "$WORK"/consts/{features,normstats,coxM,coxF,coxnull}.csv clocks/linage2_source/consts/
```

The export also writes `consts/vM.csv` and `consts/vF.csv`, which are byte-identical copies of
`vMatDat99_{M,F}_pre.csv`. They are not kept: `extract_linage2_params.py` reads the shipped
matrices directly, so there is no second copy to drift.

Never run it inside the source directory: it writes `scree_M.pdf`, `scree_F.pdf` and
`userData_out.csv` into the working directory.

Then rebuild the params file and check it still reproduces the published values:

```bash
uv run python clocks/linage2_source/extract_linage2_params.py
uv run pytest tests/test_linage2_params.py
```

## The extracted constants

| file | contents |
|---|---|
| `consts/features.csv` | the 59 model features in `dataMat` order, with the Box-Cox lambda |
| `consts/normstats.csv` | per-sex median and MAD per feature, plus the skip flag |
| `consts/coxM.csv`, `consts/coxF.csv` | the per-sex Cox betas and training means for chronological age and 17 PCs |
| `consts/coxnull.csv` | the per-sex null-model beta and mean, which set the mortality rate doubling time |

The median and MAD reference is **not** the 40-50 year-olds the paper describes. It is
`40 <= RIDAGEYR <= 50` intersected with the NHANES 1999-2000 wave, complete cases only, excluding
accidental deaths: n = 281 males and n = 304 females. Anyone regenerating the constants has to
reproduce that filter chain exactly, which `linAge2_export.R` does because it is the reference
script itself.

## Verification

The reference script reproduces the paper's published biological ages for its two example subjects
(SEQN 8881 → 88.69 y, SEQN 9106 → 64.36 y; the paper reports 88.7 and 64.4). An independent
reimplementation driven only by the constants in `consts/` reproduces both exactly, and
`tests/test_linage2_params.py` re-runs that check against `clocks/linage2_params.json` on every
test run.
