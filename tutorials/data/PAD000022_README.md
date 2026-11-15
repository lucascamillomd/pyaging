# PAD000022 proteomics example

This example contains 32 real human plasma samples and 134 measured proteins
from the Olink Explore 3072 portion of **Human Plasma Discovery Proteomics**,
deposited by Sara Ahadi in [PRIDE, PAD000022](https://www.ebi.ac.uk/pride/archive/projects/PAD000022).
The [PRIDE record](https://www.ebi.ac.uk/pride/ws/archive/v2/projects/PAD000022)
releases these source data under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
The clock weights have their own research-use terms.

Please cite Kirsher, D. Y., Chand, S., Phong, A., Nguyen, B., Szoke, B. G., and
Ahadi, S. *Current landscape of plasma proteomics from technical innovations
to biological insights and biomarker discovery*. Communications Chemistry 8,
279, 2025. [Publication](https://doi.org/10.1038/s42004-025-01665-1),
[dataset](https://doi.org/10.6019/PAD000022).

## Files

- `PAD000022_subset.pkl` is the pandas DataFrame used by the tutorial.
- `PAD000022_subset.csv` contains the same table in an open text format.
- `PAD000022_assays.csv` identifies each assay by its source name, Olink ID,
  UniProt annotation and panel.
- `PAD000022_provenance.json` records source URLs, SHA-256 checksums, sample
  identifiers, preparation steps and output checksums.

Rows retain the published Olink sample IDs. `subject_id` contains the source
de-identified subject identifier and belongs in `metadata_cols`. All other
columns are source protein names with float64 NPX values.

## Selection and preparation

The subset excludes controls, technical replicates and samples marked
"Sample not analyzed." in the authors' sample spreadsheet. Among the remaining
subjects, it selects the first 32 sorted Olink sample IDs with finite NPX and
`PASS` in both `QC_Warning` and `Assay_Warning` for every selected assay.
Spreadsheet sample IDs have leading zero padding, which is removed solely to
join them to the source NPX CSV. Each retained row is a distinct subject.

The 134 assays are the union of features for the brain, heart and kidney
OrganAge models. Every selected protein has exactly one Olink assay in this
deposit, so no duplicate-assay aggregation or gene-name substitution is used.
The matrix contains unchanged measured NPX, including values below LOD. There
is no synthetic data, imputation, exponentiation, centering or scaling.

## Use and limits

This subset covers all features of `organagechronologicalbrain`,
`organagechronologicalheart`, `organagechronologicalkidney`, and
`organagemortalitybrain`. It does not contain the full Explore 3072 panel.
Exact chronological ages are unavailable in the public subject annotations,
so this example cannot run PAC, HPS or PAOPAC. The tutorial does not substitute
age-group midpoints for measured ages.

The source reports plate-control-normalized NPX, a log2 relative-abundance
measure. These values have not been harmonized to the OrganAge UK Biobank
training data. The example demonstrates scoring and feature coverage, not
clock accuracy or calibration in this cohort. Chronological models return
years; the mortality model returns relative natural-log mortality hazard.
Changing assays or sample type requires separate preparation.

## Reproduce

From the pyaging repository root:

```bash
uv run --with openpyxl python tutorials/data/prepare_pad000022.py \
    --source-dir /tmp/pad000022 --output-dir hf_static_data/repo
```

The script verifies both original source files against fixed SHA-256 hashes
before selecting any data. The assay manifest is committed beside the script.
The pandas pickle uses protocol 4; the CSV provides a version-independent copy.
