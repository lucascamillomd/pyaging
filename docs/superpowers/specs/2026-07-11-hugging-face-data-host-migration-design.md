# Hugging Face Data Host Migration Design

**Date:** 2026-07-11  
**Status:** Approved for implementation planning

## Context

`pyaging` serves model weights and related public data directly from an AWS S3
bucket. Researchers, clean environments, CI jobs, and users who remove their
local cache repeatedly download these files, leaving the project owner liable
for S3 egress charges.

The active S3-backed assets comprise approximately 28.5 GiB:

- 173 current clock-weight files (approximately 24.92 GiB);
- one aggregate metadata file (approximately 185 KiB);
- seven example-data files (approximately 1.45 GiB); and
- 16 currently referenced supporting files (approximately 2.17 GiB).

The full bucket is approximately 30.69 GiB, but it contains obsolete or
unreferenced objects that will not be migrated. The largest current files are
approximately 1.88 GiB, below Hugging Face's recommended per-file size limit.
Hugging Face Hub model repositories support files and repositories at this
scale, provide CDN-backed downloads, and use Xet for chunk-level storage and
transfer deduplication. The similar SystemsAge weights may benefit materially
from this deduplication, although the exact stored size will only be known after
upload.

Public Hugging Face storage is best-effort for free accounts. The repository
owner should use a PRO account or pursue an open-source research storage grant
if guaranteed capacity becomes necessary. Hugging Face documents storage and
request-rate limits, but does not list per-byte public download egress charges
to the repository owner.

Relevant Hugging Face documentation:

- <https://huggingface.co/docs/hub/en/storage-limits>
- <https://huggingface.co/docs/hub/en/rate-limits>
- <https://huggingface.co/docs/huggingface_hub/en/guides/download>
- <https://huggingface.co/docs/huggingface_hub/en/guides/upload>

## Goals

1. Make Hugging Face the only operational remote host used by package code,
   documentation generation, notebooks, and maintainer release commands.
2. Move all current clock weights, aggregate metadata, example data, and
   supporting files to a public Hugging Face model repository.
3. Preserve existing public Python APIs and current flat local paths under
   `dir="pyaging_data"`.
4. Use Hugging Face's native download cache, retry behavior, and incremental
   upload support.
5. Ensure only Lucas (`lucascamillomd`) can publish or change hosted assets.
6. Remove pyaging-owned AWS egress from future package and notebook usage.
7. Publish the completed HF-native package to PyPI as `pyaging==0.3.0` and
   verify that a clean installation uses Hugging Face successfully.

## Non-goals

- Migrating obsolete or unreferenced S3 objects.
- Keeping S3 as a fallback or mirror.
- Pinning package downloads to a data tag or commit; downloads intentionally
  track the Hub repository's `main` branch.
- Converting pickled PyTorch model objects to a non-executable serialization
  format.
- Automating destructive S3 bucket deletion or access-policy changes.
- Refactoring unrelated package functionality.

## Decisions

| Area | Decision |
| --- | --- |
| Hub repository | Public model repo `lucascamillomd/pyaging-data` |
| Repository owner | Directly owned by `lucascamillomd`; no organization or collaborators |
| Remote revision | Always `main` |
| Runtime client | `huggingface_hub.hf_hub_download` |
| Local destination | Preserve flat files under caller-provided `dir` through root-level Hub paths and `local_dir=dir` |
| Fallback | None; HF-only |
| Publishing | Maintainer-run `hf upload`; never GitHub Actions |
| Routine publish order | Weights first, aggregate metadata last |
| Migrated content | Current package/notebook dependencies only |
| Repo licensing | `other`, with provenance and per-clock restrictions documented |
| Package release | Publish and verify PyPI version `0.3.0` after migration validation |

The stored Hugging Face credential was verified as a write-capable token for
the `lucascamillomd` account. The target repository does not yet exist and will
be created during implementation. The current account belongs to no Hugging
Face organizations, which is consistent with direct sole ownership.

## Hugging Face Repository Layout

```text
lucascamillomd/pyaging-data
├── README.md
├── <clock_name>.pt
├── all_clock_metadata.pt
├── <example-data files>
├── Ensembl-105-EnsDb-for-Homo-sapiens-genes.csv
└── supporting_files/
    ├── <shared dependency files>
    └── cpgpt_grimage3_dependencies/
        └── <CpGPT dependency files>
```

Package-downloaded files live at the Hub repository root because
`hf_hub_download(..., local_dir=dir)` preserves repository-relative
subdirectories. Root-level paths therefore preserve existing local paths such
as `pyaging_data/horvath2013.pt`, `pyaging_data/GSE139307.pkl`, and
`pyaging_data/Ensembl-105-EnsDb-for-Homo-sapiens-genes.csv`. The approximately
182 root files are well below Hugging Face's repository recommendations.
Notebook-only dependencies remain under `supporting_files/` because their
notebook cells control the destination filename explicitly. The new layout also
removes the S3-specific `weights0.1.0` and `metadata0.1.0` directory names;
`main` is the live catalog, so versioned folder names would incorrectly imply
immutability.

The repository README will document:

- the relationship between the data repository and `pyaging`;
- the four asset categories and their intended consumers;
- provenance and citations;
- per-clock research-only or other usage restrictions;
- the `other` license designation for the mixed-provenance collection;
- the fact that PyTorch `.pt` objects must only be loaded from trusted commits;
  and
- the sole-maintainer publishing policy.

## Asset Scope

### Included

- All 173 `.pt` files currently present in `clocks/weights/`.
- `clocks/metadata/all_clock_metadata.pt`.
- The seven example files referenced by `download_example_data`.
- Supporting files referenced by current package code or source notebooks,
  including the Ensembl metadata, PC clock dependency, GrimAge files, and
  CpGPT GrimAge 3 dependencies.
- Any source-notebook dependency that resolves through a pyaging S3 URL at the
  time implementation begins, provided it remains referenced by a current
  notebook.

### Explicitly excluded

- `clocks/weights0.1.0/cpgptgrimage3_before15-12-2025.pt`;
- `supporting_files/altumage_data.pkl`;
- the unreferenced `supporting_files/cpgpt_grimage3_dependencies/reliable/`
  notebook and clock object (`cpgptgrimage3_reliable.ipynb` and
  `cpgptgrimage3_reliable.pt`); and
- zero-byte S3 directory placeholder objects.

The implementation will record the final included and excluded path lists in
the migration verification output. This is an audit artifact, not a new runtime
manifest subsystem.

## Package Architecture

### Internal HF download boundary

A small internal module will own all Hub-specific behavior. It will:

- define `REPO_ID = "lucascamillomd/pyaging-data"`;
- define `REVISION = "main"` explicitly;
- accept a path within the Hub repository, caller-provided local directory,
  logger, and indentation level;
- call `hf_hub_download` with `local_dir=dir`;
- return the resolved local filepath; and
- translate known Hugging Face failures into clear pyaging-facing errors while
  preserving the original exception through chaining.

No package consumer will construct Hugging Face URLs or call the SDK directly.
This keeps repository identity, path rules, logging, and failure handling in one
place.

### Consumer mappings

| Consumer | Hub path |
| --- | --- |
| `load_clock(clock_name, ...)` | `<clock_name>.pt` |
| `load_clock_metadata(...)` | `all_clock_metadata.pt` |
| `download_example_data(...)` | `<mapped filename>` |
| Ensembl preprocessing | `Ensembl-105-EnsDb-for-Homo-sapiens-genes.csv` |
| Documentation catalog generator | `all_clock_metadata.pt` |

Each consumer will load the filepath returned by the helper rather than
reconstructing a presumed local path. Root-level Hub paths and `local_dir=dir`
preserve the existing `<dir>/<filename>` layout, while Hugging Face stores its
revision metadata under `dir/.cache/huggingface/` and avoids unnecessary
redownloads. Internal cleanup code will continue to target the flat clock path,
and tests will verify that this compatibility contract does not regress.

### Notebooks

All active source notebooks and their documentation copies will stop referring
to `pyaging.s3.amazonaws.com`. A notebook cell that needs a local file will use
`hf_hub_download`. A cell or external command that specifically needs a URL may
use the canonical public HF resolver URL for the relevant `main` path. Both are
HF-native; the choice follows the cell's existing interface.

Notebook migration includes current files under `clocks/notebooks/` and their
copies under `docs/source/clock_notebooks/`. Generated `docs/_build/` output is
not edited directly.

### Documentation generation

`docs/source/make_clock_data.py` will obtain aggregate metadata through the HF
download boundary or the same `hf_hub_download` configuration. The existing
committed JSON/CSV fallback remains so documentation builds can succeed without
network access. Comments and tests will refer to a remote-host failure rather
than an S3 failure.

### Dependency cleanup

The package will add `huggingface_hub` as a runtime dependency. After all
consumers move to the HF helper, implementation will confirm that the generic
`urlretrieve` downloader, S3 freshness helper, `requests` import, and
`requests` dependency are unused before removing them.

## Runtime Data Flow

1. A public API maps a clock, dataset, or support resource to its Hub-relative
   filename.
2. The internal helper asks `hf_hub_download` for that file from
   `lucascamillomd/pyaging-data` at `main`, with the caller's `dir` as
   `local_dir`.
3. Hugging Face checks the local metadata and downloads only when the `main`
   file has changed or is absent.
4. The helper returns the local path.
5. The caller loads the file using its existing Torch, pandas, NumPy, or other
   domain-specific loader.

There is no S3 retry or fallback branch at any point.

## Error Handling

- A missing `<clock>.pt` maps to the current user-facing
  clock-not-available outcome, with the Hub exception chained.
- Repository, network, authentication, rate-limit, and local-cache errors remain
  distinguishable and must not be mislabeled as an unknown clock.
- The HF client owns temporary and partial-download handling; callers only see a
  completed local filepath.
- Documentation-generation network failures use the existing committed catalog
  fallback and log the actual HF failure.
- Upload targets fail before publishing if the authenticated account is not
  `lucascamillomd`.
- Upload or verification failures stop the release and do not proceed to tags or
  subsequent release steps.

## Publishing Workflow

### Makefile targets

The root Makefile will remove `upload-to-s3` and all `aws s3` commands.

`upload-clocks-to-hf` will be the routine release dependency. It will:

1. verify that the HF CLI is available and authenticated as `lucascamillomd`;
2. upload the contents of `clocks/weights/` to the Hub repository root;
3. stop if any weight upload fails;
4. upload `clocks/metadata/all_clock_metadata.pt` to
   `all_clock_metadata.pt` at the Hub repository root; and
5. report the resulting Hub revision.

Weights are published before metadata so a newly advertised clock is never
missing its weight. `hf upload` supplies incremental comparison, Xet-backed
chunk deduplication, retry/resume behavior, and commit batching for the large
folder.

`upload-static-data-to-hf` will publish intentionally prepared local example
files and the Ensembl CSV to the Hub repository root, and notebook-only assets
to `supporting_files/`. It is used for the initial migration and later explicit
static-asset updates, but is not a routine release dependency. Large static
files therefore do not need to remain present on every maintainer machine.

Both `release` and `release-slim` will depend on `upload-clocks-to-hf` instead of
the removed S3 target.

### PyPI 0.3.0 release

Publishing `0.3.0` is the final delivery step, not an optional follow-up. The
implementation will:

1. change the package version from `0.2.0` to `0.3.0` consistently in
   `pyproject.toml` and `pyaging/__init__.py`;
2. build both wheel and source-distribution artifacts from the final tested
   commit;
3. inspect the built metadata and contents before upload;
4. create and push the `v0.3.0` git tag using the repository's established
   release workflow;
5. publish the artifacts to PyPI through the repository's authenticated PyPI
   publishing path; and
6. verify the published version by installing `pyaging==0.3.0` into a clean
   environment and running a small public HF download/prediction smoke test.

PyPI versions are immutable. No upload will occur until the HF repository,
package tests, documentation, notebooks, artifact inspection, and pre-publish
smoke tests pass. If publication fails partway through, implementation will
diagnose the existing PyPI state rather than attempting to overwrite or reuse a
published `0.3.0` artifact.

### Authentication and ownership

- Publishing happens only from Lucas's authenticated maintainer environment.
- GitHub Actions will not receive an HF write token and will not publish data.
- The data repository will have no collaborators or organization ownership.
- A new fine-grained token restricted to `lucascamillomd/pyaging-data` should be
  used for publishing.
- The existing broader `cpgpt-upload-write` token should be revoked after the
  migration if it is not required for another repository.
- Tokens are never committed, logged, or passed as command-line arguments;
  authentication uses `HF_TOKEN` or the local HF credential store.

## Initial Migration and Cutover

1. Create the public model repository `lucascamillomd/pyaging-data`.
2. Add its README and mixed-provenance licensing notice.
3. Build the current-dependency inventory and explicit exclusion list.
4. Upload the 173 local clock weights.
5. Upload the current aggregate metadata.
6. Download the actively referenced example and supporting files from S3 once,
   then upload them to their HF paths.
7. Compare source and remote file counts, paths, sizes, and checksums.
8. Exercise representative HF downloads and full clock validation.
9. Change package code, notebooks, documentation generation, and Makefiles to
   HF-only behavior.
10. Set the package version to `0.3.0`, build the distributions, and complete
    every pre-publication validation.
11. Commit the final package changes and create the `v0.3.0` release tag.
12. Publish `pyaging==0.3.0` to PyPI.
13. Install `pyaging==0.3.0` from PyPI in a clean environment and verify an HF
    download and representative prediction.
14. Monitor the HF repository, PyPI release, and package issue reports during
    cutover.
15. Separately retire public S3 access after deciding how long direct legacy URL
    users should be allowed to transition.

The S3 retirement step is manual. The code change alone stops new pyaging
versions from generating S3 egress, but researchers using old package versions
or copied S3 URLs may continue generating AWS charges until public S3 access is
removed. No destructive AWS action is part of this implementation.

## Security Considerations

The clock files are Python pickles loaded through
`torch.load(..., weights_only=False)`. Because downloads track mutable `main`, a
compromised publisher account or write token could replace a weight with code
that executes on a user's machine. This risk already exists with mutable S3
objects, but it must be explicit in the HF design.

Mitigations in scope are:

- direct user ownership with no additional writers;
- a repository-scoped fine-grained write token;
- no write credential in CI;
- HF commit history for every change;
- weights-first, metadata-last controlled publishing;
- a repository warning that `.pt` files must come from trusted commits; and
- representative load/prediction checks before cutover.

Cryptographically signed manifests or conversion to a non-executable model
format would provide stronger protection but conflict with the chosen ability
to update `main` without a package release or require a separate model-format
redesign. They are out of scope for this migration.

## Testing Strategy

### Hermetic automated tests

- Verify the helper passes the expected repository, `main` revision, filename,
  and `local_dir` to `hf_hub_download`.
- Mock successful clock, metadata, example, supporting-file, and documentation
  downloads.
- Verify missing-file, network, repository, authentication, and rate-limit error
  behavior.
- Verify flat paths under the caller-selected download root, cleanup behavior,
  and cached-file reuse.
- Verify documentation generation and its committed fallback.
- Verify Makefile release targets depend on `upload-clocks-to-hf` and contain no
  AWS CLI operations.
- Verify package version metadata is consistently `0.3.0` and built wheel/sdist
  contents reference the HF-native implementation.
- Scan active package code, root/docs Makefiles, source notebooks, and source
  documentation for operational S3 URLs or AWS commands. Historical design and
  plan documents are excluded because they accurately describe prior work.

### Online smoke tests

Online CI or release smoke testing will download only:

- aggregate metadata;
- one small representative clock; and
- one small example file.

This confirms public anonymous access and real HF integration without large
routine transfers.

### Full catalog validation

The existing 173-clock gold-standard test currently downloads approximately
25 GiB per fresh job. Running it across the current two-OS/four-Python matrix
would transfer roughly 200 GiB per release. The full test will instead run once
during migration against the populated HF repository. Routine compatibility
jobs will mock remote downloads. A broader online job may be run explicitly on
one Ubuntu/Python configuration when required, but will not run across the full
matrix by default.

## Acceptance Criteria

The migration is complete only when all of the following are true:

1. Every clock in aggregate metadata has exactly one corresponding HF weight.
2. All active example and supporting dependencies exist at their designed HF
   paths.
3. Remote paths, sizes, and checksums match the migration sources.
4. Representative `.pt` files load and produce their expected predictions.
5. The one-time full 173-clock gold-standard validation passes against HF.
6. Package, documentation, and notebook tests pass.
7. Documentation builds with HF and with its committed offline fallback.
8. Active-source scanning finds no operational S3 URL or AWS CLI dependency.
9. `make release` and `make release-slim` route clock publication through HF.
10. Public downloads require no user token.
11. Only `lucascamillomd` and credentials issued by that account can write to
    the HF repository.
12. No AWS fallback is present in package runtime behavior.
13. The final commit is tagged `v0.3.0`.
14. PyPI serves `pyaging==0.3.0` with the expected wheel and source
    distribution.
15. A clean environment can install `pyaging==0.3.0`, download a public clock
    from `lucascamillomd/pyaging-data`, and run a representative prediction.
