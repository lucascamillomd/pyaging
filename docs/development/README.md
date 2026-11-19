# Development checks

Run commands from the repository root:

```bash
uv sync --locked --no-default-groups --group test --group lint
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The default suite runs offline. Tests that need local clock builds skip when
`clocks/weights/*.pt` is absent; CUDA tests skip on hosts without CUDA.
Preserve the scientific reference fixtures under `tests/data` and the golden
prediction baselines. Review prediction differences before regenerating either.

Additional checks require data or network access:

```bash
uv run pytest tests/integration/test_hf_smoke.py -m online
uv run pytest -m full_catalog
```

The full catalog can download about 25 GiB. Boundary checks also need locally
built weights. CI runs the offline suite on macOS and Linux with Python
3.11–3.14, plus lint, documentation, and workflow-security checks.

# Package and data releases

The Git tag triggers `.github/workflows/release.yaml`, which verifies the
version and main-branch ancestry, tests, builds, and runs the online tutorials
against Hugging Face `main`. It publishes to PyPI only when both pass, then
creates the GitHub release. Published package versions and tags stay fixed.

`make release VERSION=vX.Y.Z` and `make release-slim VERSION=vX.Y.Z` both
upload clock artifacts and tag Hugging Face repositories; they are not
package-only release commands. Installed packages read Hugging Face `main`, so
the steps run in this order:

1. `release-preflight` stops unless the branch is `main`, `VERSION` has the
   form `vX.Y.Z`, and the tag exists neither locally nor on `origin`.
2. Local work: version bump, lint, build, clock metadata, tests and docs.
   `release` also re-executes the clock notebooks and tutorials.
3. `commit` commits and pushes `main`.
4. `upload-clocks-to-hf` uploads per-clock weights and sidecars, then the
   aggregate metadata.
5. `tag-hf-data-repo` tags the data repository and every per-clock repository
   with `VERSION`.
6. `tag` pushes the Git tag, which starts the PyPI release.

Git and Hugging Face tags keep the `v` prefix; the package version and the
version stamped into clock metadata use the bare number. If a step after
`commit` fails, fix the cause and rerun the same command: `commit` succeeds when
there is nothing new to commit.

The shared data repository remains an active fallback for per-clock
repositories. Set `PYAGING_DATA_REVISION` to an existing data revision when
reproducibility is required. A package-only release need not create a data tag,
but a release whose weights changed must have its Hugging Face data tag. The
`v0.6.1` and `v0.6.2` tags were added on 2026-11-19, after those releases:
on the shared repository, on `pyaging/phenoage`, and on the eight clocks that
have no copy in the shared repository. 0.6.0 and 0.6.3 changed no weights and have no tag.

For an authorized data release, `clocks/hf_repo_sync.py` uploads weights,
metadata, model cards, and clock-specific `*_*.csv.gz` sidecars. tAge requires
`tage_gene_mapping.csv.gz` in `pyaging/tage`; both tAge clocks use that mapping.
It stops before contacting the Hub when a weight file is not in
`clocks/metadata/clock_metadata.json`, and, in a run without clock names, when a
registered clock has no local weight; name the clocks to sync, or pass
`--allow-missing` to skip the missing ones deliberately. Upload weights and sidecars before aggregate metadata, then verify anonymous
downloads, predictions, and the clock gallery. Tutorials using new remote
artifacts must be executed after those artifacts are published.

The [0.5.3 audit](audit-0.5.3.md) records validation and outstanding artifact
work. Historical implementation plans are retained in Git history.
