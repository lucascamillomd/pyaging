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
version and main-branch ancestry, tests, builds, and publishes to PyPI before
creating the GitHub release. Published package versions and tags stay fixed.

Model publication is a separate operation. `make release` and `release-slim`
both upload clock artifacts and tag Hugging Face repositories; they are not
package-only release commands. The shared data repository remains an active
fallback for per-clock repositories. Set `PYAGING_DATA_REVISION` to an existing
data revision when reproducibility is required; package-only releases need not
create a matching data tag.

For an authorized data release, `clocks/hf_repo_sync.py` uploads weights,
metadata, model cards, and clock-specific `*_*.csv.gz` sidecars. tAge requires
`tage_gene_mapping.csv.gz` in `pyaging/tage`; both tAge clocks use that mapping.
Upload weights and sidecars before aggregate metadata, then verify anonymous
downloads, predictions, and the clock gallery. Tutorials using new remote
artifacts must be executed after those artifacts are published.

The [0.5.3 audit](audit-0.5.3.md) records validation and outstanding artifact
work. Historical implementation plans are retained in Git history.
