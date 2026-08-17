# Contributing to pyaging

Thank you for your interest in improving `pyaging`. This guide covers local setup, testing, documentation, the workflow for adding a new aging clock, and how releases work. AI agents (and anyone who wants the package's usage conventions) should also read [AGENTS.md](AGENTS.md).

## Development setup

The repository is managed with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/lucascamillomd/pyaging.git
cd pyaging
uv sync
```

`uv sync` installs the package plus the `dev` dependency group, an umbrella that includes the `test`, `docs`, `notebooks`, and `lint` groups. Install a leaner subset with `uv sync --no-default-groups --group test` (or `docs`, `notebooks`, `lint`). Histone-mark clocks need an optional extra: `uv sync --extra histone`.

Run all commands through `uv run`, for example `uv run python` or `uv run pytest`.

## Running tests

```bash
uv run pytest
```

By default this runs the offline suite: tests marked `full_catalog` (downloads and validates the complete clock catalog, ~25 GiB) and `online` (accesses public remote services) are deselected. Opt in explicitly when needed:

```bash
uv run pytest -m online
uv run pytest -m full_catalog
```

## Linting and formatting

Linting and formatting use `ruff`, wired through pre-commit:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

`make lint` and `make format` run the same ruff checks directly.

## Building the documentation

```bash
make docs
```

This copies the tutorial and clock notebooks into `docs/source/` and builds the Sphinx site with `uv run make -C docs html`. The rendered pages land in `docs/_build/html/`.

## Adding a new clock

1. Copy `clocks/notebooks/template.ipynb` to `clocks/notebooks/<clock_name>.ipynb` and implement the clock there. Executing the notebook writes the model weights to `clocks/weights/`; weights are not committed to git and are published to the [`lucascamillomd/pyaging-data`](https://huggingface.co/lucascamillomd/pyaging-data) Hugging Face repository by the maintainer.
2. Add the clock's entry to `clocks/metadata/clock_metadata.json`, using only terms from `clocks/metadata/controlled_vocabulary.json` for the controlled fields.
3. Metadata is validated by `clocks/metadata/validate_metadata.py`; run the validation through `uv run pytest tests/test_clock_metadata.py`.
4. Run the offline test suite and the pre-commit hooks before opening a pull request.

If you cannot prepare the notebook yourself, open a [clock request issue](https://github.com/lucascamillomd/pyaging/issues/new/choose) with the paper DOI and weight availability instead.

## Releases (maintainers)

Releases are tag-driven. The version is single-sourced in `src/pyaging/__init__.py`, which hatchling reads at build time. Pushing a `v*` tag runs `.github/workflows/release.yaml`, which verifies that the tag matches `pyaging.__version__` and that the tagged commit is on `main`, runs the offline test suite, builds the distribution, publishes it to PyPI, and creates the GitHub release.
