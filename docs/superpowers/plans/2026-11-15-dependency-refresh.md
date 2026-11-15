# Dependency refresh implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task.

**Goal:** Refresh the dependency lock and bring the existing dependency PRs onto current main for validation.

**Architecture:** Resolve the existing dependency ranges with uv, preserving supported Python versions and optional groups. Align the pre-commit Ruff version with the lockfile. Keep the four workflow updates in their existing PRs and refresh all nine PR branches without rewriting their history.

**Tech stack:** uv, Python 3.11 to 3.14, pytest, Ruff, Sphinx, GitHub Actions.

**Spec:** User request to update uv.lock and dependencies and try to fix the PRs.

## Global constraints

- Use uv sync and uv run inside this repository.
- Preserve Python >=3.11 support and current package dependency ranges unless compatibility evidence requires a change.
- Keep online and full-catalog tests separate from the offline suite.
- Do not merge or close the existing PRs as part of branch repair.

## Review focus

- Python-specific dependency resolution must remain valid across 3.11 to 3.14.
- New Ruff rules must not leave lint or formatting checks failing.
- Updated scientific libraries must preserve the offline clock gold standards.
- Documentation dependencies must still produce the Sphinx site.
- Updating a PR must preserve its existing commits and avoid stale-head writes.

### Task 1: Refresh and validate dependencies

**Files:** Modify `uv.lock` and `.pre-commit-config.yaml`; modify source only for demonstrated compatibility failures.

**Interfaces:** Existing `pyproject.toml` ranges remain the resolver input; the lockfile is consumed by CI and local uv commands.

- [ ] Run `uv lock --upgrade` and inspect the resolved changes.
- [ ] Set the Ruff hook revision to the resolved Ruff version.
- [ ] Run a locked test/lint/docs sync, Ruff checks, the offline pytest suite, the Sphinx build, and a package build.
- [ ] Fix only reproduced compatibility failures and rerun affected checks.
- [ ] Commit and push a reviewable dependency refresh PR.

### Task 2: Refresh existing PR branches

**Files:** Remote branches for PRs #30, #31, #32, #33, #34, #48, #49, #52, #54.

**Interfaces:** GitHub's update-branch API consumes an expected head SHA and merges current main without a force push.

- [ ] Inspect current PR state and diffs.
- [ ] Update each still-open branch using its freshly observed head SHA.
- [ ] Link modified PRs and inspect new check states.
- [ ] Report completed local checks separately from pending remote checks; watch pending PRs.
