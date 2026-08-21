## 10. USER-GATED: HuggingFace upload checklist

**Not performed. Requires the user present.** The HF permission classifier has blocked
unattended uploads on this repo before, and creating public model repos is outward-facing
and hard to reverse.

Flow: `clocks/hf_repo_sync.py` / `make upload-clocks-to-hf`.

**→ `pyaging/tage`**
- `clocks/weights/tage.pt` (354,981 bytes)
- the clock's `config.json` metadata as emitted by the sync script
- `clocks/weights/tage_gene_mapping.csv.gz` (1,385,364 bytes)

**→ `pyaging/tagemortality`**
- `clocks/weights/tagemortality.pt` (354,725 bytes)
- the clock's `config.json` metadata as emitted by the sync script

The gene mapping ships only with `tage`; `prepare_tage` resolves it from that one repo for
both clocks, so it must not be duplicated into `pyaging/tagemortality`.

**Also required for the aggregate:** `all_clock_metadata.pt` must be re-published, or the
Clock Explorer's HF fallback path and anything else reading the published aggregate will
keep reporting 177 clocks (see Concerns §1).

**Post-upload live verification:**

1. Re-run `uv run pytest tests/integration/test_tage_end_to_end.py` **without** the local-
   weights monkeypatch seam, so the clocks load from HuggingFace over the wire.
2. Confirm the gene-mapping download resolves — `prepare_tage` must fetch
   `tage_gene_mapping.csv.gz` from `pyaging/tage` and not fall back or error.
3. Re-run `uv run python docs/source/make_clock_data.py` with **no** `--metadata-path`; it
   should now print `generated 179 clocks` and leave the committed artifacts byte-identical.
   That is the cleanest single signal that the aggregate published correctly.

**Sidecar assets are now automatic (final-fix wave).** `clocks/hf_repo_sync.py` uploads every
`clocks/weights/<clock>_*.csv.gz` alongside the weight file, so `tage_gene_mapping.csv.gz`
reaches `pyaging/tage` as part of the ordinary `make upload-clocks-to-hf` run — no manual
`hf upload` step. The prefix is clock-scoped, so `tagemortality` still gets no copy. Before
the fix the sync uploaded only `config.json`, `README.md` and `<clock>.pt`, which would have
made every user's first `prepare_tage` call 404 on the mapping download.

**Checklist to run with the user present:**

- [ ] `make upload-clocks-to-hf` (uploads weights, `config.json`, README, and the `tage`
      gene-mapping sidecar).
- [ ] Confirm `tage_gene_mapping.csv.gz` is actually listed in the `pyaging/tage` repo tree —
      it is the one asset whose absence is invisible until a user runs `prepare_tage`.
- [ ] Re-publish `clocks/metadata/all_clock_metadata.pt` to `lucascamillomd/pyaging-data`
      (the Makefile's metadata upload step), or the aggregate keeps reporting 177 clocks.
- [ ] `make -C docs data` regenerates the Clock Explorer artifacts (`docs/_static/clocks.json`,
      `clock_glossary.csv`, `all_clock_metadata.pt`) from what is live on HuggingFace, including
      the per-repo download counts that drive the popularity ranking. Re-run it **after** the
      uploads land and verify the regenerated artifacts before committing them — running it
      before the upload bakes the stale 177-clock aggregate into the docs.
