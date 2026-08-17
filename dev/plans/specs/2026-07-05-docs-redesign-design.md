# pyaging documentation redesign — design spec

**Date:** 2026-07-05
**Status:** Approved (brainstorming) — ready for implementation planning
**Author:** Lucas Camillo (with Claude)

## 1. Goal & motivation

The pyaging docs site (`pyaging.readthedocs.io`) looks plain, and its clocks
"glossary" is a static CSV table with a hand-rolled filter script that does not
exploit the newly-added rich per-clock metadata (tissue, predicts, unit,
model_type, platform, population, journal, last_author, n_features, citations, …).

Two outcomes:
1. **A sleeker, more modern site** — better theme, a real landing page, cohesive
   branding derived from the logo.
2. **A flagship interactive "Clock Explorer"** — faceted filtering, search, sort,
   a table⇄card toggle, and inline detail expansion, so users can find the best
   clock for their needs.

## 2. Decisions (locked during brainstorming)

| Decision | Choice |
|---|---|
| Scope | Full redesign (theme + landing + explorer + polish) |
| Theme | `pydata-sphinx-theme` (top navbar, light/dark; scanpy/numpy ecosystem standard; `scanpydoc` already imported) |
| Clocks UI | Dedicated interactive Clock Explorer |
| Explorer engine | **Bespoke vanilla-JS** (no third-party grid library) |
| Palette | Derived from `logo.png` |
| Explorer views | Table (default) + Card view toggle, shared filter/sort state |
| Row click | Inline expand detail panel |
| Asset delivery | Bundled/self-contained in `docs/_static` (offline-safe, CSP-safe) |

## 3. Palette (from `logo.png`)

Sampled dominant logo colors → design tokens (light + dark):

| Token | Light | Dark | Source |
|---|---|---|---|
| Primary (links, buttons, active) | `#3a7ca5` | `#5fa8d3` | logo blue `#4080b0` |
| Secondary (badges, highlights) | `#10a0b0` | `#2bc4d4` | logo cyan `#10b0c0` |
| Accent (call-outs, hover) | `#e0b000` | `#f0d030` | logo gold `#f0d030` |
| Ink / dark surface | `#0b1b2b` | `#0e1826` | logo navy `#001020` |

Exact shades finalized during implementation against contrast (WCAG AA).

## 4. Architecture & scope

All work happens inside the existing Sphinx build. **No changes** to package code
(`pyaging/**`), models, clock behavior, tutorials (notebooks), API autodoc content,
or individual clock notebooks — they inherit the new theme automatically. No
re-running notebooks; no S3 weight changes.

### 4.1 Theme & site chrome
- `docs/environment.yml`: add `pydata-sphinx-theme` (drop `sphinx-book-theme`).
- `docs/source/conf.py`: `html_theme = "pydata_sphinx_theme"`; translate options
  (repository → `github_url` / `icon_links`), enable top navbar with links
  (Home · Install · **Clocks** · Tutorials · API), built-in theme switcher,
  icon links (GitHub, PyPI, paper DOI), secondary "on this page" sidebar,
  light/dark pygments styles, footer.
- `docs/_static/custom.css`: site-wide theme — CSS variables mapping the palette
  onto `--pst-color-*`, modern font stack, softened radii, subtle shadows, spacing.
  Light + dark. This removes the "plain" feel across all pages.

### 4.2 Landing page (`docs/source/index.rst`)
Rebuilt from `sphinx_design` components:
- **Hero**: logo, name, tagline ("GPU-accelerated biological aging clocks in
  Python"), CTA buttons → Get started / Explore the clocks / GitHub / Paper.
- **Stats strip**: `{clocks} · {data types} · {species}` — all three counts
  generated from the aggregate metadata at build time so they never go stale.
- **Feature cards** (grid): 170+ published clocks · multi-omic (DNAm, histone,
  ATAC, RNA, blood chemistry) · GPU-optimized PyTorch backend · one-line API.
- **Quick-start** code block (`pip install pyaging` + minimal predict example).
- **Supported data types** card row linking to matching tutorials.
- Graphical abstract retained, restyled.

### 4.3 Clock Explorer

**Data pipeline.** Rename `make_clock_glossary.py` → `make_clock_data.py`, which
downloads the aggregate `all_clock_metadata.pt` from the **public** S3 HTTPS URL
and writes:
- `docs/_static/clocks.json` — array of clock objects: all metadata fields + notes
  + doi + `notebook` href (`clock_notebooks/<clock>.html`).
- `docs/_static/clock_glossary.csv` — retained for download + no-JS fallback.

Generation runs from a `conf.py` `builder-inited` hook so it regenerates on every
build (local **and** Read the Docs, which runs only `sphinx-build`). On any error
(S3 unreachable, torch import), it falls back to the committed `clocks.json`.
`clocks.json` is committed so the fallback always exists. A `Makefile` target keeps
local regeneration convenient.

**UI.** Bespoke vanilla-JS app mounted into the existing glossary page
(`clock_glossary.rst`, URL stays `clock_glossary.html`, retitled "Clock Explorer").
The existing static `csv-table` remains inside the mount container as a **no-JS
fallback** (SEO + accessibility); the JS replaces it on load.

Components (self-contained in `docs/_static`):
- `clock_explorer.js` — organized into clearly separated units:
  - data load/normalize (fetch `clocks.json`)
  - **pure** filter + sort functions (unit-tested with Node)
  - table renderer, card renderer, detail-panel renderer, facet/toolbar controller
- `clock_explorer.css` — palette-matched styling for both views.

Features:
- **Toolbar**: global search (name/author/notes), Table⇄Cards toggle, live result
  count, Reset, Download CSV.
- **Faceted filters** (multi-select chips): Data type, Species, Platform,
  Model type, Unit, Predicts — AND across facets, OR within a facet.
- **Sort**: click any table header (numeric-aware for Citations/Year/N features);
  a sort dropdown drives the card view.
- **Table view**: sensible default columns + a Columns show/hide menu; click a row
  to expand the inline detail panel.
- **Card view**: one card per clock with key badges + expand.
- **Inline detail panel**: full notes, every metadata field, DOI link, link to the
  clock's notebook.
- **Responsive**: table scrolls horizontally on mobile; cards stack.

## 5. File map

| File | Change |
|---|---|
| `docs/environment.yml` | add `pydata-sphinx-theme`, drop `sphinx-book-theme` |
| `docs/source/conf.py` | theme + options; `builder-inited` hook to gen `clocks.json`; js/css files |
| `docs/_static/custom.css` | site-wide palette + polish (light/dark) |
| `docs/source/index.rst` | new landing page (sphinx_design) |
| `docs/source/make_clock_data.py` | (renamed) emits `clocks.json` + CSV from aggregate |
| `docs/source/clock_glossary.rst` | retitled; mount container + no-JS fallback table |
| `docs/_static/clock_explorer.js` | new bespoke explorer app |
| `docs/_static/clock_explorer.css` | new explorer styles |
| `docs/_static/clocks.json` | new committed data file (fallback) |
| `docs/_static/clock_glossary.js` | removed (superseded) |
| `docs/Makefile` | local regen target for `clocks.json` |

## 6. Testing & verification

- **Pure logic**: Node unit tests for the filter/sort functions (sample data,
  edge cases: empty facets, multi-facet AND/OR, numeric vs string sort).
- **Data**: assert `clocks.json` count == aggregate count and required fields
  present for every clock.
- **Build**: local `make html` builds clean (no new errors); `clocks.json`,
  `clock_explorer.js/.css` copied to `_build/html/_static`; explorer container +
  no-JS fallback present in rendered HTML; landing page renders.
- **Theme sanity**: notebook gallery (`clock_implementation`) + tutorials render
  correctly under PyData theme.
- **Visual sign-off**: build the real page and share the rendered result with the
  user; iterate on look/feel before finalizing.

## 7. Risks & mitigations

- **Theme-option translation** (book-theme → pydata): verified via build; known
  differences (repository button → `github_url`/`icon_links`).
- **`sphinx_design` compatibility**: fully supported on PyData theme.
- **Build-time S3 fetch on RTD**: public HTTPS URL, no creds; committed fallback on
  failure so builds never break.
- **Offline/CSP**: everything bundled in `_static`; no CDN.
- **URL stability**: explorer keeps `clock_glossary.html`.

## 8. Non-goals

- No package/model/behavior changes; no notebook re-runs; no S3 weight changes.
- Tutorial and API-reference **content** unchanged (only restyled by the theme).
- No new external hosting or framework — stays Sphinx on Read the Docs.
