# Clock Catalogue + landing polish — design spec

**Date:** 2026-07-06
**Status:** Approved (brainstorming) — ready for implementation planning
**Author:** Lucas Camillo (with Claude)

## 1. Goal & motivation

A second polish pass on the pyaging docs (already migrated to pydata-sphinx-theme
with the interactive explorer, PR #12 merged). Four things fall short:

1. The explorer's chip-facet panel is cramped and doesn't scale — some columns
   (platform, last author, predicts) have many values, and only 6 columns are
   filterable.
2. The table is too narrow: the left section-nav and right on-this-page sidebars
   waste horizontal space that the table should use.
3. The landing "Contents" section (stacked caption + toctree links + "Indices and
   Tables") looks unpolished next to the rest of the page.
4. The landing embeds a "Quick start" code subsection instead of pointing at the
   real tutorial, and the hero lacks the logo.

## 2. Decisions (locked during brainstorming)

| Decision | Choice |
|---|---|
| Explorer rename | **Clock Catalogue** (page title + navbar; URL stays `clock_glossary.html`) |
| Filter widget | **Multi-select dropdowns with in-dropdown search** (checkbox + count); AND across columns, OR within |
| Filterable columns | **All categorical columns**; numeric columns stay sort-only |
| Full-width | Remove **both** sidebars on the Catalogue page + relax article max-width (wide, not literally edge-to-edge) |
| Landing Contents | **Remove entirely**; toctrees become `:hidden:` (still power navbar/sidebar); drop "Indices and Tables" |
| Quick-start | **Remove the section**; point the top "Get started" CTA card at the Illumina Human Methylation Arrays tutorial |
| Hero logo | Add the pyaging logo centered at the top of the hero, above the "pyaging" heading; keep the badge row just below the logo |

## 3. Scope & architecture

All work stays inside the Sphinx build. **No changes** to `pyaging/**`, models,
clock behavior, notebooks, or clock-data content. The Catalogue keeps its URL.
Vanilla ES5 JS (no framework), pydata-sphinx-theme, `sphinx_design`.

### 3.1 Clock Catalogue rename

- `docs/source/clock_glossary.rst`: retitle "Aging Clock Explorer" → **Clock
  Catalogue**; update the intro paragraph (facet list wording). URL unchanged
  (filename stays `clock_glossary`).
- `docs/source/index.rst`: the CTA card "Clock Explorer" → "Clock Catalogue"
  (link `clock_glossary` unchanged).

### 3.2 Filter model — dropdown filter bar

Replaces the left chip-facet panel (`.ce-facets`) with a horizontal **filter bar**
of dropdown buttons rendered above the table, inside the toolbar area.

- **Filterable columns** (categorical), in this order: `data_type`, `species`,
  `platform`, `model_type`, `unit`, `tissue`, `last_author`, `journal`,
  `predicts`, `population`, `approved_by_author`. Numeric columns (`citations`,
  `year`, `n_features`) are **not** filters — they remain sortable via header
  click and the sort dropdown.
- **`clock_explorer_core.js`**: `FACET_FIELDS` expands to the categorical list
  above (drop the numeric ones). `computeFacets` and `filterClocks` are otherwise
  unchanged — `filterClocks` already applies AND across fields and OR within a
  field for an arbitrary `FACET_FIELDS`. Counts shown are over the **full**
  dataset (matches current behaviour).
- **`clock_explorer.js`** (DOM): each filterable column renders as a `<button>`
  in the filter bar showing the column label and, when active, a count badge
  (e.g. `Platform 3`). Clicking toggles a popover panel anchored under the button
  containing: (a) a search `<input>` that substring-filters the checkbox list,
  and (b) a scrollable checkbox list of `{value, count}` for that column.
  Toggling a checkbox updates `state.selected[field]` and re-renders. One popover
  open at a time; close on outside-click and Escape. Built with `createElement`,
  ES5.
- **Active filters**: a row of removable chips below the filter bar shows each
  selected `value` (e.g. `Platform: Illumina 450K ✕`); the `✕` removes just that
  value. **Reset** clears all `state.selected`.
- The obsolete "Filters" collapse toggle (there is no left panel to collapse now)
  is removed from the toolbar.

### 3.3 Full-width layout (Catalogue page only)

- **Left (primary) sidebar** — remove on this page via `conf.py`:
  `html_sidebars = {"clock_glossary": []}`.
- **Right (secondary) sidebar** — remove via page-wide metadata at the very top
  of `clock_glossary.rst`: a field-list line `:html_theme.sidebar_secondary.remove: true`.
- **Article width** — the Catalogue JS adds a body class on mount
  (`document.body.classList.add("ce-fullwidth")`); `clock_explorer.css` then
  relaxes the theme's content cap scoped to that class
  (`.ce-fullwidth .bd-main .bd-content, .ce-fullwidth .bd-article { max-width: none; }`),
  keeping a comfortable page gutter (not edge-to-edge). Scoping to the JS-added
  class means no other page is affected and the no-JS fallback keeps normal width.
- With filters moved to the top bar, the explorer body (`.ce-main`) is a single
  full-width column for the table/cards.

### 3.4 Landing page polish (`docs/source/index.rst` + `custom.css`)

- **Hero logo** — add, as the topmost hero element, a centered logo image above
  the "pyaging" heading. Raw HTML (output-relative path):
  `<div class="pyaging-hero-logo-wrap"><img class="pyaging-hero-logo" src="_static/logo.png" alt="pyaging logo"></div>`
  placed before the badge `<center>` block. Final top order: **logo → badge row →
  "pyaging" heading → tagline**. `custom.css`: `.pyaging-hero-logo { display:block;
  margin: 1.6rem auto 0.4rem; width: 108px; height:auto; }`.
- **Remove the "Contents" section and "Indices and Tables"** — delete the visible
  `Contents` heading, the per-caption visible toctrees, and the genindex/modindex/
  search list. Re-add the same toctrees at the bottom of the file with `:hidden:`
  so the navbar and left sidebar still build (Getting Started → installation,
  clock_glossary; Tutorials → tutorials/index; API Reference → pyaging; Clock
  implementation → clock_implementation). The page's last visible element is the
  graphical abstract.
- **Quick-start** — remove the `Quick start` heading and its `pip install` + code
  blocks. Repoint the top **"Get started"** CTA card:
  `:link: tutorials/tutorial_dnam_illumina_human_array` `:link-type: doc`, and
  reword its blurb so install is still discoverable (e.g. "Install pyaging and run
  your first prediction — the Illumina 450K/EPIC walkthrough."). The graphical
  abstract stays.

## 4. File map

| File | Change |
|---|---|
| `docs/source/clock_glossary.rst` | retitle → Clock Catalogue; add `:html_theme.sidebar_secondary.remove: true`; update intro |
| `docs/source/index.rst` | hero logo; remove Contents + Indices; hidden toctrees; remove Quick start; Get-started card → tutorial; Clock Explorer card → Clock Catalogue |
| `docs/source/conf.py` | `html_sidebars = {"clock_glossary": []}` |
| `docs/_static/clock_explorer_core.js` | expand `FACET_FIELDS` to all categorical columns |
| `docs/_static/clock_explorer.js` | filter-bar dropdowns (search + checkboxes), active-filter chips, remove Filters-collapse toggle, add `ce-fullwidth` body class |
| `docs/_static/clock_explorer.css` | filter-bar + popover + active-chip styles; `.ce-fullwidth` width override; drop left-facet-panel styles |
| `docs/_static/custom.css` | `.pyaging-hero-logo` styling |
| `docs/_static/tests/clock_explorer_core.test.js` | cover expanded `FACET_FIELDS` (multi-column AND/OR incl. tissue/last_author) |

## 5. Testing & verification

- **Pure logic**: Node unit tests — `computeFacets` returns groups for the new
  categorical fields; `filterClocks` ANDs across two new fields (e.g. tissue +
  platform) and ORs within one; numeric fields are absent from facets.
- **Build**: `make clean html` clean (no new warnings); Catalogue page renders the
  filter bar + full-width table with both sidebars gone; landing shows logo, no
  Contents/Indices, and the Get-started card links to the tutorial; `:hidden:`
  toctrees still populate the navbar/left sidebar on other pages.
- **Data test**: `pytest docs/source/test_make_clock_data.py` still passes and
  stays hermetic (clock data is unchanged by this iteration).
- **Visual sign-off**: headless-Chrome screenshots of the Catalogue (filter
  dropdown open, full-width table) and the landing (logo hero, no Contents),
  shared for approval before finalising.

## 6. Risks & mitigations

- **`sidebar_secondary.remove` metadata syntax** — verify it removes the right
  sidebar in the build; fallback is CSS hiding scoped by the `.ce-fullwidth` body
  class.
- **Full-width CSS leaking to other pages** — scoped to the JS-added `.ce-fullwidth`
  body class, which is set only where the Catalogue mounts; no-JS fallback keeps
  normal width.
- **Popover overflow on mobile** — filter bar wraps; popovers use a capped
  max-height with internal scroll and stay within the viewport.
- **Removing the left nav on the Catalogue page** — intended; the navbar still
  provides site navigation.
- **Logo raw-HTML path** — landing `index.html` is at the html root, so
  `src="_static/logo.png"` resolves; verified in the build.

## 7. Non-goals

- No `pyaging/**`, model, notebook, or clock-data (`clocks.json`/CSV) changes.
- No new dependencies or framework; stays Sphinx on Read the Docs.
- URL stability: the Catalogue keeps `clock_glossary.html`.
- No dynamic (filter-aware) facet counts in this pass — counts stay over the full
  dataset; can be revisited later.
