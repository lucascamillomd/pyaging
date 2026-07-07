# Clock Catalogue + Landing Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the explorer to "Clock Catalogue", replace its chip facets with multi-select searchable dropdown filters over every categorical column, make its table span the full width, and polish the landing page (hero logo, remove the Contents section, point quick-start at a tutorial).

**Architecture:** All changes are inside the Sphinx docs build. The bespoke ES5 Clock Explorer app (`clock_explorer_core.js` pure logic + `clock_explorer.js` DOM + `clock_explorer.css`) gains a top filter bar of dropdown buttons (search + checkboxes) replacing the left facet panel; the Catalogue page drops both pydata sidebars and relaxes the article width via a JS-set body class. Landing edits are RST + CSS only.

**Tech Stack:** Sphinx, pydata-sphinx-theme, sphinx_design, vanilla ES5 JavaScript (no framework), Node.js `node:assert` for the pure-logic tests.

## Global Constraints

- Naming: the explorer is renamed **Clock Catalogue** (page title + navbar); the URL stays `clock_glossary.html` (filename `clock_glossary` unchanged).
- Filter widget: multi-select dropdowns with an in-dropdown search box; logic is AND across columns, OR within a column. Facet counts are over the **full** dataset.
- Filterable columns are the **categorical** ones only: `data_type, species, platform, model_type, unit, tissue, last_author, journal, predicts, population, approved_by_author`. Numeric columns (`citations, year, n_features`) stay sortable but are never filters.
- Full-width applies to the Catalogue page **only** (scoped via a JS-set `ce-fullwidth` body class); no other page's layout changes; the no-JS fallback keeps normal width.
- No changes to `pyaging/**`, models, notebooks, or clock data (`clocks.json` / `clock_glossary.csv` content).
- JavaScript stays ES5-compatible: no arrow functions, template literals, `const`/`let`; build DOM with `document.createElement`; keep the `window`/`module.exports` dual export in the core.
- Landing hero order: logo → badge row → "pyaging" heading → tagline.

---

### Task 1: Expand core facet fields to all categorical columns

**Files:**
- Modify: `docs/_static/clock_explorer_core.js:5`
- Test: `docs/_static/tests/clock_explorer_core.test.js`

**Interfaces:**
- Consumes: nothing new.
- Produces: `ClockExplorerCore.FACET_FIELDS` is now the 11-element categorical list below; `computeFacets`, `filterClocks`, `sortClocks`, `toCSV` signatures unchanged.

- [ ] **Step 1: Update the test to cover the new categorical facets**

Replace the entire contents of `docs/_static/tests/clock_explorer_core.test.js` with:

```js
const assert = require("node:assert");
const core = require("../clock_explorer_core.js");

const data = [
  { clock_name: "b", data_type: "methylation", species: "Homo sapiens", platform: "Illumina 450K", model_type: "Elastic net", unit: "years", predicts: "chronological age", tissue: "blood", journal: "Aging Cell", population: "adult", approved_by_author: "approved", citations: 10, year: 2018, n_features: 100, last_author: "X", notes: "blood clock" },
  { clock_name: "a", data_type: "rna", species: "Mus musculus", platform: "RNA-seq", model_type: "LASSO", unit: "years", predicts: "chronological age", tissue: "liver", journal: "Nature", population: "adult", approved_by_author: "not approved", citations: 50, year: 2020, n_features: 5, last_author: "Y", notes: "liver" },
  { clock_name: "c", data_type: "methylation", species: "Homo sapiens", platform: "Illumina EPIC", model_type: "Elastic net", unit: "weeks", predicts: "gestational age", tissue: "cord blood", journal: "Aging Cell", population: "neonate", approved_by_author: "approved", citations: 5, year: 2016, n_features: 200, last_author: "Z", notes: "cord blood" },
];

const facets = core.computeFacets(data);
// All categorical columns are faceted, including the newly-added ones.
assert.deepStrictEqual(core.FACET_FIELDS, ["data_type", "species", "platform", "model_type", "unit", "tissue", "last_author", "journal", "predicts", "population", "approved_by_author"]);
assert.strictEqual(facets.data_type.length, 2);
assert.deepStrictEqual(facets.last_author.map((x) => x.value), ["X", "Y", "Z"]);
assert.strictEqual(facets.journal.find((x) => x.value === "Aging Cell").count, 2);
assert.strictEqual(facets.tissue.length, 3);
// Numeric columns are NOT faceted.
assert.strictEqual(facets.citations, undefined);
assert.strictEqual(facets.year, undefined);

// Filtering: OR within a field, AND across fields, over any categorical column.
assert.deepStrictEqual(core.filterClocks(data, { last_author: ["X"] }, "").map((c) => c.clock_name), ["b"]);
assert.deepStrictEqual(core.filterClocks(data, { data_type: ["methylation"] }, "").map((c) => c.clock_name).sort(), ["b", "c"]);
assert.strictEqual(core.filterClocks(data, { data_type: ["methylation", "rna"] }, "").length, 3);
assert.deepStrictEqual(core.filterClocks(data, { tissue: ["blood"], platform: ["Illumina 450K"] }, "").map((c) => c.clock_name), ["b"]);
assert.deepStrictEqual(core.filterClocks(data, { data_type: ["methylation"], unit: ["weeks"] }, "").map((c) => c.clock_name), ["c"]);

// Search still scans all fields.
assert.deepStrictEqual(core.filterClocks(data, {}, "LIVER").map((c) => c.clock_name), ["a"]);
assert.deepStrictEqual(core.filterClocks(data, {}, "2020").map((c) => c.clock_name), ["a"]);
assert.deepStrictEqual(core.filterClocks(data, {}, "450k").map((c) => c.clock_name), ["b"]);
assert.deepStrictEqual(core.filterClocks(data, {}, "cord blood").map((c) => c.clock_name), ["c"]);

assert.deepStrictEqual(core.sortClocks(data, "citations", "desc").map((c) => c.clock_name), ["a", "b", "c"]);
assert.deepStrictEqual(core.sortClocks(data, "clock_name", "asc").map((c) => c.clock_name), ["a", "b", "c"]);
assert.strictEqual(core.toCSV([data[0]], ["clock_name", "citations"]), "clock_name,citations\nb,10");

console.log("all clock_explorer_core tests passed");
```

- [ ] **Step 2: Run the test to verify it FAILS**

Run: `node docs/_static/tests/clock_explorer_core.test.js`
Expected: FAIL — an `AssertionError` on the `FACET_FIELDS` deep-equal (current value still includes `approved_by_author` but not `tissue/last_author/journal/predicts/population`).

- [ ] **Step 3: Expand `FACET_FIELDS`**

In `docs/_static/clock_explorer_core.js`, replace line 5:

```js
  var FACET_FIELDS = ["data_type", "species", "platform", "model_type", "unit", "approved_by_author"];
```

with:

```js
  var FACET_FIELDS = ["data_type", "species", "platform", "model_type", "unit", "tissue", "last_author", "journal", "predicts", "population", "approved_by_author"];
```

- [ ] **Step 4: Run the test to verify it PASSES**

Run: `node docs/_static/tests/clock_explorer_core.test.js`
Expected: `all clock_explorer_core tests passed`

- [ ] **Step 5: Commit**

```bash
git add docs/_static/clock_explorer_core.js docs/_static/tests/clock_explorer_core.test.js
git commit -m "docs: make every categorical column a Clock Catalogue facet"
```

---

### Task 2: Replace the chip facet panel with a dropdown filter bar

**Files:**
- Modify (full rewrite): `docs/_static/clock_explorer.js`
- Modify: `docs/_static/clock_explorer.css:19-29` and `:67-71`

**Interfaces:**
- Consumes: `ClockExplorerCore.FACET_FIELDS` (11 categorical fields from Task 1), `computeFacets`, `filterClocks`, `sortClocks`, `toCSV`.
- Produces: a `.ce-filterbar` of dropdown buttons + a `.ce-active` chip row; the controller adds the `ce-fullwidth` class to `<body>` on mount (consumed by Task 3's CSS).

- [ ] **Step 1: Rewrite `clock_explorer.js`**

Replace the entire contents of `docs/_static/clock_explorer.js` with:

```js
/* pyaging Clock Catalogue — bespoke DOM controller. Depends on clock_explorer_core.js. */
(function () {
  "use strict";

  function staticBase() {
    var s = document.currentScript || document.querySelector('script[src*="clock_explorer.js"]');
    return s ? s.src.replace(/clock_explorer\.js.*$/, "") : "_static/";
  }

  var core = window.ClockExplorerCore;
  var FACET_LABELS = {
    data_type: "Data type", species: "Species", platform: "Platform",
    model_type: "Model type", unit: "Unit", tissue: "Tissue",
    last_author: "Last author", journal: "Journal", predicts: "Predicts",
    population: "Population", approved_by_author: "Approval",
  };
  var COLUMNS = [
    { key: "clock_name", label: "Clock", def: true },
    { key: "data_type", label: "Data type", def: true },
    { key: "species", label: "Species", def: true },
    { key: "predicts", label: "Predicts", def: true },
    { key: "model_type", label: "Model", def: true },
    { key: "platform", label: "Platform", def: true },
    { key: "citations", label: "Citations", def: true, num: true },
    { key: "year", label: "Year", def: true, num: true },
    { key: "n_features", label: "N features", def: false, num: true },
    { key: "unit", label: "Unit", def: false },
    { key: "tissue", label: "Tissue", def: false },
    { key: "population", label: "Population", def: false },
    { key: "last_author", label: "Last author", def: false },
    { key: "journal", label: "Journal", def: false },
  ];
  var DETAIL_FIELDS = [
    ["predicts", "Predicts"], ["unit", "Unit"], ["tissue", "Tissue"],
    ["platform", "Platform"], ["population", "Population"], ["model_type", "Model type"],
    ["n_features", "N features"], ["year", "Year"], ["citations", "Citations"],
    ["last_author", "Last author"], ["journal", "Journal"], ["species", "Species"],
    ["data_type", "Data type"], ["approved_by_author", "Approved by author"],
  ];
  // Every sortable table column is offered in the quick-sort dropdown so a
  // column-header click always has a matching option and render() can re-sync
  // the dropdown instead of leaving it showing a stale key.
  var SORT_OPTIONS = COLUMNS.map(function (c) { return { key: c.key, label: c.label }; });

  var state = {
    clocks: [], selected: {}, search: "", sortKey: "citations", sortDir: "desc",
    view: "table", cols: null, expanded: {},
  };
  var mount, body, sortSelEl, sortDirBtn, viewToggleBtns;
  var filterBtns, activeChipsEl, checkboxIndex, openPopover = null;

  function el(tag, cls, txt) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt != null) e.textContent = txt;
    return e;
  }
  function cols() {
    if (!state.cols) { state.cols = {}; COLUMNS.forEach(function (c) { state.cols[c.key] = c.def; }); }
    return COLUMNS.filter(function (c) { return state.cols[c.key]; });
  }
  function visible() {
    return core.sortClocks(core.filterClocks(state.clocks, state.selected, state.search), state.sortKey, state.sortDir);
  }
  function fmt(v) { return v == null || v === "" ? "—" : String(v); }

  // ---------- filter bar ----------
  function boxKey(f, value) { return f + " " + value; }
  function selectedCount(f) { return (state.selected[f] || []).length; }

  function closePopover() {
    if (openPopover) { openPopover.style.display = "none"; openPopover = null; }
  }

  function updateFilterBadges() {
    filterBtns.forEach(function (fb) {
      var n = selectedCount(fb.field);
      fb.badge.textContent = n ? String(n) : "";
      fb.badge.style.display = n ? "" : "none";
      if (n) fb.btn.classList.add("active"); else fb.btn.classList.remove("active");
    });
  }

  function toggleValue(f, value, on) {
    var sel = state.selected[f] || (state.selected[f] = []);
    var i = sel.indexOf(value);
    if (on && i === -1) sel.push(value);
    else if (!on && i !== -1) sel.splice(i, 1);
    if (!sel.length) delete state.selected[f];
    updateFilterBadges();
    updateActiveChips();
    render();
  }

  function buildFilterBar() {
    var facets = core.computeFacets(state.clocks);
    filterBtns = [];
    checkboxIndex = {};
    var bar = el("div", "ce-filterbar");
    core.FACET_FIELDS.forEach(function (f) {
      var opts = facets[f];
      if (!opts || !opts.length) return;
      var wrap = el("div", "ce-filter");

      var btn = el("button", "ce-btn ce-filter-btn");
      btn.type = "button";
      btn.appendChild(document.createTextNode(FACET_LABELS[f] || f));
      btn.appendChild(el("span", "ce-filter-caret", "▾"));
      var badge = el("span", "ce-filter-badge");
      badge.style.display = "none";
      btn.appendChild(badge);

      var pop = el("div", "ce-filter-pop");
      pop.style.display = "none";
      pop.addEventListener("click", function (e) { e.stopPropagation(); });

      var search = el("input", "ce-filter-search");
      search.type = "search";
      search.placeholder = "Search " + (FACET_LABELS[f] || f).toLowerCase() + "…";
      var list = el("div", "ce-filter-list");
      opts.forEach(function (o) {
        var row = el("label", "ce-filter-opt");
        var cb = el("input");
        cb.type = "checkbox";
        cb.checked = (state.selected[f] || []).indexOf(o.value) !== -1;
        cb.addEventListener("change", function () { toggleValue(f, o.value, cb.checked); });
        checkboxIndex[boxKey(f, o.value)] = cb;
        row.appendChild(cb);
        row.appendChild(el("span", "ce-opt-label", o.value));
        row.appendChild(el("span", "ce-opt-count", String(o.count)));
        row._label = String(o.value).toLowerCase();
        list.appendChild(row);
      });
      search.addEventListener("input", function () {
        var q = search.value.trim().toLowerCase();
        for (var i = 0; i < list.childNodes.length; i++) {
          var row = list.childNodes[i];
          row.style.display = (!q || row._label.indexOf(q) !== -1) ? "" : "none";
        }
      });
      pop.appendChild(search);
      pop.appendChild(list);

      btn.addEventListener("click", function (e) {
        e.stopPropagation();
        var isOpen = pop.style.display !== "none";
        closePopover();
        if (!isOpen) { pop.style.display = "block"; openPopover = pop; search.value = ""; search.focus(); }
      });

      wrap.appendChild(btn);
      wrap.appendChild(pop);
      bar.appendChild(wrap);
      filterBtns.push({ field: f, btn: btn, badge: badge });
    });
    return bar;
  }

  function buildActiveChips() {
    activeChipsEl = el("div", "ce-active");
    updateActiveChips();
    return activeChipsEl;
  }
  function updateActiveChips() {
    if (!activeChipsEl) return;
    activeChipsEl.innerHTML = "";
    core.FACET_FIELDS.forEach(function (f) {
      (state.selected[f] || []).forEach(function (value) {
        var chip = el("span", "ce-active-chip");
        chip.appendChild(el("span", "ce-active-label", (FACET_LABELS[f] || f) + ": " + value));
        var x = el("button", "ce-active-x", "×");
        x.type = "button";
        x.title = "Remove filter";
        x.addEventListener("click", function () {
          var cb = checkboxIndex[boxKey(f, value)];
          if (cb) cb.checked = false;
          toggleValue(f, value, false);
        });
        chip.appendChild(x);
        activeChipsEl.appendChild(chip);
      });
    });
    activeChipsEl.style.display = activeChipsEl.childNodes.length ? "flex" : "none";
  }

  // ---------- toolbar ----------
  function buildToolbar() {
    var bar = el("div", "ce-toolbar");

    var search = el("input", "ce-search");
    search.type = "search";
    search.placeholder = "Search clocks…";
    search.value = state.search;
    search.addEventListener("input", function () { state.search = search.value; render(); });
    bar.appendChild(search);

    sortSelEl = el("select", "ce-sort");
    SORT_OPTIONS.forEach(function (o) {
      var opt = el("option", null, o.label); opt.value = o.key;
      if (o.key === state.sortKey) opt.selected = true;
      sortSelEl.appendChild(opt);
    });
    sortSelEl.addEventListener("change", function () { state.sortKey = sortSelEl.value; render(); });
    sortDirBtn = el("button", "ce-btn ce-sortdir", state.sortDir === "desc" ? "▼" : "▲");
    sortDirBtn.type = "button";
    sortDirBtn.title = "Toggle sort direction";
    sortDirBtn.addEventListener("click", function () {
      state.sortDir = state.sortDir === "desc" ? "asc" : "desc";
      render();
    });

    var toggle = el("div", "ce-viewtoggle");
    viewToggleBtns = [];
    ["table", "cards"].forEach(function (v) {
      var b = el("button", "ce-btn" + (state.view === v ? " active" : ""), v === "table" ? "Table" : "Cards");
      b.type = "button";
      b.addEventListener("click", function () { state.view = v; render(); });
      viewToggleBtns.push({ view: v, btn: b });
      toggle.appendChild(b);
    });

    var dl = el("button", "ce-btn", "Download CSV");
    dl.type = "button";
    dl.addEventListener("click", function () {
      var rows = visible();
      var keys = [], seen = {};
      rows.forEach(function (c) {
        for (var k in c) {
          if (c.hasOwnProperty(k) && k !== "notebook" && !seen[k]) { seen[k] = true; keys.push(k); }
        }
      });
      var csv = core.toCSV(rows, keys);
      var blob = new Blob([csv], { type: "text/csv" });
      var a = el("a"); a.href = URL.createObjectURL(blob); a.download = "pyaging_clocks.csv";
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      URL.revokeObjectURL(a.href);
    });

    var reset = el("button", "ce-btn", "Reset");
    reset.type = "button";
    reset.addEventListener("click", function () {
      state.selected = {}; state.search = ""; state.expanded = {}; buildAll();
    });

    var count = el("span", "ce-count");
    count.id = "ce-count";

    [sortSelEl, sortDirBtn, toggle, dl, reset, count].forEach(function (n) { bar.appendChild(n); });
    return bar;
  }

  // ---------- detail ----------
  function buildDetailBox(c) {
    var box = el("div", "ce-detail");
    if (c.notes) box.appendChild(el("p", "ce-notes", c.notes));
    var grid = el("dl", "ce-detail-grid");
    DETAIL_FIELDS.forEach(function (pair) {
      grid.appendChild(el("dt", null, pair[1]));
      grid.appendChild(el("dd", null, fmt(c[pair[0]])));
    });
    box.appendChild(grid);
    var links = el("div", "ce-links");
    if (c.doi) { var a = el("a", "ce-link", "Paper (DOI)"); a.href = c.doi; a.target = "_blank"; a.rel = "noopener"; links.appendChild(a); }
    if (c.notebook) { var nb = el("a", "ce-link", "Implementation notebook"); nb.href = staticBase() + "../" + c.notebook; nb.target = "_blank"; nb.rel = "noopener"; links.appendChild(nb); }
    box.appendChild(links);
    return box;
  }
  function detailPanel(c, colspan) {
    var td = el("td", "ce-detail-cell");
    if (colspan) td.colSpan = colspan;
    td.appendChild(buildDetailBox(c));
    return td;
  }

  // ---------- table ----------
  function buildTable(rows) {
    var scroller = el("div", "ce-scroll");
    var table = el("table", "ce-table");
    var thead = el("thead"), htr = el("tr");
    htr.appendChild(el("th", "ce-expander-col", ""));
    cols().forEach(function (col) {
      var th = el("th", col.num ? "ce-num" : null, col.label);
      if (state.sortKey === col.key) th.classList.add(state.sortDir === "desc" ? "sort-desc" : "sort-asc");
      th.addEventListener("click", function () {
        if (state.sortKey === col.key) state.sortDir = state.sortDir === "desc" ? "asc" : "desc";
        else { state.sortKey = col.key; state.sortDir = col.num ? "desc" : "asc"; }
        render();
      });
      htr.appendChild(th);
    });
    thead.appendChild(htr); table.appendChild(thead);

    var tbody = el("tbody");
    var span = cols().length + 1;
    rows.forEach(function (c) {
      var tr = el("tr", "ce-row");
      var exp = el("td", "ce-expander");
      exp.appendChild(el("span", "ce-caret" + (state.expanded[c.clock_name] ? " open" : ""), "▸"));
      tr.appendChild(exp);
      cols().forEach(function (col) {
        var td = el("td", col.num ? "ce-num" : null);
        td.appendChild(document.createTextNode(fmt(c[col.key])));
        tr.appendChild(td);
      });
      tr.addEventListener("click", function () {
        state.expanded[c.clock_name] = !state.expanded[c.clock_name]; render();
      });
      tbody.appendChild(tr);
      if (state.expanded[c.clock_name]) {
        var dtr = el("tr", "ce-detail-row");
        dtr.appendChild(detailPanel(c, span));
        tbody.appendChild(dtr);
      }
    });
    table.appendChild(tbody); scroller.appendChild(table);
    return scroller;
  }

  // ---------- cards ----------
  function buildCards(rows) {
    var grid = el("div", "ce-cards");
    rows.forEach(function (c) {
      var card = el("div", "ce-card");
      var head = el("div", "ce-card-head");
      head.appendChild(el("h3", "ce-card-title", c.clock_name));
      if (c.citations != null) head.appendChild(el("span", "ce-card-cites", c.citations + " cites"));
      card.appendChild(head);
      var badges = el("div", "ce-badges");
      [c.data_type, c.species, c.model_type].filter(Boolean).forEach(function (b) {
        badges.appendChild(el("span", "ce-badge", b));
      });
      card.appendChild(badges);
      card.appendChild(el("p", "ce-card-predicts", fmt(c.predicts) + (c.unit ? " (" + c.unit + ")" : "")));
      var more = el("button", "ce-btn ce-card-more", state.expanded[c.clock_name] ? "Hide details" : "Details");
      more.type = "button";
      more.addEventListener("click", function () { state.expanded[c.clock_name] = !state.expanded[c.clock_name]; render(); });
      card.appendChild(more);
      if (state.expanded[c.clock_name]) {
        card.appendChild(buildDetailBox(c));
      }
      grid.appendChild(card);
    });
    return grid;
  }

  // ---------- render ----------
  function render() {
    var rows = visible();
    var countEl = document.getElementById("ce-count");
    if (countEl) countEl.textContent = rows.length + " / " + state.clocks.length + " clocks";
    if (sortDirBtn) sortDirBtn.textContent = state.sortDir === "desc" ? "▼" : "▲";
    if (sortSelEl) {
      var hasKey = false;
      SORT_OPTIONS.forEach(function (o) { if (o.key === state.sortKey) hasKey = true; });
      if (hasKey) sortSelEl.value = state.sortKey;
    }
    if (viewToggleBtns) {
      viewToggleBtns.forEach(function (t) {
        if (t.view === state.view) t.btn.classList.add("active");
        else t.btn.classList.remove("active");
      });
    }
    body.innerHTML = "";
    body.appendChild(state.view === "cards" ? buildCards(rows) : buildTable(rows));
  }

  function buildAll() {
    mount.innerHTML = "";
    // The Catalogue page uses the full content width once its sidebars are gone;
    // Task 3's CSS keys off this body class (scoped to where the app mounts).
    document.body.classList.add("ce-fullwidth");
    var layout = el("div", "ce-root");
    layout.appendChild(buildToolbar());
    layout.appendChild(buildFilterBar());
    layout.appendChild(buildActiveChips());
    var main = el("div", "ce-main");
    body = el("div", "ce-body");
    main.appendChild(body);
    layout.appendChild(main);
    mount.appendChild(layout);
    // Hide the static no-JS/SEO fallback (intro note + full csv-table) now that
    // the live app has mounted. Done in JS so it works even without CSS :has().
    var sib = mount.nextElementSibling;
    while (sib) { sib.style.display = "none"; sib = sib.nextElementSibling; }
    render();
  }

  function init() {
    mount = document.getElementById("clock-explorer");
    if (!mount || !core) return;
    document.addEventListener("click", closePopover);
    document.addEventListener("keydown", function (e) { if (e.key === "Escape") closePopover(); });
    fetch(staticBase() + "clocks.json")
      .then(function (r) { return r.json(); })
      .then(function (data) { state.clocks = data; buildAll(); })
      .catch(function (e) {
        mount.appendChild(el("p", "ce-error", "Could not load clock data. See the table below or the GitHub repository. (" + e + ")"));
      });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
```

- [ ] **Step 2: Verify the file parses**

Run: `node --check docs/_static/clock_explorer.js`
Expected: exit 0, no output.

- [ ] **Step 3: Replace the facet-panel CSS with filter-bar CSS**

In `docs/_static/clock_explorer.css`, replace lines 19-29 (the block from `.ce-main { display: grid; ...` through `.ce-chip-count { ... }`) with:

```css
.ce-main { display: block; }

/* Filter bar — one dropdown button per categorical column, above the table. */
.ce-filterbar { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }
.ce-filter { position: relative; }
.ce-filter-btn { display: inline-flex; align-items: center; gap: 6px; }
.ce-filter-caret { opacity: 0.6; font-size: 0.75rem; }
.ce-filter-badge { display: inline-flex; min-width: 18px; height: 18px; padding: 0 5px; align-items: center;
  justify-content: center; border-radius: 999px; background: var(--ce-primary); color: #fff; font-size: 0.7rem;
  font-variant-numeric: tabular-nums; }
.ce-filter-pop { position: absolute; z-index: 30; top: calc(100% + 4px); left: 0; width: 260px; max-width: 80vw;
  background: var(--pst-color-background, #fff); border: 1px solid var(--ce-border); border-radius: 10px;
  box-shadow: 0 6px 24px rgba(11, 27, 43, 0.16); padding: 8px; }
.ce-filter-search { width: 100%; box-sizing: border-box; padding: 6px 9px; border: 1px solid var(--ce-border);
  border-radius: 7px; background: var(--pst-color-background, #fff); color: inherit; margin-bottom: 6px; }
.ce-filter-list { max-height: 260px; overflow-y: auto; display: flex; flex-direction: column; gap: 1px; }
.ce-filter-opt { display: flex; align-items: center; gap: 8px; padding: 4px 6px; border-radius: 6px; cursor: pointer; }
.ce-filter-opt:hover { background: var(--ce-surface); }
.ce-filter-opt input { margin: 0; flex: none; }
.ce-opt-label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ce-opt-count { margin-left: auto; opacity: 0.65; font-variant-numeric: tabular-nums; font-size: 0.8rem; }

/* Active-filter chips */
.ce-active { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
.ce-active-chip { display: inline-flex; align-items: center; gap: 6px; padding: 3px 4px 3px 10px; border-radius: 999px;
  background: var(--ce-surface); border: 1px solid var(--ce-border); font-size: 0.8rem; }
.ce-active-x { cursor: pointer; border: none; background: none; color: var(--ce-muted); font-size: 1rem; line-height: 1;
  padding: 0 4px; }
.ce-active-x:hover { color: var(--ce-primary); }
```

- [ ] **Step 4: Simplify the responsive block**

In `docs/_static/clock_explorer.css`, replace the media query (originally lines 67-71):

```css
@media (max-width: 820px) {
  .ce-main { grid-template-columns: 1fr; }
  .ce-facets { position: static; }
  .ce-count { margin-left: 0; }
}
```

with:

```css
@media (max-width: 820px) {
  .ce-count { margin-left: 0; }
  .ce-filter-pop { width: 220px; }
}
```

- [ ] **Step 5: Build and verify the app ships**

Run: `cd docs && source ../.venv/bin/activate && make html`
Expected: build exits 0, `[clocks] regenerated clocks.json with 173 clocks`, no new warnings.
Run: `grep -c "ce-filterbar" docs/_build/html/_static/clock_explorer.js` → `1`; `grep -c "buildFacets" docs/_build/html/_static/clock_explorer.js` → `0`.

- [ ] **Step 6: Commit**

```bash
git add docs/_static/clock_explorer.js docs/_static/clock_explorer.css
git commit -m "docs: dropdown filter bar (search + multi-select) for the Clock Catalogue"
```

---

### Task 3: Full-width Catalogue layout + rename

**Files:**
- Modify: `docs/source/conf.py:81-86` (add `html_sidebars`)
- Modify: `docs/source/clock_glossary.rst:1-7` (metadata + title + intro)
- Modify: `docs/_static/clock_explorer.css` (append `.ce-fullwidth` rules)

**Interfaces:**
- Consumes: the `ce-fullwidth` body class set by `buildAll()` in Task 2.
- Produces: the Catalogue page renders without either sidebar and with a full-width article; page title/navbar read "Clock Catalogue".

- [ ] **Step 1: Remove the left (primary) sidebar on the Catalogue page**

In `docs/source/conf.py`, immediately after the `html_context = { ... }` block (ends at line 86), add:

```python
# The Clock Catalogue owns the full width — drop its left section-nav sidebar.
html_sidebars = {"clock_glossary": []}
```

- [ ] **Step 2: Rename the page and remove the right (secondary) sidebar**

Replace the top of `docs/source/clock_glossary.rst` (lines 1-7):

```rst
Aging Clock Explorer
====================

Browse and filter every aging clock available in ``pyaging``. Use the facets to
narrow by data type, species, platform, model type, unit, and prediction target;
search by name, author, or notes; sort any column; toggle between table and card
views; and click a clock to expand its full details.
```

with:

```rst
:html_theme.sidebar_secondary.remove: true

Clock Catalogue
===============

Browse and filter every aging clock available in ``pyaging``. Filter by any
categorical column — data type, species, platform, model type, unit, tissue,
last author, journal, and more; search by name, author, or notes; sort any
column; toggle between table and card views; and click a clock to expand its
full details.
```

(The `:html_theme.sidebar_secondary.remove: true` field-list line MUST be the first line of the file, before the title.)

- [ ] **Step 3: Append the full-width CSS**

Append to the end of `docs/_static/clock_explorer.css`:

```css

/* Full-width Catalogue. buildAll() adds .ce-fullwidth to <body> when the app
   mounts, so this is scoped to the Catalogue page only (no-JS fallback keeps the
   normal width). The secondary-sidebar hide is belt-and-suspenders alongside the
   page's :html_theme.sidebar_secondary.remove metadata. */
.ce-fullwidth .bd-sidebar-secondary { display: none; }
.ce-fullwidth .bd-main .bd-content { max-width: 100%; }
.ce-fullwidth .bd-article-container { max-width: 100%; }
.ce-fullwidth .bd-article { max-width: 100%; }
```

- [ ] **Step 4: Build and verify**

Run: `cd docs && source ../.venv/bin/activate && make html`
Expected: build exits 0, no new warnings.
Run: `grep -o "<h1>Clock Catalogue" docs/_build/html/clock_glossary.html` → prints `<h1>Clock Catalogue`.
Run: `grep -c "ce-fullwidth" docs/_build/html/_static/clock_explorer.css` → `4`.
Run: `grep -c "bd-sidebar-secondary" docs/_build/html/clock_glossary.html` — note the value; if the secondary sidebar `<div>` is absent the metadata worked, and the CSS covers the case where it is present. Either way the `.ce-fullwidth` rule hides it at runtime.

- [ ] **Step 5: Commit**

```bash
git add docs/source/conf.py docs/source/clock_glossary.rst docs/_static/clock_explorer.css
git commit -m "docs: full-width Clock Catalogue page (drop both sidebars, relax width)"
```

---

### Task 4: Landing page — hero logo, remove Contents, point quick-start at the tutorial

**Files:**
- Modify (full rewrite): `docs/source/index.rst`
- Modify: `docs/_static/custom.css` (append `.pyaging-hero-logo` rules)

**Interfaces:**
- Consumes: nothing from other tasks (the "Clock Catalogue" CTA link uses the filename `clock_glossary`, unaffected by the title rename).
- Produces: cleaned landing page; hidden toctrees still power the navbar/left sidebar site-wide.

- [ ] **Step 1: Rewrite `index.rst`**

Replace the entire contents of `docs/source/index.rst` with:

```rst
.. pyaging documentation master file, created by
   sphinx-quickstart on Sun Nov 19 17:35:20 2023.
   This file is the entry point to the pyaging package documentation.

.. raw:: html

   <div class="pyaging-hero-logo-wrap"><img class="pyaging-hero-logo" src="_static/logo.png" alt="pyaging logo"></div>

.. raw:: html

   <center>

.. image:: https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat
   :target: https://pyaging.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/pyaging.svg
   :target: https://pypi.python.org/pypi/pyaging
   :alt: PyPI version

.. image:: https://img.shields.io/github/license/lucascamillomd/pyaging.svg
   :target: https://github.com/lucascamillomd/pyaging/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/badge/DOI-10.1093%2Fbioinformatics%2Fbtae200-blue.svg
   :target: https://doi.org/10.1093/bioinformatics/btae200
   :alt: DOI

.. raw:: html

   </center>

pyaging
=======

.. raw:: html

   <div class="pyaging-hero">
     <p class="tagline">GPU-accelerated biological aging clocks in Python — 170+ published clocks across DNA methylation, histone marks, ATAC-seq, RNA-seq, and blood chemistry, behind a one-line prediction API.</p>
   </div>

.. grid:: 2 2 4 4
   :gutter: 3
   :class-container: sd-text-center

   .. grid-item-card:: :octicon:`rocket;1.5em;sd-text-primary` Get started
      :link: tutorials/tutorial_dnam_illumina_human_array
      :link-type: doc

      Install pyaging and run your first prediction — the Illumina 450K/EPIC walkthrough.

   .. grid-item-card:: :octicon:`telescope;1.5em;sd-text-primary` Clock Catalogue
      :link: clock_glossary
      :link-type: doc

      Filter, sort, and search every available clock.

   .. grid-item-card:: :octicon:`beaker;1.5em;sd-text-primary` Tutorials
      :link: tutorials/index
      :link-type: doc

      End-to-end walkthroughs for each data type.

   .. grid-item-card:: :octicon:`mark-github;1.5em;sd-text-primary` GitHub
      :link: https://github.com/lucascamillomd/pyaging

      Source, issues, and contributions.

Why pyaging
-----------

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: 170+ clocks

      A comprehensive, curated collection of published aging clocks, each cross-validated against its source.

   .. grid-item-card:: Multi-omic

      DNA methylation, histone marks, ATAC-seq, RNA-seq, and blood chemistry — one consistent interface.

   .. grid-item-card:: GPU-optimized

      A PyTorch backend runs predictions on CPU or GPU with no code changes.

.. image:: ../_static/pyaging_graphical_abstract.png
   :align: center
   :alt: pyaging graphical abstract
   :class: pyaging-abstract

.. raw:: html

   <br><br>

.. toctree::
   :hidden:
   :caption: Getting Started

   installation
   clock_glossary

.. toctree::
   :hidden:
   :caption: Tutorials

   tutorials/index

.. toctree::
   :hidden:
   :caption: API Reference

   pyaging

.. toctree::
   :hidden:
   :caption: Clock implementation

   clock_implementation
```

- [ ] **Step 2: Style the hero logo**

In `docs/_static/custom.css`, find the block that begins with the comment `/* Landing hero.` (the `#pyaging > h1 { ... }` rule near line 35-38). Immediately **before** that comment, insert:

```css
/* Hero logo sits above the page title on the landing page. */
.pyaging-hero-logo-wrap { text-align: center; }
.pyaging-hero-logo { display: block; margin: 1.6rem auto 0.4rem; width: 108px; height: auto; }

```

- [ ] **Step 3: Build and verify**

Run: `cd docs && source ../.venv/bin/activate && make html`
Expected: build exits 0, no new warnings.
Run: `grep -c 'class="pyaging-hero-logo"' docs/_build/html/index.html` → `1`.
Run: `grep -c "Quick start" docs/_build/html/index.html` → `0`; `grep -c "Indices and Tables" docs/_build/html/index.html` → `0`.
Run: `grep -c "tutorial_dnam_illumina_human_array" docs/_build/html/index.html` → at least `1` (the Get started card link).
Run: `grep -o "Clock Catalogue" docs/_build/html/index.html | head -1` → prints `Clock Catalogue` (the renamed CTA card).

- [ ] **Step 4: Commit**

```bash
git add docs/source/index.rst docs/_static/custom.css
git commit -m "docs: landing hero logo; remove Contents; point Get started at the Illumina tutorial"
```

---

## Final verification (after all tasks)

- [ ] `node docs/_static/tests/clock_explorer_core.test.js` → passes.
- [ ] `node --check docs/_static/clock_explorer.js` → clean.
- [ ] `cd docs && source ../.venv/bin/activate && make clean html` → exit 0, `[clocks] regenerated clocks.json with 173 clocks`, no new warnings.
- [ ] `pytest docs/source/test_make_clock_data.py -q` → passes, and `git status` shows the committed `clocks.json`/`clock_glossary.csv` are unchanged by the test run.
- [ ] Visual sign-off via headless-Chrome screenshots served over HTTP: the Catalogue shows the dropdown filter bar (open one popover, confirm search + checkboxes + counts), active-filter chips, a full-width table with both sidebars gone; the landing shows the logo above "pyaging", no Contents/Indices section, and the Get started card pointing at the Illumina tutorial.
```
