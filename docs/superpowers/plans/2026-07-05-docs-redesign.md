# pyaging Documentation Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modernize the pyaging docs (PyData theme + logo-derived palette + a new landing page) and replace the static clocks glossary with a bespoke, dependency-free interactive Clock Explorer (faceted filters, search, sort, table/card toggle, inline detail expansion).

**Architecture:** All work is inside the existing Sphinx build. A `conf.py` `builder-inited` hook regenerates `docs/_static/clocks.json` from the public S3 aggregate metadata on every build (local + Read the Docs), falling back to the committed file. The Explorer is plain browser JS split into a pure, Node-tested core (`clock_explorer_core.js`) and a DOM controller (`clock_explorer.js`), styled by `clock_explorer.css`, mounted into the existing glossary page (URL preserved).

**Tech Stack:** Sphinx, `pydata-sphinx-theme`, `sphinx_design`, vanilla ES5-compatible JavaScript (no framework), Node.js (test runner via `node:assert`), Python 3.12 + torch + pandas (build-time data generation).

## Global Constraints

- Python floor: 3.12; torch >= 2.2.0; pandas >= 2.2.0 (already in `docs/environment.yml`).
- No changes to `pyaging/**` package code, models, clock behavior, individual clock notebooks, tutorial notebooks, or API autodoc content. Do not re-run notebooks. Do not touch S3 weights.
- All front-end assets are bundled in `docs/_static` (no CDN, CSP-safe, offline-capable).
- Explorer page URL stays `clock_glossary.html`.
- Data source (public, no credentials): `https://pyaging.s3.amazonaws.com/clocks/metadata0.1.0/all_clock_metadata.pt`.
- Palette tokens (light / dark): primary `#3a7ca5` / `#5fa8d3`; secondary `#10a0b0` / `#2bc4d4`; accent `#e0b000` / `#f0d030`; ink/navy `#0b1b2b` / `#0e1826`.
- JavaScript must be ES5-compatible (no optional chaining / arrow-only) so it runs without a build step; use IIFEs, not ES modules.
- Commit after each task. Run the docs build with the repo venv active (`source /Users/lucascamillo/pyaging/.venv/bin/activate`) so the pre-commit hook finds its tools.

---

## File Structure

| File | Responsibility |
|---|---|
| `docs/environment.yml` | Swap `sphinx-book-theme` → `pydata-sphinx-theme` (RTD deps) |
| `docs/source/conf.py` | PyData theme + options; `builder-inited` hook to regenerate data; js/css registration |
| `docs/_static/custom.css` | Site-wide palette + polish (light/dark), landing-page styles |
| `docs/source/index.rst` | New landing page from `sphinx_design` |
| `docs/source/make_clock_data.py` | Build-time generator → `clocks.json` + `clock_glossary.csv` |
| `docs/_static/clocks.json` | Committed data file (fallback + served data) |
| `docs/_static/clock_explorer_core.js` | Pure filter/sort/facet/CSV logic (Node-tested) |
| `docs/_static/clock_explorer.js` | DOM controller: toolbar, facets, table, cards, detail |
| `docs/_static/clock_explorer.css` | Explorer styling (light/dark, responsive) |
| `docs/_static/tests/clock_explorer_core.test.js` | Node unit tests for the core |
| `docs/source/clock_glossary.rst` | Retitled "Clock Explorer"; mount container + no-JS fallback |
| `docs/source/make_clock_glossary.py` | Deleted (superseded by `make_clock_data.py`) |
| `docs/_static/clock_glossary.js` | Deleted (superseded by explorer) |
| `docs/Makefile` | `data` target for local regeneration |

---

### Task 1: Clock data pipeline (`make_clock_data.py` + conf hook)

Build the data first so later front-end tasks have real `clocks.json` to render.

**Files:**
- Create: `docs/source/make_clock_data.py`
- Modify: `docs/source/conf.py` (add generator hook via `setup(app)`)
- Delete: `docs/source/make_clock_glossary.py`
- Modify: `docs/Makefile` (add `data` target)
- Test: `docs/source/test_make_clock_data.py`

**Interfaces:**
- Produces: `make_clock_data.generate() -> int` (writes `docs/_static/clocks.json` and `docs/_static/clock_glossary.csv`, returns clock count). `clocks.json` is a JSON array; each element has keys: `clock_name`, `data_type`, `species`, `predicts`, `unit`, `tissue`, `platform`, `population`, `model_type`, `n_features`, `year`, `citations`, `citations_date`, `last_author`, `journal`, `doi`, `notes`, `preprocess`, `postprocess`, `reference_values`, `approved_by_author`, `notebook`.

- [ ] **Step 1: Write the failing test**

Create `docs/source/test_make_clock_data.py`:

```python
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.normpath(os.path.join(HERE, "..", "_static"))
REQUIRED = {
    "clock_name", "data_type", "species", "predicts", "unit", "tissue",
    "platform", "population", "model_type", "n_features", "year", "citations",
    "last_author", "journal", "doi", "notes", "notebook",
}


def test_generate_writes_valid_json():
    sys.path.insert(0, HERE)
    import make_clock_data

    n = make_clock_data.generate()
    assert n >= 170, f"expected >=170 clocks, got {n}"

    with open(os.path.join(STATIC, "clocks.json"), encoding="utf-8") as fh:
        rows = json.load(fh)
    assert isinstance(rows, list) and len(rows) == n
    # required keys present on every row
    for row in rows:
        missing = REQUIRED - set(row)
        assert not missing, f"{row.get('clock_name')} missing {missing}"
    # sorted by clock_name (case-insensitive)
    names = [r["clock_name"].lower() for r in rows]
    assert names == sorted(names)
    # notebook link points into the gallery
    assert rows[0]["notebook"].startswith("clock_notebooks/")
    # JSON is serializable with no numpy leakage (all values are JSON scalars/None/list/str)
    json.dumps(rows)


def test_csv_also_written():
    assert os.path.exists(os.path.join(STATIC, "clock_glossary.csv"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/lucascamillo/pyaging && source .venv/bin/activate && pytest docs/source/test_make_clock_data.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'make_clock_data'`.

- [ ] **Step 3: Write the generator**

Create `docs/source/make_clock_data.py`:

```python
"""Build-time generator for the Clock Explorer data.

Downloads the public aggregate clock metadata from S3 and writes:
  - docs/_static/clocks.json  (array consumed by the Explorer front-end)
  - docs/_static/clock_glossary.csv  (download + no-JS fallback)
"""
import json
import os
from urllib.request import urlretrieve

import pandas as pd
import torch

URL = "https://pyaging.s3.amazonaws.com/clocks/metadata0.1.0/all_clock_metadata.pt"
STATIC = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "_static"))

FIELDS = [
    "data_type", "species", "predicts", "unit", "tissue", "platform",
    "population", "model_type", "n_features", "year", "citations",
    "citations_date", "last_author", "journal", "doi", "notes",
    "preprocess", "postprocess", "reference_values", "approved_by_author",
]


def _json_safe(o):
    if hasattr(o, "item"):          # numpy scalar
        return o.item()
    if hasattr(o, "tolist"):        # numpy array / tensor
        return o.tolist()
    return str(o)


def _shorten(v):
    # reference_values can be a long array; keep the file lean.
    if isinstance(v, (list, tuple)) and len(v) > 8:
        return "{} values".format(len(v))
    return v


def generate():
    os.makedirs(STATIC, exist_ok=True)
    pt_path = os.path.join(STATIC, "all_clock_metadata.pt")
    urlretrieve(URL, pt_path)
    meta = torch.load(pt_path, weights_only=False)

    rows = []
    for name, m in meta.items():
        row = {"clock_name": name}
        for f in FIELDS:
            row[f] = _shorten(m.get(f))
        row["notebook"] = "clock_notebooks/{}.html".format(name)
        rows.append(row)
    rows.sort(key=lambda r: r["clock_name"].lower())

    with open(os.path.join(STATIC, "clocks.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, default=_json_safe, separators=(",", ":"))

    # CSV mirror (human-friendly column order) for download + no-JS fallback
    df = pd.DataFrame(rows).drop(columns=["notebook"]).set_index("clock_name")
    df.to_csv(os.path.join(STATIC, "clock_glossary.csv"))
    return len(rows)


if __name__ == "__main__":
    print("generated {} clocks".format(generate()))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/lucascamillo/pyaging && source .venv/bin/activate && pytest docs/source/test_make_clock_data.py -q`
Expected: PASS (2 passed). Requires network to S3.

- [ ] **Step 5: Wire the generator into `conf.py` and remove the old script**

In `docs/source/conf.py`, append at the end of the file:

```python
# -- Generate Clock Explorer data at build time (local + Read the Docs) -------

def _generate_clock_data(app):
    try:
        from make_clock_data import generate

        n = generate()
        print("[clocks] regenerated clocks.json with {} clocks".format(n))
    except Exception as exc:  # noqa: BLE001 — never break the build
        print("[clocks] WARNING: using committed clocks.json ({})".format(exc))


def setup(app):
    app.connect("builder-inited", _generate_clock_data)
```

Delete the superseded script:

```bash
git rm docs/source/make_clock_glossary.py
```

In `docs/Makefile`, replace the `html` recipe's generator line and add a `data` target:

```make
html: data
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

data:
	python source/make_clock_data.py
```

- [ ] **Step 6: Regenerate and commit**

Run: `cd /Users/lucascamillo/pyaging && source .venv/bin/activate && python docs/source/make_clock_data.py`
Expected: prints `generated 173 clocks` (or current count); `docs/_static/clocks.json` updated.

```bash
git add docs/source/make_clock_data.py docs/source/test_make_clock_data.py docs/source/conf.py docs/Makefile docs/_static/clocks.json docs/_static/clock_glossary.csv
git rm --cached docs/source/make_clock_glossary.py 2>/dev/null; true
git commit -m "docs: add clocks.json build-time data pipeline for Clock Explorer"
```

---

### Task 2: Explorer pure core + Node tests

**Files:**
- Create: `docs/_static/clock_explorer_core.js`
- Test: `docs/_static/tests/clock_explorer_core.test.js`

**Interfaces:**
- Produces (global `window.ClockExplorerCore`, and `module.exports` under Node):
  - `computeFacets(clocks) -> {field: [{value, count}, ...]}` for fields `["data_type","species","platform","model_type","unit","predicts"]`, values sorted case-insensitively, blanks skipped.
  - `filterClocks(clocks, selected, search) -> clocks[]` — `selected` is `{field: string[]}`; AND across fields, OR within a field; `search` matches (case-insensitive substring) against name/last_author/notes/predicts/tissue/journal.
  - `sortClocks(clocks, key, dir) -> clocks[]` — numeric for `n_features|year|citations` (null → -Infinity), else case-insensitive string; `dir` is `"asc"|"desc"`; returns a new array.
  - `toCSV(clocks, columns) -> string` — RFC-4180 quoting.

- [ ] **Step 1: Write the failing test**

Create `docs/_static/tests/clock_explorer_core.test.js`:

```js
const assert = require("node:assert");
const core = require("../clock_explorer_core.js");

const data = [
  { clock_name: "b", data_type: "methylation", species: "Homo sapiens", platform: "Illumina 450K", model_type: "Elastic net", unit: "years", predicts: "chronological age", citations: 10, year: 2018, n_features: 100, last_author: "X", notes: "blood clock" },
  { clock_name: "a", data_type: "rna", species: "Mus musculus", platform: "RNA-seq", model_type: "LASSO", unit: "years", predicts: "chronological age", citations: 50, year: 2020, n_features: 5, last_author: "Y", notes: "liver" },
  { clock_name: "c", data_type: "methylation", species: "Homo sapiens", platform: "Illumina EPIC", model_type: "Elastic net", unit: "weeks", predicts: "gestational age", citations: 5, year: 2016, n_features: 200, last_author: "Z", notes: "cord blood" },
];

const facets = core.computeFacets(data);
assert.strictEqual(facets.data_type.length, 2);
assert.deepStrictEqual(facets.data_type.map((x) => x.value), ["methylation", "rna"]);
assert.strictEqual(facets.data_type.find((x) => x.value === "methylation").count, 2);

assert.deepStrictEqual(core.filterClocks(data, { data_type: ["methylation"] }, "").map((c) => c.clock_name).sort(), ["b", "c"]);
assert.strictEqual(core.filterClocks(data, { data_type: ["methylation", "rna"] }, "").length, 3);
assert.deepStrictEqual(core.filterClocks(data, { data_type: ["methylation"], unit: ["weeks"] }, "").map((c) => c.clock_name), ["c"]);
assert.deepStrictEqual(core.filterClocks(data, {}, "LIVER").map((c) => c.clock_name), ["a"]);

assert.deepStrictEqual(core.sortClocks(data, "citations", "desc").map((c) => c.clock_name), ["a", "b", "c"]);
assert.deepStrictEqual(core.sortClocks(data, "clock_name", "asc").map((c) => c.clock_name), ["a", "b", "c"]);

assert.strictEqual(core.toCSV([data[0]], ["clock_name", "citations"]), "clock_name,citations\nb,10");

console.log("all clock_explorer_core tests passed");
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node docs/_static/tests/clock_explorer_core.test.js`
Expected: FAIL with `Cannot find module '../clock_explorer_core.js'`.

- [ ] **Step 3: Write the core**

Create `docs/_static/clock_explorer_core.js`:

```js
/* Pure, framework-free logic for the Clock Explorer. Browser + Node. */
(function (root) {
  "use strict";

  var FACET_FIELDS = ["data_type", "species", "platform", "model_type", "unit", "predicts"];
  var NUMERIC = { n_features: true, year: true, citations: true };
  var SEARCH_FIELDS = ["clock_name", "last_author", "notes", "predicts", "tissue", "journal"];

  function computeFacets(clocks) {
    var facets = {};
    FACET_FIELDS.forEach(function (f) {
      var counts = {};
      clocks.forEach(function (c) {
        var v = c[f];
        if (v === null || v === undefined || v === "") return;
        counts[v] = (counts[v] || 0) + 1;
      });
      facets[f] = Object.keys(counts)
        .sort(function (a, b) { return a.toLowerCase().localeCompare(b.toLowerCase()); })
        .map(function (v) { return { value: v, count: counts[v] }; });
    });
    return facets;
  }

  function filterClocks(clocks, selected, search) {
    var q = (search || "").trim().toLowerCase();
    return clocks.filter(function (c) {
      for (var f in selected) {
        if (!selected.hasOwnProperty(f)) continue;
        var vals = selected[f] || [];
        if (vals.length && vals.indexOf(c[f]) === -1) return false;
      }
      if (q) {
        var hay = SEARCH_FIELDS.map(function (k) { return c[k]; })
          .filter(Boolean).join(" ").toLowerCase();
        if (hay.indexOf(q) === -1) return false;
      }
      return true;
    });
  }

  function sortClocks(clocks, key, dir) {
    var mult = dir === "desc" ? -1 : 1;
    return clocks.slice().sort(function (a, b) {
      var av = a[key], bv = b[key];
      if (NUMERIC[key]) {
        av = av == null || av === "" ? -Infinity : Number(av);
        bv = bv == null || bv === "" ? -Infinity : Number(bv);
        return (av < bv ? -1 : av > bv ? 1 : 0) * mult;
      }
      av = (av == null ? "" : String(av)).toLowerCase();
      bv = (bv == null ? "" : String(bv)).toLowerCase();
      return av.localeCompare(bv) * mult;
    });
  }

  function toCSV(clocks, columns) {
    function esc(v) {
      if (v == null) return "";
      v = String(v);
      return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
    }
    var lines = [columns.map(esc).join(",")];
    clocks.forEach(function (c) {
      lines.push(columns.map(function (col) { return esc(c[col]); }).join(","));
    });
    return lines.join("\n");
  }

  var api = {
    FACET_FIELDS: FACET_FIELDS, NUMERIC: NUMERIC,
    computeFacets: computeFacets, filterClocks: filterClocks,
    sortClocks: sortClocks, toCSV: toCSV,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.ClockExplorerCore = api;
})(typeof window !== "undefined" ? window : this);
```

- [ ] **Step 4: Run test to verify it passes**

Run: `node docs/_static/tests/clock_explorer_core.test.js`
Expected: PASS — prints `all clock_explorer_core tests passed`.

- [ ] **Step 5: Commit**

```bash
git add docs/_static/clock_explorer_core.js docs/_static/tests/clock_explorer_core.test.js
git commit -m "docs: add tested pure core for Clock Explorer (filter/sort/facets/csv)"
```

---

### Task 3: Explorer DOM controller (`clock_explorer.js`)

**Files:**
- Create: `docs/_static/clock_explorer.js`

**Interfaces:**
- Consumes: `window.ClockExplorerCore` (Task 2); a mount element `#clock-explorer` (Task 6); `clocks.json` from the `_static` dir.
- Produces: on `DOMContentLoaded`, fetches data and renders the full UI into `#clock-explorer` (replacing its no-JS fallback children).

- [ ] **Step 1: Write the controller**

Create `docs/_static/clock_explorer.js`:

```js
/* pyaging Clock Explorer — bespoke DOM controller. Depends on clock_explorer_core.js. */
(function () {
  "use strict";

  function staticBase() {
    var s = document.currentScript || document.querySelector('script[src*="clock_explorer.js"]');
    return s ? s.src.replace(/clock_explorer\.js.*$/, "") : "_static/";
  }

  var core = window.ClockExplorerCore;
  var FACET_LABELS = {
    data_type: "Data type", species: "Species", platform: "Platform",
    model_type: "Model type", unit: "Unit", predicts: "Predicts",
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
  var SORT_OPTIONS = [
    { key: "clock_name", label: "Name" }, { key: "citations", label: "Citations" },
    { key: "year", label: "Year" }, { key: "n_features", label: "N features" },
  ];

  var state = {
    clocks: [], selected: {}, search: "", sortKey: "citations", sortDir: "desc",
    view: "table", cols: null, expanded: {},
  };
  var mount, body;

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

  // ---------- facets ----------
  function buildFacets() {
    var facets = core.computeFacets(state.clocks);
    var wrap = el("aside", "ce-facets");
    core.FACET_FIELDS.forEach(function (f) {
      if (!facets[f] || !facets[f].length) return;
      var group = el("div", "ce-facet");
      group.appendChild(el("h3", null, FACET_LABELS[f]));
      var chips = el("div", "ce-chips");
      facets[f].forEach(function (o) {
        var chip = el("button", "ce-chip");
        chip.type = "button";
        chip.appendChild(el("span", "ce-chip-label", o.value));
        chip.appendChild(el("span", "ce-chip-count", String(o.count)));
        chip.addEventListener("click", function () {
          var sel = state.selected[f] || (state.selected[f] = []);
          var i = sel.indexOf(o.value);
          if (i === -1) { sel.push(o.value); chip.classList.add("active"); }
          else { sel.splice(i, 1); chip.classList.remove("active"); }
          render();
        });
        chips.appendChild(chip);
      });
      group.appendChild(chips);
      wrap.appendChild(group);
    });
    return wrap;
  }

  // ---------- toolbar ----------
  function buildToolbar() {
    var bar = el("div", "ce-toolbar");

    var search = el("input", "ce-search");
    search.type = "search";
    search.placeholder = "Search name, author, notes…";
    search.value = state.search;
    search.addEventListener("input", function () { state.search = search.value; render(); });
    bar.appendChild(search);

    var sortSel = el("select", "ce-sort");
    SORT_OPTIONS.forEach(function (o) {
      var opt = el("option", null, o.label); opt.value = o.key;
      if (o.key === state.sortKey) opt.selected = true;
      sortSel.appendChild(opt);
    });
    sortSel.addEventListener("change", function () { state.sortKey = sortSel.value; render(); });
    var sortDir = el("button", "ce-btn ce-sortdir", state.sortDir === "desc" ? "▼" : "▲");
    sortDir.type = "button";
    sortDir.title = "Toggle sort direction";
    sortDir.addEventListener("click", function () {
      state.sortDir = state.sortDir === "desc" ? "asc" : "desc";
      sortDir.textContent = state.sortDir === "desc" ? "▼" : "▲"; render();
    });

    var toggle = el("div", "ce-viewtoggle");
    ["table", "cards"].forEach(function (v) {
      var b = el("button", "ce-btn" + (state.view === v ? " active" : ""), v === "table" ? "Table" : "Cards");
      b.type = "button";
      b.addEventListener("click", function () { state.view = v; render(); });
      toggle.appendChild(b);
    });

    var dl = el("button", "ce-btn", "Download CSV");
    dl.type = "button";
    dl.addEventListener("click", function () {
      var csv = core.toCSV(visible(), COLUMNS.map(function (c) { return c.key; }));
      var blob = new Blob([csv], { type: "text/csv" });
      var a = el("a"); a.href = URL.createObjectURL(blob); a.download = "pyaging_clocks.csv";
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
    });

    var reset = el("button", "ce-btn", "Reset");
    reset.type = "button";
    reset.addEventListener("click", function () {
      state.selected = {}; state.search = ""; state.expanded = {}; buildAll();
    });

    var count = el("span", "ce-count");
    count.id = "ce-count";

    [sortSel, sortDir, toggle, dl, reset, count].forEach(function (n) { bar.appendChild(n); });
    return bar;
  }

  // ---------- detail ----------
  function detailPanel(c, colspan) {
    var td = el("td", "ce-detail-cell");
    if (colspan) td.colSpan = colspan;
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
    if (c.notebook) { var nb = el("a", "ce-link", "Implementation notebook"); nb.href = staticBase() + "../" + c.notebook; nb.appendChild(document.createTextNode("")); links.appendChild(nb); }
    box.appendChild(links);
    td.appendChild(box);
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
        var box = el("div", "ce-detail");
        if (c.notes) box.appendChild(el("p", "ce-notes", c.notes));
        var grid2 = el("dl", "ce-detail-grid");
        DETAIL_FIELDS.forEach(function (pair) { grid2.appendChild(el("dt", null, pair[1])); grid2.appendChild(el("dd", null, fmt(c[pair[0]]))); });
        box.appendChild(grid2);
        if (c.doi) { var a = el("a", "ce-link", "Paper (DOI)"); a.href = c.doi; a.target = "_blank"; a.rel = "noopener"; box.appendChild(a); }
        card.appendChild(box);
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
    body.innerHTML = "";
    body.appendChild(state.view === "cards" ? buildCards(rows) : buildTable(rows));
  }

  function buildAll() {
    mount.innerHTML = "";
    var layout = el("div", "ce-root");
    layout.appendChild(buildToolbar());
    var main = el("div", "ce-main");
    main.appendChild(buildFacets());
    body = el("div", "ce-body");
    main.appendChild(body);
    layout.appendChild(main);
    mount.appendChild(layout);
    render();
  }

  function init() {
    mount = document.getElementById("clock-explorer");
    if (!mount || !core) return;
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

- [ ] **Step 2: Sanity-check syntax**

Run: `node --check docs/_static/clock_explorer.js`
Expected: no output, exit 0 (valid syntax).

- [ ] **Step 3: Commit**

```bash
git add docs/_static/clock_explorer.js
git commit -m "docs: add Clock Explorer DOM controller (facets, table/cards, detail)"
```

---

### Task 4: Explorer styles (`clock_explorer.css`)

**Files:**
- Create: `docs/_static/clock_explorer.css`

**Interfaces:**
- Consumes: DOM class names emitted by Task 3 (`ce-*`). Uses palette CSS variables from Task 5's `custom.css` with fallbacks so it works standalone.

- [ ] **Step 1: Write the stylesheet**

Create `docs/_static/clock_explorer.css`:

```css
/* Clock Explorer styling. Reads palette vars (custom.css) with fallbacks. */
.ce-root { --ce-primary: var(--pst-color-primary, #3a7ca5); --ce-border: var(--pst-color-border, #e2e8f0);
  --ce-surface: var(--pst-color-surface, #f6f8fa); --ce-muted: var(--pst-color-text-muted, #6b7a8d);
  font-size: 0.9rem; }

.ce-toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 14px; }
.ce-search { flex: 1 1 240px; min-width: 200px; padding: 8px 12px; border: 1px solid var(--ce-border);
  border-radius: 8px; background: var(--pst-color-background, #fff); color: inherit; }
.ce-btn { padding: 7px 12px; border: 1px solid var(--ce-border); border-radius: 8px; background: var(--ce-surface);
  color: inherit; cursor: pointer; line-height: 1; }
.ce-btn:hover { border-color: var(--ce-primary); }
.ce-btn.active { background: var(--ce-primary); color: #fff; border-color: var(--ce-primary); }
.ce-viewtoggle { display: inline-flex; }
.ce-viewtoggle .ce-btn:first-child { border-radius: 8px 0 0 8px; }
.ce-viewtoggle .ce-btn:last-child { border-radius: 0 8px 8px 0; margin-left: -1px; }
.ce-sort { padding: 7px 10px; border: 1px solid var(--ce-border); border-radius: 8px; background: var(--ce-surface); color: inherit; }
.ce-count { margin-left: auto; color: var(--ce-muted); font-size: 0.85rem; }

.ce-main { display: grid; grid-template-columns: 220px 1fr; gap: 18px; align-items: start; }
.ce-facets { position: sticky; top: 80px; display: flex; flex-direction: column; gap: 16px; }
.ce-facet h3 { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--ce-muted); margin: 0 0 6px; }
.ce-chips { display: flex; flex-wrap: wrap; gap: 5px; }
.ce-chip { display: inline-flex; align-items: center; gap: 6px; padding: 3px 9px; border: 1px solid var(--ce-border);
  border-radius: 999px; background: var(--pst-color-background, #fff); color: inherit; cursor: pointer; font-size: 0.8rem; }
.ce-chip:hover { border-color: var(--ce-primary); }
.ce-chip.active { background: var(--ce-primary); color: #fff; border-color: var(--ce-primary); }
.ce-chip-count { opacity: 0.7; font-variant-numeric: tabular-nums; }

.ce-scroll { overflow-x: auto; border: 1px solid var(--ce-border); border-radius: 10px; }
.ce-table { width: 100%; border-collapse: collapse; white-space: nowrap; }
.ce-table th, .ce-table td { padding: 9px 12px; text-align: left; border-bottom: 1px solid var(--ce-border); }
.ce-table thead th { position: sticky; top: 0; background: var(--ce-surface); cursor: pointer; user-select: none;
  font-weight: 600; z-index: 1; }
.ce-table thead th.sort-asc::after { content: " ▲"; color: var(--ce-primary); }
.ce-table thead th.sort-desc::after { content: " ▼"; color: var(--ce-primary); }
.ce-num { text-align: right; font-variant-numeric: tabular-nums; }
.ce-row { cursor: pointer; }
.ce-row:hover { background: var(--ce-surface); }
.ce-expander-col { width: 26px; }
.ce-caret { display: inline-block; transition: transform 0.15s ease; color: var(--ce-muted); }
.ce-caret.open { transform: rotate(90deg); color: var(--ce-primary); }

.ce-detail-row td { background: var(--ce-surface); }
.ce-detail { padding: 6px 4px 10px; white-space: normal; }
.ce-notes { margin: 0 0 10px; max-width: 70ch; }
.ce-detail-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 4px 18px; margin: 0; }
.ce-detail-grid dt { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.03em; color: var(--ce-muted); }
.ce-detail-grid dd { margin: 0 0 6px; }
.ce-links { margin-top: 10px; display: flex; gap: 14px; }
.ce-link { color: var(--ce-primary); font-weight: 600; text-decoration: none; }
.ce-link:hover { text-decoration: underline; }

.ce-cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 14px; }
.ce-card { border: 1px solid var(--ce-border); border-radius: 12px; padding: 14px; background: var(--pst-color-background, #fff);
  transition: box-shadow 0.15s ease, border-color 0.15s ease; }
.ce-card:hover { box-shadow: 0 4px 18px rgba(11, 27, 43, 0.08); border-color: var(--ce-primary); }
.ce-card-head { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }
.ce-card-title { margin: 0; font-size: 1rem; word-break: break-word; }
.ce-card-cites { color: var(--ce-muted); font-size: 0.8rem; white-space: nowrap; }
.ce-badges { display: flex; flex-wrap: wrap; gap: 5px; margin: 8px 0; }
.ce-badge { padding: 2px 8px; border-radius: 999px; background: var(--ce-surface); font-size: 0.72rem; }
.ce-card-predicts { margin: 6px 0 10px; color: var(--ce-muted); }
.ce-error { padding: 12px; border: 1px solid var(--ce-border); border-radius: 8px; }

@media (max-width: 820px) {
  .ce-main { grid-template-columns: 1fr; }
  .ce-facets { position: static; }
  .ce-count { margin-left: 0; }
}
```

- [ ] **Step 2: Commit**

```bash
git add docs/_static/clock_explorer.css
git commit -m "docs: add Clock Explorer styles"
```

---

### Task 5: Theme migration + logo palette (`conf.py`, `environment.yml`, `custom.css`)

**Files:**
- Modify: `docs/environment.yml`
- Modify: `docs/source/conf.py` (theme block only; keep Task 1's `setup`/hook)
- Rewrite: `docs/_static/custom.css` (site-wide theme; explorer/glossary-specific rules removed — they live in `clock_explorer.css`)

**Interfaces:**
- Produces: PyData theme active; palette CSS variables consumed by `clock_explorer.css` (`--pst-color-primary`, `--pst-color-border`, `--pst-color-surface`, `--pst-color-text-muted`, `--pst-color-background`).

- [ ] **Step 1: Update dependencies**

In `docs/environment.yml`, replace the line `- sphinx-book-theme>=1.1.0` with:

```yaml
      - pydata-sphinx-theme>=0.15.4
```

Install into the local venv:

Run: `cd /Users/lucascamillo/pyaging && source .venv/bin/activate && pip install "pydata-sphinx-theme>=0.15.4"`
Expected: installs successfully.

- [ ] **Step 2: Update the theme block in `conf.py`**

Replace the current HTML-output block (`html_theme` through `html_js_files`) in `docs/source/conf.py` with:

```python
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/lucascamillomd/pyaging",
    "icon_links": [
        {"name": "PyPI", "url": "https://pypi.org/project/pyaging/", "icon": "fa-brands fa-python"},
        {"name": "Paper", "url": "https://doi.org/10.1093/bioinformatics/btae200", "icon": "fa-solid fa-book-open"},
    ],
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_align": "left",
    "show_prev_next": False,
    "navigation_with_keys": False,
    "pygments_light_style": "friendly",
    "pygments_dark_style": "monokai",
    "header_links_before_dropdown": 6,
}
html_context = {"default_mode": "auto"}
html_logo = "../_static/logo.png"
html_css_files = ["custom.css", "clock_explorer.css"]
html_js_files = ["clock_explorer_core.js", "clock_explorer.js"]
```

- [ ] **Step 3: Rewrite `custom.css` for the palette + polish**

Replace the entire contents of `docs/_static/custom.css` with:

```css
/* pyaging docs — logo-derived palette + polish (light + dark). */
:root {
  --pst-color-primary: #3a7ca5;
  --pst-color-secondary: #10a0b0;
  --pst-color-accent: #e0b000;
  --pst-color-link: #2f6f96;
  --pst-color-link-hover: #10a0b0;
  --pst-color-inline-code: #2f6f96;
  --pst-color-surface: #f6f8fa;
  --pst-color-text-muted: #5b6b7c;
  --pst-heading-color: #0b1b2b;
  --pst-font-family-base: "Inter", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
html[data-theme="dark"] {
  --pst-color-primary: #5fa8d3;
  --pst-color-secondary: #2bc4d4;
  --pst-color-accent: #f0d030;
  --pst-color-link: #5fa8d3;
  --pst-color-link-hover: #2bc4d4;
  --pst-color-surface: #14202e;
  --pst-color-text-muted: #9fb0c0;
  --pst-heading-color: #e8eef4;
  --pst-color-background: #0e1826;
}

/* Logo sizing */
.navbar-brand img, .sidebar-logo { max-height: 40px; width: auto; }

/* Sleeker cards/buttons from sphinx_design */
.sd-card { border-radius: 12px; border: 1px solid var(--pst-color-border, #e2e8f0);
  transition: box-shadow 0.15s ease, transform 0.15s ease; }
.sd-card:hover { box-shadow: 0 6px 22px rgba(11, 27, 43, 0.10); transform: translateY(-2px); }
.bd-content .sd-btn-primary { background: var(--pst-color-primary); border-color: var(--pst-color-primary); }

/* Landing hero */
.pyaging-hero { text-align: center; padding: 2.5rem 1rem 1rem; }
.pyaging-hero h1 { font-size: 2.6rem; margin: 0.4rem 0; color: var(--pst-heading-color); }
.pyaging-hero .tagline { font-size: 1.15rem; color: var(--pst-color-text-muted); max-width: 60ch; margin: 0 auto 1.2rem; }
.pyaging-stats { display: flex; justify-content: center; gap: 2.2rem; flex-wrap: wrap; margin: 1.4rem 0; }
.pyaging-stat { text-align: center; }
.pyaging-stat .num { font-size: 1.8rem; font-weight: 700; color: var(--pst-color-primary); display: block; }
.pyaging-stat .lbl { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--pst-color-text-muted); }

/* Tighten default heading weights for a cleaner look */
h1, h2, h3 { letter-spacing: -0.01em; }
```

- [ ] **Step 4: Build and verify the theme**

Run: `cd /Users/lucascamillo/pyaging/docs && source ../.venv/bin/activate && make html 2>&1 | tail -5`
Expected: build completes; `The HTML pages are in _build/html`.

Run: `grep -o "pydata-sphinx-theme" docs/_build/html/index.html | head -1`
Expected: prints `pydata-sphinx-theme` (theme active).

- [ ] **Step 5: Commit**

```bash
git add docs/environment.yml docs/source/conf.py docs/_static/custom.css
git commit -m "docs: migrate to pydata-sphinx-theme with logo-derived palette"
```

---

### Task 6: Explorer page wiring + remove old glossary JS

**Files:**
- Modify: `docs/source/clock_glossary.rst`
- Delete: `docs/_static/clock_glossary.js`

**Interfaces:**
- Consumes: `#clock-explorer` mount (Task 3), `clocks.json` (Task 1), assets registered in Task 5.

- [ ] **Step 1: Rewrite the Explorer page**

Replace the contents of `docs/source/clock_glossary.rst` with:

```rst
Clock Explorer
==============

Browse and filter every aging clock available in ``pyaging``. Use the facets to
narrow by data type, species, platform, model type, unit, and prediction target;
search by name, author, or notes; sort any column; toggle between table and card
views; and click a clock to expand its full details.

.. raw:: html

   <div id="clock-explorer">
     <noscript>
       <p>Enable JavaScript for the interactive explorer, or
       <a href="_static/clock_glossary.csv">download the full table as CSV</a>.</p>
     </noscript>
   </div>

.. only:: html

   The data below is a static fallback rendered without JavaScript.

.. csv-table::
   :file: ../_static/clock_glossary.csv
   :header-rows: 1
   :class: ce-fallback
```

- [ ] **Step 2: Hide the static fallback once the app mounts**

Append to `docs/_static/clock_explorer.css`:

```css
/* Hide the RST csv-table fallback when the Explorer has mounted. */
#clock-explorer:not(:empty) ~ .ce-fallback,
#clock-explorer:not(:empty) ~ p { display: none; }
```

- [ ] **Step 3: Remove the superseded glossary script**

```bash
git rm docs/_static/clock_glossary.js
```

- [ ] **Step 4: Build and verify the Explorer mounts**

Run: `cd /Users/lucascamillo/pyaging/docs && source ../.venv/bin/activate && make html 2>&1 | tail -3`
Expected: build completes.

Run: `grep -c 'id="clock-explorer"' docs/_build/html/clock_glossary.html && ls docs/_build/html/_static/clock_explorer.js docs/_build/html/_static/clocks.json`
Expected: `1` and both files listed (assets copied).

- [ ] **Step 5: Commit**

```bash
git add docs/source/clock_glossary.rst docs/_static/clock_explorer.css
git rm --cached docs/_static/clock_glossary.js 2>/dev/null; true
git commit -m "docs: mount Clock Explorer on the glossary page with no-JS fallback"
```

---

### Task 7: New landing page (`index.rst`)

**Files:**
- Modify: `docs/source/index.rst`

**Interfaces:**
- Consumes: `sphinx_design` (already an extension), hero styles from `custom.css` (Task 5).

- [ ] **Step 1: Rewrite the landing page**

Replace the body of `docs/source/index.rst` (keep the top comment and the trailing `toctree` blocks unchanged) so the content between the title and the `Contents` toctrees is:

```rst
.. raw:: html

   <div class="pyaging-hero">
     <h1>pyaging</h1>
     <p class="tagline">GPU-accelerated biological aging clocks in Python — 170+ published clocks across DNA methylation, histone marks, ATAC-seq, RNA-seq, and blood chemistry, behind a one-line prediction API.</p>
   </div>

.. grid:: 2 2 4 4
   :gutter: 3
   :class-container: sd-text-center

   .. grid-item-card:: :octicon:`rocket;1.5em;sd-text-primary` Get started
      :link: installation
      :link-type: doc

      Install pyaging and predict ages in a few lines.

   .. grid-item-card:: :octicon:`telescope;1.5em;sd-text-primary` Clock Explorer
      :link: clock_glossary
      :link-type: doc

      Filter, sort, and search every available clock.

   .. grid-item-card:: :octicon:`beaker;1.5em;sd-text-primary` Tutorials
      :link: tutorials/tutorial_dnam_illumina_human_array
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

Quick start
-----------

.. code-block:: bash

   pip install pyaging

.. code-block:: python

   import pyaging as pya

   adata = pya.data.download_example_data("GSE139307")
   pya.pred.predict_age(adata, ["horvath2013", "grimage2", "pcphenoage"])
   adata.obs.head()

.. image:: ../_static/pyaging_graphical_abstract.png
   :align: center
   :alt: pyaging graphical abstract
   :class: only-light

.. raw:: html

   <br><br>
```

Keep the existing `Contents`/`toctree` sections below this content unchanged.

- [ ] **Step 2: Build and verify the landing page**

Run: `cd /Users/lucascamillo/pyaging/docs && source ../.venv/bin/activate && make html 2>&1 | tail -3`
Expected: build completes.

Run: `grep -c "pyaging-hero" docs/_build/html/index.html && grep -c "sd-card" docs/_build/html/index.html`
Expected: `1` (hero present) and a positive count (cards rendered).

- [ ] **Step 3: Commit**

```bash
git add docs/source/index.rst
git commit -m "docs: modern landing page with hero + feature cards + quick start"
```

---

### Task 8: Full build verification + visual sign-off

**Files:** none created; verification + fixups only.

- [ ] **Step 1: Clean rebuild**

Run: `cd /Users/lucascamillo/pyaging/docs && source ../.venv/bin/activate && make clean && make html 2>&1 | tee /tmp/pyaging_docs_build.log | tail -6`
Expected: build completes with no new ERROR lines.

- [ ] **Step 2: Verify no new build errors**

Run: `grep -iE "^.*(ERROR|CRITICAL)" /tmp/pyaging_docs_build.log | grep -v "toctree" | head`
Expected: no output (no errors). Investigate and fix any that appear.

- [ ] **Step 3: Verify all key pages + assets exist**

Run:
```bash
cd /Users/lucascamillo/pyaging/docs/_build/html
for f in index.html clock_glossary.html installation.html pyaging.html; do test -f "$f" && echo "ok $f" || echo "MISSING $f"; done
for a in _static/clocks.json _static/clock_explorer.js _static/clock_explorer_core.js _static/clock_explorer.css _static/custom.css; do test -f "$a" && echo "ok $a" || echo "MISSING $a"; done
```
Expected: all `ok`.

- [ ] **Step 4: Re-run the core unit tests and data test**

Run: `cd /Users/lucascamillo/pyaging && source .venv/bin/activate && node docs/_static/tests/clock_explorer_core.test.js && pytest docs/source/test_make_clock_data.py -q`
Expected: core prints pass; pytest `2 passed`.

- [ ] **Step 5: Visual sign-off**

Send the rendered `docs/_build/html/clock_glossary.html` and `index.html` to the user (SendUserFile) for a look, and iterate on palette/spacing/labels per feedback. Only proceed once the user is happy.

- [ ] **Step 6: Final commit + push**

```bash
cd /Users/lucascamillo/pyaging && source .venv/bin/activate
git add -A
git commit -m "docs: verify redesign build; refresh generated clocks.json"
git push origin main
```

---

## Self-Review

**Spec coverage (spec §):**
- §4.1 theme → Task 5. §3 palette → Task 5 (custom.css) consumed by Task 4.
- §4.2 landing → Task 7. §4.3 explorer data → Task 1; pure logic → Task 2; UI → Task 3; CSS → Task 4; page mount + no-JS fallback → Task 6.
- §5 build/data flow (conf hook, Makefile, js/css files, rename generator, remove old js) → Tasks 1, 5, 6.
- §6 testing (Node unit tests, data integrity, build, visual) → Tasks 2, 1, 8.
- §5 file map — every file accounted for across tasks.

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to". Every code step ships complete code. Reference-value bloat handled explicitly (`_shorten`). `clocks.json` fetch failure handled (`.catch`). Build-time S3 failure handled (`conf.py` hook try/except + committed fallback).

**Type consistency:** `generate()` writes the keys listed in Task 1 Interfaces; Task 2 `filterClocks/sortClocks/computeFacets/toCSV` signatures match Task 3 call sites (`core.filterClocks(clocks, selected, search)`, `core.sortClocks(rows, key, dir)`, `core.toCSV(rows, columns)`, `core.computeFacets(clocks)`, `core.FACET_FIELDS`). `COLUMNS`/`DETAIL_FIELDS` keys are a subset of the keys `generate()` emits. Facet fields in core (`data_type, species, platform, model_type, unit, predicts`) match `FACET_LABELS` in the controller.

**Note on ordering:** Task 1 (data) → Task 2 (core+tests) → Task 3 (controller) → Task 4 (css) → Task 5 (theme+palette+custom.css) → Task 6 (page mount) → Task 7 (landing) → Task 8 (verify). Each task ends with an independently testable/committable deliverable.
