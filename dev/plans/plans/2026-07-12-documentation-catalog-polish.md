# Documentation Catalogue Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the landing logo circular and reorganize the Clock Catalogue around approved-first alphabetical discovery and compact-first table columns.

**Architecture:** Keep the change inside the existing Sphinx/vanilla-JavaScript documentation frontend. Add one pure default-order comparator to the catalogue core, let the browser controller distinguish that default from explicit single-column sorts, and use semantic column classes for compact sizing and approval presentation.

**Tech Stack:** Sphinx, pydata-sphinx-theme, vanilla ES5 browser JavaScript, Node `assert`, CSS, pytest, uv

---

## File map

- Modify `docs/_static/clock_explorer_core.js`: expose the pure approved-first/name default ordering function.
- Modify `docs/_static/tests/clock_explorer_core.test.js`: specify default ordering and retain explicit-sort behavior.
- Modify `docs/_static/clock_explorer.js`: reorder/show columns, use default ordering, add semantic cell classes and approval output, and improve sort-control labeling.
- Modify `docs/_static/clock_explorer.css`: compact column widths, long-column overflow behavior, and approval status styling.
- Modify `docs/_static/custom.css`: circular landing-logo clipping.
- Verify `docs/source/test_make_clock_data.py`: ensure generated artifacts already follow the browser default.

### Task 1: Specify and implement approved-first default ordering

**Files:**
- Modify: `docs/_static/tests/clock_explorer_core.test.js`
- Modify: `docs/_static/clock_explorer_core.js`

- [ ] **Step 1: Add the failing default-order test**

Add a mixed-case fixture assertion after the existing explicit sort assertions:

```javascript
const defaultOrdered = core.defaultOrder([
  { clock_name: "Zulu", approved_by_author: "not approved" },
  { clock_name: "beta", approved_by_author: "approved" },
  { clock_name: "Alpha", approved_by_author: "approved" },
  { clock_name: "aardvark", approved_by_author: "not approved" },
]);
assert.deepStrictEqual(defaultOrdered.map((c) => c.clock_name), ["Alpha", "beta", "aardvark", "Zulu"]);
```

- [ ] **Step 2: Run the Node test and confirm the new assertion fails**

Run: `node docs/_static/tests/clock_explorer_core.test.js`

Expected: failure reporting that `core.defaultOrder` is not a function.

- [ ] **Step 3: Implement the pure comparator**

Add this function to `clock_explorer_core.js` and export it on `api`:

```javascript
function defaultOrder(clocks) {
  return clocks.slice().sort(function (a, b) {
    var aApproved = String(a.approved_by_author || "").toLowerCase() === "approved" ? 0 : 1;
    var bApproved = String(b.approved_by_author || "").toLowerCase() === "approved" ? 0 : 1;
    if (aApproved !== bApproved) return aApproved - bApproved;
    return String(a.clock_name || "").toLowerCase().localeCompare(String(b.clock_name || "").toLowerCase());
  });
}
```

Add `defaultOrder: defaultOrder` to the exported object.

- [ ] **Step 4: Run the Node test and confirm it passes**

Run: `node docs/_static/tests/clock_explorer_core.test.js`

Expected: `all clock_explorer_core tests passed`.

- [ ] **Step 5: Commit the core behavior**

Run:

```bash
uv sync --group dev
uv run pre-commit run --files docs/_static/clock_explorer_core.js docs/_static/tests/clock_explorer_core.test.js
git add docs/_static/clock_explorer_core.js docs/_static/tests/clock_explorer_core.test.js
git commit -m "docs: default catalogue to approved clocks first"
```

`pre-commit` is already a locked project dependency in `pyproject.toml`; syncing the uv environment should provide the missing executable, so no dependency-file edit is expected.

### Task 2: Reorganize catalogue columns and controller behavior

**Files:**
- Modify: `docs/_static/clock_explorer.js`

- [ ] **Step 1: Replace the column definitions with compact-first order**

Use the following `COLUMNS` sequence, making every requested field visible and assigning semantic width classes:

```javascript
var COLUMNS = [
  { key: "clock_name", label: "Clock", def: true, cls: "ce-col-clock" },
  { key: "approved_by_author", label: "Approval", def: true, cls: "ce-col-approval" },
  { key: "data_type", label: "Data type", def: true, cls: "ce-col-short" },
  { key: "species", label: "Species", def: true, cls: "ce-col-short" },
  { key: "year", label: "Year", def: true, num: true, cls: "ce-col-num" },
  { key: "citations", label: "Citations", def: true, num: true, cls: "ce-col-num" },
  { key: "n_features", label: "N features", def: true, num: true, cls: "ce-col-num" },
  { key: "unit", label: "Unit", def: true, cls: "ce-col-short" },
  { key: "model_type", label: "Model", def: true, cls: "ce-col-long" },
  { key: "platform", label: "Platform", def: true, cls: "ce-col-long" },
  { key: "predicts", label: "Predicts", def: true, cls: "ce-col-long" },
  { key: "tissue", label: "Tissue", def: true, cls: "ce-col-long" },
  { key: "population", label: "Population", def: true, cls: "ce-col-long" },
  { key: "last_author", label: "Last author", def: true, cls: "ce-col-long" },
  { key: "journal", label: "Journal", def: true, cls: "ce-col-long" },
];
```

- [ ] **Step 2: Represent the default ordering honestly in state and controls**

Add `{ key: "default", label: "Approved first, then name" }` before the column-derived sort options, initialize `sortKey: "default"` and `sortDir: "asc"`, and update `visible()` to call `core.defaultOrder(filtered)` when `state.sortKey === "default"`; otherwise retain `core.sortClocks`.

When a column header is clicked, retain the current explicit-sort behavior. The direction button remains available for explicit sorts; for the default option it does not reverse the approved-first grouping.

- [ ] **Step 3: Add semantic classes and approval rendering**

When building headers and cells, combine `ce-num` with each column's `cls`. For `approved_by_author`, append a status span instead of raw text:

```javascript
function approvalBadge(value) {
  var approved = String(value || "").toLowerCase() === "approved";
  return el("span", "ce-approval " + (approved ? "is-approved" : "is-pending"), approved ? "Approved" : "Not approved");
}
```

Set an accessible label on the direction button (`aria-label` and `title`), updating it whenever direction changes. Stop table-row expansion when the user is selecting/copying text only if the existing click flow makes this necessary during browser verification; do not broaden scope otherwise.

- [ ] **Step 4: Run syntax and logic checks**

Run:

```bash
node --check docs/_static/clock_explorer.js
node docs/_static/tests/clock_explorer_core.test.js
```

Expected: JavaScript syntax check exits 0 and core tests print their pass message.

- [ ] **Step 5: Commit the controller change**

Run:

```bash
uv run pre-commit run --files docs/_static/clock_explorer.js
git add docs/_static/clock_explorer.js
git commit -m "docs: prioritize compact catalogue columns"
```

### Task 3: Apply circular logo and catalogue presentation styles

**Files:**
- Modify: `docs/_static/custom.css`
- Modify: `docs/_static/clock_explorer.css`

- [ ] **Step 1: Clip the landing logo to its circular artwork**

Replace the landing logo rule with:

```css
.pyaging-hero-logo {
  display: block;
  margin: 1.6rem auto 0.4rem;
  width: 108px;
  height: 108px;
  border-radius: 50%;
  object-fit: cover;
}
```

- [ ] **Step 2: Add compact column and approval styles**

Add scoped rules to `clock_explorer.css`:

```css
.ce-table .ce-col-clock { min-width: 15rem; font-weight: 600; }
.ce-table .ce-col-approval,
.ce-table .ce-col-num,
.ce-table .ce-col-short { width: 1%; }
.ce-table .ce-col-long { min-width: 11rem; }
.ce-approval { display: inline-flex; align-items: center; border-radius: 999px; padding: 2px 8px; font-size: 0.72rem; font-weight: 600; }
.ce-approval.is-approved { color: #166534; background: #dcfce7; }
.ce-approval.is-pending { color: var(--ce-muted); background: var(--ce-surface); border: 1px solid var(--ce-border); }
html[data-theme="dark"] .ce-approval.is-approved { color: #bbf7d0; background: #14532d; }
```

Preserve horizontal table scrolling and sticky headers. Adjust exact minimum widths during visual verification only when needed to meet the compact-first design.

- [ ] **Step 3: Run formatting hooks and build the docs**

Run:

```bash
uv run pre-commit run --files docs/_static/custom.css docs/_static/clock_explorer.css
uv run sphinx-build -W --keep-going -b html docs/source docs/_build/html
```

Expected: hooks exit 0; Sphinx exits 0 without warnings promoted to errors.

- [ ] **Step 4: Commit the presentation change**

Run:

```bash
git add docs/_static/custom.css docs/_static/clock_explorer.css
git commit -m "docs: polish landing logo and catalogue density"
```

### Task 4: Full verification and visual QA

**Files:**
- Verify: `docs/_build/html/index.html`
- Verify: `docs/_build/html/clock_glossary.html`
- Verify: `docs/source/test_make_clock_data.py`

- [ ] **Step 1: Run all focused automated checks from a clean build**

Run:

```bash
node docs/_static/tests/clock_explorer_core.test.js
uv run pytest docs/source/test_make_clock_data.py -q
rm -rf docs/_build/html
uv run sphinx-build -W --keep-going -b html docs/source docs/_build/html
uv run pre-commit run --all-files
```

Expected: Node pass message; 2 pytest tests pass; Sphinx exits 0; all hooks pass. If unrelated existing warnings prevent `-W`, rerun without `-W`, record the exact warnings, and ensure none originate from changed files.

- [ ] **Step 2: Serve and inspect the built documentation**

Run `uv run python -m http.server 8765 --directory docs/_build/html` and use browser inspection at approximately 1440×900 and 390×844.

Check:

- landing logo is a seamless circle in light and dark themes;
- initial catalogue rows are approved and alphabetized, followed by alphabetized unapproved rows;
- the sort dropdown initially says `Approved first, then name`;
- clicking Name, Approval, Citations, and Year headers explicitly sorts correctly;
- compact fields appear before Model/Platform/Predicts/Tissue/Population/Last author/Journal;
- long fields sit to the right and horizontal scrolling remains usable;
- approval badges remain readable in both themes;
- filters, reset, CSV download, card view, and row expansion still work;
- narrow layouts do not overflow the page outside the intended table scroller.

- [ ] **Step 3: Review the final diff against the approved spec**

Run:

```bash
git diff HEAD~3 -- docs/_static docs/source/test_make_clock_data.py
git status --short
```

Confirm every approved requirement is represented and there are no unrelated modifications.

- [ ] **Step 4: Record any final QA-only corrections**

If visual QA required small CSS/controller corrections, rerun the focused Node, pytest, Sphinx, and hook commands, then commit only those corrections with:

```bash
git add docs/_static
git commit -m "docs: finalize catalogue visual QA"
```
