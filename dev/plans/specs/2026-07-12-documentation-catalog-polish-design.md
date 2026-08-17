# Documentation landing and catalogue polish — design spec

**Date:** 2026-07-12
**Status:** Approved in conversation
**Scope:** Sphinx documentation frontend only

## Goal

Remove the white square around the landing-page pyaging logo, make the Clock
Catalogue surface the author-approved clocks first, and reorganize the table so
compact information is visible before width-hungry descriptive fields. Include
small, directly related usability improvements discovered during the work.

## Landing-page logo

The existing PNG has an alpha channel, but its white corner pixels are opaque.
The landing page will therefore clip the image to a circle in CSS instead of
altering the shared logo asset. The landing logo receives equal width and height,
`border-radius: 50%`, and `object-fit: cover`. This makes the blue circular mark
blend cleanly into both light and dark documentation backgrounds without
changing navbar or sidebar rendering.

## Catalogue default ordering

The initial catalogue order will be:

1. Clocks whose `approved_by_author` value is `approved`.
2. All remaining clocks.

Clock names are sorted case-insensitively from A to Z inside both groups. The
catalogue controller will represent this as a dedicated default sort rather than
as a misleading single-column sort. Selecting a sort option or clicking a table
header replaces the default ordering with the user's requested sort.

The generated static JSON and CSV already use the same approved-first/name order;
the browser controller currently overrides it with citation count. The browser
logic and data-generation fallback will therefore agree after this change.

## Table information hierarchy

The table will expose all catalogue fields, ordered from compact identifying
information to long descriptive information:

1. Clock
2. Approval
3. Data type
4. Species
5. Year
6. Citations
7. N features
8. Unit
9. Model
10. Platform
11. Predicts
12. Tissue
13. Population
14. Last author
15. Journal

All columns are visible by default. Compact columns receive width/whitespace
hints so they consume only the space their content needs. Long text columns sit
at the right edge, where horizontal scrolling reveals progressive detail instead
of hiding compact fields behind empty space. The Clock column remains prominent
and readable.

Approval values will use compact status styling with text retained for clarity
and accessibility. Existing filtering, explicit sorting, table/card switching,
row expansion, and CSV download behavior remain intact.

## Related polish

The implementation may include only low-risk improvements adjacent to the
requested controls and table:

- clearer accessible names for icon-only sort controls;
- compact numeric alignment and column sizing;
- sensible overflow behavior for long table content;
- consistent approval presentation in table and detail/card contexts where the
  current component structure permits it without unrelated refactoring.

No navigation redesign, content rewrite, dependency addition, or changes under
`pyaging/**` are in scope.

## Files and responsibilities

- `docs/_static/custom.css`: circular landing-logo treatment.
- `docs/_static/clock_explorer_core.js`: approved-first/name default comparator.
- `docs/_static/clock_explorer.js`: default sort state, reordered/default-visible
  columns, approval rendering hooks, and accessible sort control labeling.
- `docs/_static/clock_explorer.css`: compact/long column sizing and approval
  status styling.
- `docs/_static/tests/clock_explorer_core.test.js`: approved-first alphabetical
  ordering and existing explicit-sort regression coverage.
- `docs/source/test_make_clock_data.py`: retain verification that generated data
  follows the same default order.

## Verification

- Run Node tests for catalogue filtering, explicit sorting, CSV output, and the
  new approved-first alphabetical default.
- Run the clock-data generator tests.
- Build the Sphinx HTML documentation and check for new warnings or errors.
- Inspect the generated landing and catalogue pages at desktop and narrow
  viewport widths, checking the circular logo, initial row order, column order,
  horizontal overflow, explicit sort interactions, and approval status display.

## Non-goals

- Editing the underlying logo PNG.
- Changing clock metadata, model behavior, notebooks, or package APIs.
- Removing existing catalogue fields or filter options.
- Redesigning documentation pages unrelated to the landing or catalogue.
