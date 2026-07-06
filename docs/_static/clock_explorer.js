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
  function boxKey(f, value) { return f + " " + value; }
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
        if (!isOpen) {
          pop.style.display = "block";
          openPopover = pop;
          // Reset the in-dropdown search: clear the box AND re-show every row
          // (setting value alone does not fire the input handler).
          search.value = "";
          for (var i = 0; i < list.childNodes.length; i++) list.childNodes[i].style.display = "";
          // Keep the popover inside the viewport on the full-width page: if it
          // would overflow the right edge, anchor it to the button's right side.
          pop.style.left = "0"; pop.style.right = "auto";
          if (pop.getBoundingClientRect().right > document.documentElement.clientWidth - 4) {
            pop.style.left = "auto"; pop.style.right = "0";
          }
          search.focus();
        }
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
