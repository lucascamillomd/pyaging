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
