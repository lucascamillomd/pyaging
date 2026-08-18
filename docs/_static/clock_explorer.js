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
    training_target: "Training target", population: "Population", approved_by_author: "Verified",
  };
  var COLUMNS = [
    { key: "clock_name", label: "Clock", def: true, cls: "ce-col-clock" },
    { key: "citations", label: "Citations", def: true, num: true, cls: "ce-col-num" },
    { key: "downloads", label: "Downloads", def: true, num: true, cls: "ce-col-num" },
    { key: "approved_by_author", label: "Verified", def: true, cls: "ce-col-approval" },
    { key: "data_type", label: "Data type", def: true, cls: "ce-col-short" },
    { key: "species", label: "Species", def: true, cls: "ce-col-short" },
    { key: "year", label: "Year", def: true, num: true, cls: "ce-col-num" },
    { key: "n_features", label: "N features", def: false, num: true, cls: "ce-col-num" },
    { key: "unit", label: "Unit", def: false, cls: "ce-col-short" },
    { key: "model_type", label: "Model", def: false, cls: "ce-col-long" },
    { key: "platform", label: "Platform", def: false, cls: "ce-col-long" },
    { key: "predicts", label: "Predicts", def: true, cls: "ce-col-long" },
    { key: "tissue", label: "Tissue", def: true, cls: "ce-col-long" },
    { key: "population", label: "Population", def: false, cls: "ce-col-long" },
    { key: "last_author", label: "Last author", def: false, cls: "ce-col-long" },
    { key: "journal", label: "Journal", def: false, cls: "ce-col-long" },
  ];
  var DETAIL_FIELDS = [
    ["predicts", "Predicts"], ["training_target", "Training target"], ["unit", "Unit"], ["tissue", "Tissue"],
    ["platform", "Platform"], ["population", "Population"], ["model_type", "Model type"],
    ["n_features", "N features"], ["year", "Year"], ["citations", "Citations"], ["downloads", "Downloads"],
    ["last_author", "Last author"], ["journal", "Journal"], ["species", "Species"],
    ["data_type", "Data type"], ["approved_by_author", "Verified"],
  ];
  // Every sortable table column is offered in the quick-sort dropdown so a
  // column-header click always has a matching option and render() can re-sync
  // the dropdown instead of leaving it showing a stale key.
  var SORT_OPTIONS = [{ key: "default", label: "Verified first, then name" }].concat(
    COLUMNS.map(function (c) { return { key: c.key, label: c.label }; })
  );

  var state = {
    clocks: [], selected: {}, search: "", sortKey: "default", sortDir: "asc",
    cols: null, expanded: {},
  };
  var mount, body, sortSelEl, sortDirBtn;
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
    var filtered = core.filterClocks(state.clocks, state.selected, state.search);
    return state.sortKey === "default" ? core.defaultOrder(filtered) : core.sortClocks(filtered, state.sortKey, state.sortDir);
  }
  function fmt(v) {
    var formatted = core.formatValue(v);
    return formatted === "" ? "—" : formatted;
  }
  function columnClass(col) { return (col.num ? "ce-num " : "") + (col.cls || ""); }
  function approvalBadge(value) {
    var verified = String(value || "").toLowerCase() === "by authors";
    return el("span", "ce-approval " + (verified ? "is-approved" : "is-pending"), verified ? "By authors" : "Not yet");
  }

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
      // Build the (potentially long, free-text) checkbox list lazily on first
      // open, so mounting the page doesn't materialize thousands of hidden rows.
      var listBuilt = false;
      function buildList() {
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
      }
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
          if (!listBuilt) { buildList(); listBuilt = true; }
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

  // ---------- popular strip ----------
  // Top clocks by Hub downloads once counters accumulate; by paper citations
  // until then. Clicking a chip filters the table to that clock.
  function buildPopular() {
    var hasDownloads = state.clocks.some(function (c) { return (c.downloads || 0) > 0; });
    var metric = hasDownloads ? "downloads" : "citations";
    var top = state.clocks
      .filter(function (c) { return c[metric] != null; })
      .slice()
      .sort(function (a, b) { return (b[metric] || 0) - (a[metric] || 0); })
      .slice(0, 8);
    if (!top.length) return el("span");
    var wrap = el("div", "ce-popular");
    wrap.appendChild(el("span", "ce-popular-label", hasDownloads ? "Most downloaded" : "Most cited"));
    top.forEach(function (c, i) {
      var chip = el("button", "ce-popular-chip");
      chip.type = "button";
      chip.title = "Show " + c.clock_name + " in the table";
      chip.appendChild(el("span", "ce-popular-rank", String(i + 1)));
      chip.appendChild(el("span", "ce-popular-name", c.clock_name));
      chip.appendChild(el("span", "ce-popular-metric", core.formatValue(c[metric])));
      chip.addEventListener("click", function () {
        state.search = c.clock_name;
        state.expanded = {};
        state.expanded[c.clock_name] = true;
        buildAll();
      });
      wrap.appendChild(chip);
    });
    return wrap;
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
    sortDirBtn.setAttribute("aria-label", "Toggle sort direction");
    sortDirBtn.addEventListener("click", function () {
      state.sortDir = state.sortDir === "desc" ? "asc" : "desc";
      render();
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

    [sortSelEl, sortDirBtn, dl, reset, count].forEach(function (n) { bar.appendChild(n); });
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
      var th = el("th", columnClass(col), col.label);
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
        var td = el("td", columnClass(col));
        td.appendChild(col.key === "approved_by_author" ? approvalBadge(c[col.key]) : document.createTextNode(fmt(c[col.key])));
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

  // ---------- render ----------
  function render() {
    var rows = visible();
    var countEl = document.getElementById("ce-count");
    if (countEl) countEl.textContent = rows.length + " / " + state.clocks.length + " clocks";
    if (sortDirBtn) sortDirBtn.textContent = state.sortDir === "desc" ? "▼" : "▲";
    if (sortDirBtn) {
      sortDirBtn.disabled = state.sortKey === "default";
      sortDirBtn.title = state.sortKey === "default" ? "Default order is verified first, then name" : "Toggle sort direction";
      sortDirBtn.setAttribute("aria-label", sortDirBtn.title);
    }
    if (sortSelEl) {
      var hasKey = false;
      SORT_OPTIONS.forEach(function (o) { if (o.key === state.sortKey) hasKey = true; });
      if (hasKey) sortSelEl.value = state.sortKey;
    }
    body.innerHTML = "";
    body.appendChild(buildTable(rows));
  }

  function buildAll() {
    mount.innerHTML = "";
    // The Catalogue page uses the full content width once its sidebars are gone;
    // Task 3's CSS keys off this body class (scoped to where the app mounts).
    document.body.classList.add("ce-fullwidth");
    var layout = el("div", "ce-root");
    // Group the search/sort/view controls, the filter bar, and the active-filter
    // chips into one block above the bounded, self-scrolling table (.ce-scroll),
    // so they stay reachable while the 170+ rows scroll inside their box.
    var controls = el("div", "ce-controls");
    controls.appendChild(buildPopular());
    controls.appendChild(buildToolbar());
    controls.appendChild(buildFilterBar());
    controls.appendChild(buildActiveChips());
    layout.appendChild(controls);
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

  // Per-clock download counts come live from the Hub API: each clock has its
  // own model repo under the pyaging org, so its repo download counter IS the
  // per-clock metric. Fetched client-side, merged in when it arrives; the
  // catalogue works unchanged if the request fails (cells show an em dash).
  function fetchDownloads() {
    var counts = {};
    function page(url) {
      return fetch(url).then(function (r) {
        if (!r.ok) throw new Error("HF API " + r.status);
        var link = r.headers.get("Link");
        return r.json().then(function (models) {
          models.forEach(function (m) {
            var name = String(m.id || "").split("/")[1];
            var n = m.downloadsAllTime != null ? m.downloadsAllTime : m.downloads;
            if (name && n != null) counts[name] = n;
          });
          var next = link && /<([^>]+)>;\s*rel="next"/.exec(link);
          return next ? page(next[1]) : counts;
        });
      });
    }
    return page("https://huggingface.co/api/models?author=pyaging&expand[]=downloadsAllTime&limit=100");
  }

  function init() {
    mount = document.getElementById("clock-explorer");
    if (!mount || !core) return;
    document.addEventListener("click", closePopover);
    document.addEventListener("keydown", function (e) { if (e.key === "Escape") closePopover(); });
    fetch(staticBase() + "clocks.json")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        state.clocks = data;
        buildAll();
        fetchDownloads()
          .then(function (counts) {
            state.clocks.forEach(function (c) {
              var n = counts[String(c.clock_name || "").toLowerCase()];
              if (n != null) c.downloads = n;
            });
            buildAll();
          })
          .catch(function () { /* counts stay blank; the catalogue is fully usable without them */ });
      })
      .catch(function (e) {
        mount.appendChild(el("p", "ce-error", "Could not load clock data. See the table below or the GitHub repository. (" + e + ")"));
      });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
