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
