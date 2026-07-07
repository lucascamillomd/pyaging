/* Pure, framework-free logic for the Clock Explorer. Browser + Node. */
(function (root) {
  "use strict";

  var FACET_FIELDS = ["data_type", "species", "platform", "model_type", "unit", "tissue", "last_author", "journal", "predicts", "population", "approved_by_author"];
  var NUMERIC = { n_features: true, year: true, citations: true };
  // Search scans every metadata value on a clock except these keys (notebook is
  // an internal link path, not user-facing text worth matching on).
  var SEARCH_EXCLUDE = { notebook: true };

  function computeFacets(clocks) {
    var facets = {};
    FACET_FIELDS.forEach(function (f) {
      // Group case-insensitively so "Chronological age" and "chronological age"
      // collapse into one option; display the first-seen casing.
      var counts = {}, display = {};
      clocks.forEach(function (c) {
        var v = c[f];
        if (v === null || v === undefined || v === "") return;
        var key = String(v).toLowerCase();
        if (!Object.prototype.hasOwnProperty.call(counts, key)) { counts[key] = 0; display[key] = String(v); }
        counts[key] += 1;
      });
      facets[f] = Object.keys(counts)
        .sort(function (a, b) { return a.localeCompare(b); })
        .map(function (key) { return { value: display[key], count: counts[key] }; });
    });
    return facets;
  }

  function filterClocks(clocks, selected, search) {
    var q = (search || "").trim().toLowerCase();
    return clocks.filter(function (c) {
      for (var f in selected) {
        if (!selected.hasOwnProperty(f)) continue;
        var vals = selected[f] || [];
        if (vals.length) {
          var cv = c[f] == null ? "" : String(c[f]).toLowerCase();
          var hit = false;
          for (var vi = 0; vi < vals.length; vi++) {
            if (String(vals[vi]).toLowerCase() === cv) { hit = true; break; }
          }
          if (!hit) return false;
        }
      }
      if (q) {
        var hay = "";
        for (var k in c) {
          if (!c.hasOwnProperty(k) || SEARCH_EXCLUDE[k]) continue;
          var val = c[k];
          if (val === null || val === undefined) continue;
          hay += String(val).toLowerCase() + " ";
        }
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
