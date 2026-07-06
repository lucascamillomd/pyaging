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
