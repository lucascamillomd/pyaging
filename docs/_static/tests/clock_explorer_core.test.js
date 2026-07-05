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
// Near full-row search: numeric/other fields beyond the old narrow list now match.
assert.deepStrictEqual(core.filterClocks(data, {}, "2020").map((c) => c.clock_name), ["a"]); // year
assert.deepStrictEqual(core.filterClocks(data, {}, "450k").map((c) => c.clock_name), ["b"]); // platform
assert.deepStrictEqual(core.filterClocks(data, {}, "cord blood").map((c) => c.clock_name), ["c"]); // notes phrase

assert.deepStrictEqual(core.sortClocks(data, "citations", "desc").map((c) => c.clock_name), ["a", "b", "c"]);
assert.deepStrictEqual(core.sortClocks(data, "clock_name", "asc").map((c) => c.clock_name), ["a", "b", "c"]);

assert.strictEqual(core.toCSV([data[0]], ["clock_name", "citations"]), "clock_name,citations\nb,10");

console.log("all clock_explorer_core tests passed");
