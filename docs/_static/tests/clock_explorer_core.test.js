const assert = require("node:assert");
const core = require("../clock_explorer_core.js");

const data = [
  { clock_name: "b", data_type: "methylation", species: "Homo sapiens", platform: ["Illumina 450K", "Illumina EPIC"], model_type: "Elastic net", unit: ["years"], predicts: ["chronological age", "mortality"], training_target: ["chronological age", "time-to-death"], tissue: ["blood", "plasma"], journal: "Aging Cell", population: "adult", approved_by_author: "By authors", citations: 10, year: 2018, n_features: 100, last_author: "X", notes: "blood clock" },
  { clock_name: "a", data_type: "rna", species: "Mus musculus", platform: ["RNA-seq"], model_type: "LASSO", unit: ["years"], predicts: ["chronological age"], training_target: ["chronological age"], tissue: ["liver"], journal: "Nature", population: "adult", approved_by_author: "Not yet", citations: 50, year: 2020, n_features: 5, last_author: "Y", notes: "liver" },
  { clock_name: "c", data_type: "methylation", species: "Homo sapiens", platform: ["Illumina EPIC"], model_type: "Elastic net", unit: ["weeks"], predicts: ["gestational age"], training_target: ["gestational age"], tissue: ["cord blood"], journal: "Aging Cell", population: "neonate", approved_by_author: "By authors", citations: 5, year: 2016, n_features: 200, last_author: "Z", notes: "cord blood" },
];

const facets = core.computeFacets(data);
// All categorical columns are faceted, including the newly-added ones.
assert.deepStrictEqual(core.FACET_FIELDS, ["data_type", "species", "platform", "model_type", "unit", "tissue", "last_author", "journal", "predicts", "training_target", "population", "approved_by_author"]);
assert.strictEqual(facets.data_type.length, 2);
assert.deepStrictEqual(facets.last_author.map((x) => x.value), ["X", "Y", "Z"]);
assert.strictEqual(facets.journal.find((x) => x.value === "Aging Cell").count, 2);
assert.strictEqual(facets.tissue.length, 4);
assert.strictEqual(facets.platform.find((x) => x.value === "Illumina EPIC").count, 2);
assert.strictEqual(facets.predicts.find((x) => x.value === "mortality").count, 1);
assert.strictEqual(facets.training_target.find((x) => x.value === "chronological age").count, 2);
// Numeric columns are NOT faceted.
assert.strictEqual(facets.citations, undefined);
assert.strictEqual(facets.year, undefined);

// Filtering: OR within a field, AND across fields, over any categorical column.
assert.deepStrictEqual(core.filterClocks(data, { last_author: ["X"] }, "").map((c) => c.clock_name), ["b"]);
assert.deepStrictEqual(core.filterClocks(data, { data_type: ["methylation"] }, "").map((c) => c.clock_name).sort(), ["b", "c"]);
assert.strictEqual(core.filterClocks(data, { data_type: ["methylation", "rna"] }, "").length, 3);
assert.deepStrictEqual(core.filterClocks(data, { tissue: ["blood"], platform: ["Illumina 450K"] }, "").map((c) => c.clock_name), ["b"]);
assert.deepStrictEqual(core.filterClocks(data, { data_type: ["methylation"], unit: ["weeks"] }, "").map((c) => c.clock_name), ["c"]);
assert.deepStrictEqual(core.filterClocks(data, { tissue: ["plasma", "liver"] }, "").map((c) => c.clock_name).sort(), ["a", "b"]);
assert.deepStrictEqual(core.filterClocks(data, { predicts: ["mortality"], training_target: ["time-to-death"] }, "").map((c) => c.clock_name), ["b"]);

// Search still scans all fields.
assert.deepStrictEqual(core.filterClocks(data, {}, "LIVER").map((c) => c.clock_name), ["a"]);
assert.deepStrictEqual(core.filterClocks(data, {}, "2020").map((c) => c.clock_name), ["a"]);
assert.deepStrictEqual(core.filterClocks(data, {}, "450k").map((c) => c.clock_name), ["b"]);
assert.deepStrictEqual(core.filterClocks(data, {}, "cord blood").map((c) => c.clock_name), ["c"]);
assert.deepStrictEqual(core.filterClocks(data, {}, "TIME-TO-DEATH").map((c) => c.clock_name), ["b"]);

assert.deepStrictEqual(core.sortClocks(data, "citations", "desc").map((c) => c.clock_name), ["a", "b", "c"]);
assert.deepStrictEqual(core.sortClocks(data, "clock_name", "asc").map((c) => c.clock_name), ["a", "b", "c"]);
assert.deepStrictEqual(core.sortClocks(data, "platform", "asc").map((c) => c.clock_name), ["b", "c", "a"]);
const defaultOrdered = core.defaultOrder([
  { clock_name: "Zulu", approved_by_author: "Not yet" },
  { clock_name: "beta", approved_by_author: "By authors" },
  { clock_name: "Alpha", approved_by_author: "By authors" },
  { clock_name: "aardvark", approved_by_author: "Not yet" },
]);
assert.deepStrictEqual(defaultOrdered.map((c) => c.clock_name), ["Alpha", "beta", "aardvark", "Zulu"]);
assert.strictEqual(core.toCSV([data[0]], ["clock_name", "citations"]), "clock_name,citations\nb,10");
assert.strictEqual(
  core.toCSV([{ tissue: ["blood", "serum, plasma"], notes: ['called "mixed"'] }], ["tissue", "notes"]),
  'tissue,notes\n"blood | serum, plasma","called ""mixed"""',
);
assert.strictEqual(
  core.toCSV([{ tissue: ["blood\rcells", "plasma"], notes: "line 1\r\nline 2" }], ["tissue", "notes"]),
  'tissue,notes\n"blood\rcells | plasma","line 1\r\nline 2"',
);

assert.deepStrictEqual(core.valuesOf(null), []);
assert.deepStrictEqual(core.valuesOf(undefined), []);
assert.deepStrictEqual(core.valuesOf(""), []);
assert.deepStrictEqual(core.valuesOf(["blood", "plasma"]), ["blood", "plasma"]);
assert.deepStrictEqual(core.valuesOf("blood"), ["blood"]);
assert.strictEqual(core.formatValue(null), "");
assert.strictEqual(core.formatValue(["blood", "plasma"]), "blood; plasma");
assert.strictEqual(core.formatValue(["blood", "plasma"], " | "), "blood | plasma");
assert.strictEqual(core.formatValue("blood"), "blood");

// Case-insensitive faceting: case variants collapse to one option (first-seen
// casing displayed), and a selected value matches every casing.
const caseData = [
  { clock_name: "p", predicts: ["chronological age", "Chronological Age", "mortality"] },
  { clock_name: "q", predicts: "Chronological age" },
  { clock_name: "r", predicts: "GESTATIONAL AGE" },
];
const cf = core.computeFacets(caseData);
assert.strictEqual(cf.predicts.length, 3);
assert.strictEqual(cf.predicts.find((x) => x.value.toLowerCase() === "chronological age").count, 2);
assert.strictEqual(core.filterClocks(caseData, { predicts: ["chronological age"] }, "").length, 2);
assert.deepStrictEqual(core.filterClocks(caseData, { predicts: ["gestational age"] }, "").map((c) => c.clock_name), ["r"]);
assert.deepStrictEqual(core.filterClocks(caseData, { predicts: ["MORTALITY", "gestational age"] }, "").map((c) => c.clock_name), ["p", "r"]);

// Prototype-named values and null-prototype/shadowed records remain ordinary
// searchable/filterable metadata rather than colliding with Object internals.
const nullProtoClock = Object.create(null);
nullProtoClock.clock_name = "prototype-safe";
nullProtoClock.predicts = ["__proto__", "constructor", "__PROTO__"];
nullProtoClock.notes = "null prototype record";
const shadowedClock = {
  clock_name: "shadowed",
  predicts: ["constructor"],
  notes: "own hasOwnProperty field",
  hasOwnProperty: "shadowed",
};
const prototypeData = [nullProtoClock, shadowedClock];
const prototypeFacets = core.computeFacets(prototypeData);
assert.strictEqual(prototypeFacets.predicts.find((x) => x.value.toLowerCase() === "__proto__").count, 1);
assert.strictEqual(prototypeFacets.predicts.find((x) => x.value.toLowerCase() === "constructor").count, 2);
const nullProtoSelected = Object.create(null);
nullProtoSelected.predicts = ["__PROTO__"];
assert.deepStrictEqual(core.filterClocks(prototypeData, nullProtoSelected, "").map((c) => c.clock_name), ["prototype-safe"]);
assert.deepStrictEqual(core.filterClocks(prototypeData, {}, "null prototype").map((c) => c.clock_name), ["prototype-safe"]);
assert.deepStrictEqual(core.filterClocks(prototypeData, {}, "own hasownproperty").map((c) => c.clock_name), ["shadowed"]);

console.log("all clock_explorer_core tests passed");
