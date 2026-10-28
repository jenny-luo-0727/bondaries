/** Offline JavaScript/Python parity gate for pinned tokenizers and the full lexer sample. */
import assert from "node:assert/strict";
import { createReadStream, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createInterface } from "node:readline";
import { isDeepStrictEqual } from "node:util";
import { Tokenizer } from "@huggingface/tokenizers";
import initRDKitModule from "@rdkit/rdkit";
import { align, analyze, greedyOwners } from "../../docs/js/align.js";
import { lex } from "../../docs/js/lexer.js";
import { classifyCrimes } from "../../docs/js/crimes.js";

const root = fileURLToPath(new URL("../../", import.meta.url));
const readJSON = (filename) => JSON.parse(readFileSync(filename, "utf8"));
const manifest = readJSON(path.join(root, "docs/models.json"));
const siteIds = new Set(manifest.map((model) => model.id));
const rdkit = await initRDKitModule();
const args = new Set(process.argv.slice(2));
const allCandidates = args.has("--all-candidates");
const goldenDirectory = path.join(root, "tests/golden");
const files = readdirSync(goldenDirectory).filter((filename) => filename.endsWith(".json")).sort();
const hub = process.env.HF_HUB_CACHE || path.join(process.env.HF_HOME || path.join(homedir(), ".cache/huggingface"), "hub");
const report = { models: [], lexer: null };
let failed = false;

assert.deepEqual(greedyOwners("BrCCl", ["B", "C", "C"]), [0, -1, 1, 2, -1]);
assert.deepEqual(greedyOwners("🧪CC", ["🧪", "CC"]), [0, 1, 1]);
const unstableTokenizer = {
  encode: () => ({ ids: [1, 2, 3], tokens: ["byte-a", "byte-b", "C"] }),
  decode: (ids) => ["", "�", "🧪", "🧪C"][ids.length],
};
assert.equal(align("🧪C", unstableTokenizer).alignment_ok, false);
assert.equal(analyze("C", unstableTokenizer, "🧪", true, rdkit).alignment_ok, false);
assert.equal(analyze("C", unstableTokenizer, "🧪", true, null).valid, false);
assert.deepEqual(classifyCrimes("BrC1", lex("BrC1"), [0, 1, 2, 3]), {
  split_atom: [{ atom_index: 0, start: 0, end: 2, owners: [0, 1] }],
  detached_ring_closure: [{ start: 3, end: 4, atom_index: 1, atom_owner: 2, owners: [3] }],
  context_merge: [], dropped_character: [],
});

console.log("| Tokenizer | IDs | Owners | Crimes | Other fields | First mismatch |");
console.log("|---|---:|---:|---:|---:|---|");
for (const filename of files) {
  const golden = readJSON(path.join(goldenDirectory, filename));
  const model = golden.model;
  if (!siteIds.has(model.id) && model.role !== "test-only" && !model.id.includes("chemberta") && !allCandidates) continue;
  assert.equal(golden.schema_version, 1);
  const base = model.mirrored
    ? path.join(root, "docs/tokenizers", model.id, model.sha)
    : path.join(hub, `models--${model.repo.replaceAll("/", "--")}`, "snapshots", model.sha);
  const row = { id: model.id, cases: golden.cases.length, ids: 0, owners: 0, crimes: 0, other: 0, text_owners: 0, first_text_mismatch: null, first_mismatch: null };
  try {
    const tokenizer = new Tokenizer(readJSON(path.join(base, "tokenizer.json")), readJSON(path.join(base, "tokenizer_config.json")));
    for (const entry of golden.cases) {
      const actual = analyze(entry.smiles, tokenizer, entry.prefix, entry.add_special_tokens, rdkit);
      const expected = entry.result;
      if (isDeepStrictEqual(actual.text_owners, expected.text_owners)) row.text_owners++;
      else row.first_text_mismatch ||= { smiles: entry.smiles, prefix: entry.prefix,
        add_special_tokens: entry.add_special_tokens, actual: actual.text_owners, expected: expected.text_owners };
      const checks = { ids: ["ids"], owners: ["owners"], crimes: ["crimes"],
        other: ["tokens", "special_tokens_mask", "lexemes", "atom_to_token", "lossless", "alignment_ok", "valid", "error", "token_count", "atoms_per_token", "tokens_per_atom"] };
      for (const [category, fields] of Object.entries(checks)) {
        const mismatch = fields.find((field) => !isDeepStrictEqual(actual[field], expected[field]));
        if (!mismatch) row[category]++;
        else if (!row.first_mismatch) row.first_mismatch = {
          smiles: entry.smiles, prefix: entry.prefix, add_special_tokens: entry.add_special_tokens,
          field: mismatch, actual: actual[mismatch], expected: expected[mismatch],
        };
      }
      if (model.id.includes("chemberta") && entry.smiles === "BrC1=CC=CC=C1Cl" && entry.prefix === "") {
        assert.deepEqual(actual.crimes.dropped_character, [{ index: 1, character: "r" }, { index: 14, character: "l" }]);
      }
    }
  } catch (error) {
    row.first_mismatch ||= { error: error.message };
  }
  if (row.first_mismatch) failed = true;
  report.models.push(row);
  const mismatch = row.first_mismatch ? JSON.stringify(row.first_mismatch) : "none";
  console.log(`| ${model.id} | ${row.ids}/${row.cases} | ${row.owners}/${row.cases} | ${row.crimes}/${row.cases} | ${row.other}/${row.cases} | ${mismatch} |`);
}
for (const id of siteIds) assert.ok(report.models.some((model) => model.id === id), `Missing golden cases for ${id}`);
assert.ok(report.models.some((model) => model.id.includes("chemberta")), "Missing ChemBERTa lossy cases");

if (!args.has("--tokenizers-only")) {
  const filename = process.env.BONDARIES_LEXER_JSONL;
  assert.ok(filename, "Set BONDARIES_LEXER_JSONL to the lexer export written by pytest (BONDARIES_LEXEMES_OUT), or pass --tokenizers-only");
  const lines = createInterface({ input: createReadStream(filename), crlfDelay: Infinity });
  let count = 0;
  for await (const line of lines) {
    if (!line.trim()) continue;
    const entry = JSON.parse(line);
    assert.deepEqual(lex(entry.smiles), entry.lexemes, `Lexer mismatch at molecule ${count + 1}: ${entry.smiles}`);
    count++;
  }
  assert.equal(count, 249455, "The full pinned sample must be checked");
  report.lexer = { matched: count, total: 249455 };
  console.log(`JS lexer: ${count}/249455 molecules matched Python.`);
}
if (process.env.BONDARIES_PARITY_REPORT) writeFileSync(process.env.BONDARIES_PARITY_REPORT, JSON.stringify(report, null, 2) + "\n");
if (failed) process.exitCode = 1;
