import { lex } from "./lexer.js";
import { classifyCrimes } from "./crimes.js";

const DECODE_OPTIONS = { skip_special_tokens: true, clean_up_tokenization_spaces: false };
const characters = (text) => Array.from(text);
const decode = (tokenizer, ids) => ids.length ? tokenizer.decode(ids, DECODE_OPTIONS) : "";

/** Greedily match decoded pieces; skipped characters remain unowned. */
export function greedyOwners(text, pieces) {
  const points = characters(text);
  const owners = new Array(points.length).fill(-1);
  let cursor = 0;
  pieces.forEach((piece, index) => {
    if (!piece) return;
    const remaining = points.slice(cursor).join("");
    const offset = remaining.indexOf(piece);
    if (offset < 0) return;
    const start = cursor + characters(remaining.slice(0, offset)).length;
    const end = start + characters(piece).length;
    owners.fill(index, start, end);
    cursor = end;
  });
  return owners;
}

/** Encode and align content using cumulative decoding or deterministic matching. */
export function align(text, tokenizer, addSpecialTokens = true) {
  const encoded = tokenizer.encode(text, { add_special_tokens: addSpecialTokens });
  const ids = Array.from(encoded.ids);
  // Vocabulary strings, not encode()'s surface text: an unknown character must show as the
  // unknown token the model receives (e.g. T5 has no "\"), matching Python's convert_ids_to_tokens.
  const tokens = ids.map((id, index) => tokenizer.id_to_token?.(id) ?? encoded.tokens[index]);
  const specialIds = new Set(tokenizer.all_special_ids ?? []);
  for (const [id, token] of tokenizer.get_added_tokens_decoder?.() ?? []) {
    if (token.special) specialIds.add(id);
  }
  const specialStrings = new Set(tokenizer.special_tokens ?? []);
  const mask = ids.map((id, index) => Number(specialIds.has(id) || specialStrings.has(tokens[index])));
  const content = ids.filter((_, index) => !mask[index]);
  const decoded = decode(tokenizer, content);
  const shift = characters(decoded).length - characters(text).length;
  const leading = shift > 0 ? characters(decoded).slice(0, shift).join("") : "";
  const lossless = decoded === text || (shift > 0 && decoded.endsWith(text) && /^\s+$/u.test(leading));
  let alignmentOk = true;
  let error = null;
  let owners;
  if (lossless) {
    const decodedOwners = new Array(characters(decoded).length).fill(-1);
    const cumulative = [];
    let previous = "";
    ids.forEach((id, index) => {
      if (mask[index]) return;
      cumulative.push(id);
      const current = decode(tokenizer, cumulative);
      if (!decoded.startsWith(current) || !current.startsWith(previous)) alignmentOk = false;
      const start = characters(previous).length;
      const end = characters(current).length;
      if (end >= start) decodedOwners.fill(index, start, end);
      previous = current;
    });
    owners = decodedOwners.slice(Math.max(shift, 0));
    if (!alignmentOk) error = "Alignment failed: decoded prefixes are not stable.";
  } else {
    const unknown = tokenizer.model?.unk_token_id;
    const pieces = ids.map((id, index) => mask[index] || id === unknown ? null : decode(tokenizer, [id]));
    owners = greedyOwners(text, pieces);
  }
  return { ids, tokens, special_tokens_mask: mask, text_owners: owners, lossless, alignment_ok: alignmentOk, error };
}

/** Analyze an exact SMILES string; pass an initialized RDKit module for validity. */
export function analyze(smiles, tokenizer, prefix = "", addSpecialTokens = true, rdkit = globalThis.RDKit) {
  if (typeof smiles !== "string" || typeof prefix !== "string") throw new TypeError("smiles and prefix must be strings");
  const data = align(prefix + smiles, tokenizer, addSpecialTokens);
  const prefixLength = characters(prefix).length;
  const owners = data.text_owners.slice(prefixLength);
  let lexemes = [];
  let error = data.error;
  try {
    lexemes = lex(smiles);
  } catch (exception) {
    error = exception.message;
  }
  const atoms = lexemes.filter((item) => item.kind === "atom");
  let valid = false;
  if (smiles && lexemes.length && rdkit) {
    let molecule;
    try {
      molecule = rdkit.get_mol(smiles, JSON.stringify({ removeHs: false }));
      valid = Boolean(molecule?.is_valid());
      if (valid && JSON.parse(molecule.get_json()).molecules[0].atoms.length !== atoms.length) {
        data.alignment_ok = false;
        error = "Alignment failed: lexer and molecule atom order differ.";
      }
    } catch {
      valid = false;
    } finally {
      molecule?.delete();
    }
  }
  if (!valid) error ||= rdkit ? "Invalid SMILES." : "Molecule validation unavailable.";
  if (data.lossless && owners.includes(-1)) {
    data.alignment_ok = false;
    error = "Alignment failed: tokenizer offsets do not cover SMILES.";
  }
  const tokenCount = data.special_tokens_mask.filter((special) => !special).length;
  return { ...data, owners, lexemes, atom_to_token: atoms.map((atom) => owners[atom.start]),
    crimes: classifyCrimes(smiles, lexemes, owners, data.text_owners, prefixLength), valid, error,
    token_count: tokenCount, atoms_per_token: tokenCount ? atoms.length / tokenCount : null,
    tokens_per_atom: atoms.length ? tokenCount / atoms.length : null };
}
