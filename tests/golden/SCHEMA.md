# bondaries cross-language contract, version 1

Python is the reference. All character indices count Unicode code points, not UTF-16 units.
All spans are half-open `[start, end)`. Owners refer to indices in the full token list,
including special tokens. Missing ownership is `-1`.

Each tokenizer golden file is an object with `schema_version: 1`, `model` (registry
metadata), and `cases` (an array). Each case has `smiles`, `prefix`,
`add_special_tokens`, and `result`. The result contains:

- `ids`: integer array; `tokens`: raw token string array.
- `special_tokens_mask`: integer array (1 special, 0 content).
- `owners`: one integer per SMILES character; `text_owners`: one per prefix + SMILES character.
- `lexemes`: array of objects with `kind`, `text`, `start`, `end`, `atom_index`
  (integer for atoms, null otherwise). Atom lexemes additionally have `element`,
  `aromatic` (boolean), and `formal_charge` (integer). Non-atoms omit these fields.
- `atom_to_token`: integer array in input atom order. Use the first character's owner;
  use `-1` when that character is dropped. Split atoms remain explicitly flagged.
- `crimes`: object with arrays `split_atom`, `detached_ring_closure`,
  `context_merge`, and `dropped_character`, defined below.
- `lossless`: boolean; `alignment_ok`: boolean; `valid`: boolean;
  `error`: string or null. An invalid molecule or unsafe alignment is never colored.
- `token_count`: count of non-special tokens; `atoms_per_token` and `tokens_per_atom`:
  numbers, or null for a zero denominator.

Crimes are deterministic, in lexeme/character order (context merges in token order):

- `split_atom`: `{atom_index, start, end, owners}` for an atom whose characters
  have more than one distinct nonnegative owner; owners sorted ascending.
- `detached_ring_closure`: `{start, end, atom_index, atom_owner, owners}` for a ring
  lexeme with any nonnegative owner differing from the nearest preceding atom's
  first-character owner. Owners sorted ascending. No preceding atom means no event.
- `context_merge`: `{token_index}` for a token owning characters on both sides
  of the prefix/SMILES boundary.
- `dropped_character`: `{index, character}` for each SMILES character with owner -1.

Lexing follows the Schwaller-style SMILES expression, extended for `%nn`, `%(nnn)`,
and `*`. Kinds: `atom`, `bond`, `branch_open`, `branch_close`, `ring_closure`, `dot`.
Unrecognized syntax is an error. Reaction syntax is optional and must be explicit.
Atom order is validated with RDKit using `removeHs=False`, including atom count,
element, aromaticity and formal charge.

In the lossless case Python uses tokenizer offsets. JavaScript uses cumulative
prefix decoding with `skip_special_tokens: true, clean_up_tokenization_spaces: false`.
Decoded prefixes must be monotonic prefixes of the complete decode; the SMILES is
at its end. A token owns the newly decoded character interval. Both methods must
agree on all SMILES owners. Decoder-added leading whitespace may be handled only
when its position is unambiguous. Unsafe alignment disables atom colors.

In the lossy case both languages use the same deterministic greedy alignment over
prefix + SMILES: each content token's decoded piece matches at the earliest position
at or after the cursor. Skipped characters stay -1. Empty and unknown pieces own
nothing. Do not trust tokenizer offsets after dropped characters. ChemBERTa's
`BrC1=CC=CC=C1Cl` must drop `r` and `l` and retain the true positions of every `C`.

Batch collapse groups contain distinct input SMILES with identical content-token id
sequences. Identical repeated strings alone are not a collapse group.

Registry source: `bondaries/registry.json`; generated browser manifest: `docs/models.json`
(an array of model records). All model revisions are pinned. Each record includes
`id`, `label`, `repo`, `sha`, `license`, `group`, `role`, `mirrored`, and
`tokenizer_json_bytes`. Runtime URL construction uses the manifest metadata.

Shared JS interface: `lex(smiles)` returns lexemes; `analyze(smiles, tokenizer,
prefix = "", addSpecialTokens = true)` returns the result above synchronously;
`loadModels()` resolves the manifest array; `loadTokenizer(model)` resolves a tokenizer.
Python `analyze` returns a dataclass with `to_dict()` for the result shape above.
Any needed contract change is coordinated before changing producers or consumers.
