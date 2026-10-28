# bondaries

Your tokenizer does not respect your bondaries.

See how LLM tokenizers split SMILES, atom by atom. Inspect token boundaries, character
ownership, colored molecular structures, and tokenization crimes in a browser or Python.

Planned site: [jenny-luo-0727.github.io/bondaries](https://jenny-luo-0727.github.io/bondaries/).
This source tree has not been published by this build.

## Python

```sh
pip install bondaries
```

Python 3.10 or newer is required. The package uses RDKit, Transformers and Tokenizers;
it ships metadata but no tokenizer assets. Supply any compatible Hugging Face tokenizer,
including one loaded from a local cache. Remote tokenizer code is never required by
bondaries itself.

```python
from transformers import AutoTokenizer
from bondaries import analyze, draw

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    revision="8faed761d45a263340a0528343f099c05c9a4323",
    trust_remote_code=False,
)
result = analyze("BrC1=CC=CC=C1Cl", tokenizer, prefix="", add_special_tokens=True)
print(result.tokens)
print(result.owners)
print(result.crimes)
svg = draw(result)
```

`analyze` returns a dataclass with IDs, raw tokens, special-token flags, lexemes,
character and atom ownership, counts, and diagnostics. `result.to_dict()` provides
serializable data. Both the result and the SVG support notebook display.
Character positions count Unicode code points. Special tokens stay in the full ID
list and appear as gray chips; token counts exclude them. Invalid SMILES and unsafe
alignments are reported and never receive atom colors.

Lossless encodings assign characters by cumulative decoding, in Python and in the
browser alike. Tokenizer offsets are not used: some tokenizers trim a token's leading
space from its span, or give the first character to a standalone word-boundary token.
Lossy encodings use deterministic left-to-right matching of decoded token pieces:
skipped characters have owner `-1`.

## CSV analysis

```sh
bondaries batch molecules.csv --col smiles \
  --tokenizer Qwen/Qwen2.5-1.5B \
  --revision 8faed761d45a263340a0528343f099c05c9a4323 \
  --out "$BONDARIES_OUT/qwen"
```

Set `BONDARIES_OUT` to an output directory outside this checkout. A local tokenizer
directory works with `--tokenizer`; use `--local-files-only` for cached files and
`--no-special-tokens` to disable special tokens. `--prefix 'SMILES: '` adds context.
Registered repository names default to the pinned commits listed below; `--revision`
explicitly overrides the default. Other repositories use the revision you provide.

The same operation is available as `bondaries.batch(csv_path, tokenizer, out=directory)`.
Only the requested CSV column is retained. Terminal newlines inside quoted SMILES are
removed. Outputs contain every input row, including invalid molecules:

- `molecules.csv`: SMILES, token and atom counts, crime counts, content IDs, validity,
  alignment status, and dropped-character fractions; each row states total `n`.
- `crimes.csv`: each crime's event count, affected-molecule count, total `n`, and rate.
- `collapse_groups.csv`: every distinct string in each shared content-ID sequence,
  occurrence counts and source row numbers. Repeated identical SMILES alone do not
  form a collapse group.
- `summary.json`: total `n`, total tokens and characters, mean tokens per molecule,
  character and molecule dropped fractions, collapse counts, and tokenizer provenance.

Rates include all input rows in their denominators. Character fractions divide by
all input SMILES characters; molecule fractions divide by `n`. Empty denominators
are `null`. Batch outputs are private analysis artifacts and should not be committed.

## Tokenization crimes

| Crime | Definition |
|---|---|
| `split_atom` | An atom lexeme has characters owned by more than one token. |
| `detached_ring_closure` | A ring-closure lexeme has an owner different from the nearest preceding atom's first-character owner. |
| `context_merge` | A token owns characters on both sides of the context-prefix boundary. |
| `dropped_character` | An input SMILES character has no owning token. |

Atoms per token and tokens per atom are informational. These diagnostics describe
string splitting; they do not rank the chemical capabilities of models.

## Browser tokenizers

The browser loads each pinned tokenizer on demand. Qwen3.5, Gemma 4 and gpt-oss are
mirrored under `docs/tokenizers`; the others load from Hugging Face. All tokenization
runs locally in the browser. Loading tokenizer files and libraries requires requests
to their hosts. Input stays local during analysis. Share links include the SMILES
and prefix in their query parameters; opening a shared link sends that query to
the site host. Do not share links containing private input.

| Tokenizer | Role | Pinned revision | License | Delivery |
|---|---|---|---|---|
| [Qwen2.5 / Qwen3](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/8faed761d45a263340a0528343f099c05c9a4323) | default | `8faed761d45a263340a0528343f099c05c9a4323` | apache-2.0 | Hugging Face |
| [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17) | default | `2fc06364715b967f1860aea9cf38778875588b17` | apache-2.0 | Mirrored (minified) |
| [Gemma 4](https://huggingface.co/google/gemma-4-E2B/tree/d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f) | default | `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f` | apache-2.0 | Mirrored (minified) |
| [OLMo 2 / 3](https://huggingface.co/allenai/OLMo-2-0425-1B/tree/a1847dff35000b4271fa70afc5db10fd29fedbdf) | default | `a1847dff35000b4271fa70afc5db10fd29fedbdf` | apache-2.0 | Hugging Face |
| [gpt-oss](https://huggingface.co/openai/gpt-oss-20b/tree/6cee5e81ee83917806bbde320786a8fb61efebee) | default | `6cee5e81ee83917806bbde320786a8fb61efebee` | apache-2.0 | Mirrored (minified) |
| [GPT-2](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e) | dropdown | `607a30d783dfa663caf39e06633721c8d4cfcd7e` | mit | Hugging Face |
| [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1/tree/56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad) | dropdown | `56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad` | mit | Hugging Face |
| [Phi-4](https://huggingface.co/microsoft/phi-4/tree/2db69c1c3e91a05d2c64a3185acfbaf36f744e25) | dropdown | `2db69c1c3e91a05d2c64a3185acfbaf36f744e25` | mit | Hugging Face |
| [Mistral-Small-3.1 (base)](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Base-2503/tree/ba6496e3dce1d0bdc93848804b1d4b9d5f3c57bc) | dropdown | `ba6496e3dce1d0bdc93848804b1d4b9d5f3c57bc` | apache-2.0 | Hugging Face |
| [GLM-4.5](https://huggingface.co/zai-org/GLM-4.5/tree/cbb2c7cfb52fa128a9660cb1a7a78e017899e115) | dropdown | `cbb2c7cfb52fa128a9660cb1a7a78e017899e115` | mit | Hugging Face |
| [SmolLM3](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/tree/a07cc9a04f16550a088caea529712d1d335b0ac1) | dropdown | `a07cc9a04f16550a088caea529712d1d335b0ac1` | apache-2.0 | Hugging Face |
| [Granite-4.0-micro](https://huggingface.co/ibm-granite/granite-4.0-micro/tree/56111ae135df9c53a78c99028e7bc24035a9e979) | dropdown | `56111ae135df9c53a78c99028e7bc24035a9e979` | apache-2.0 | Hugging Face |
| [Nemotron Nano 9B v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2/tree/6533e8de2c68e4536bf7c411d7a3ce5734111476) | dropdown | `6533e8de2c68e4536bf7c411d7a3ce5734111476` | nvidia-open-model-license | Hugging Face |
| [Nemotron 3.5 Lightning](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/tree/a9904d24bcc1d289a1950fa9d2b978c47cf903b9) | dropdown | `a9904d24bcc1d289a1950fa9d2b978c47cf903b9` | openmdw-1.1 | Hugging Face |
| [ChemFM-1B](https://huggingface.co/ChemFM/ChemFM-1B/tree/f99dc2e89726539bb9cf31b2e2b4360650bac6a8) | dropdown | `f99dc2e89726539bb9cf31b2e2b4360650bac6a8` | mit | Hugging Face |
| [Galactica-1.3B](https://huggingface.co/facebook/galactica-1.3b/tree/f711c69357d598defb703ddce93c5d7f7bc6e6da) | dropdown | `f711c69357d598defb703ddce93c5d7f7bc6e6da` | cc-by-nc-4.0 | Hugging Face |
| [MolT5-base](https://huggingface.co/laituan245/molt5-base/tree/7f2da6b30fd8ddc55b7867d53ce75e09bb85f284) | dropdown | `7f2da6b30fd8ddc55b7867d53ce75e09bb85f284` | apache-2.0 | Hugging Face |

Aliases are merged only after matching the complete curated encoding matrix, including
special tokens. Qwen2.5 / Qwen3 and OLMo 2 / 3 each matched 228/228 cases. Browser parity
is checked against Python on IDs, character owners and crime records. Mirrored files
are used directly by the parity gate.

## Not included

| Tokenizer | Reason |
|---|---|
| Llama | Gated; omitted from the browser selection. Locally available tokenizers can still be supplied to Python. |
| MoLFormer-XL | Its intended tokenizer requires repository code. |
| ether0 | The pinned tokenizer file raises the known-incorrect-regex warning in Transformers; the shipped file does not match the intended tokenizer. |
| ChemBERTa-77M-MTR | Test-only dropped-character reference. Its BPE tokenizer has zero merges and no unknown token; `BrC1=CC=CC=C1Cl` loses `r` and `l`, and reported offsets are misleading. Its license was not specified in the cached metadata. |
| Kimi-K2, ChemLLM, GT4SD T5, seyonec ChemBERTa-zinc, gpt2_zinc | No compatible `tokenizer.json` in the initial browser selection. |

## Development

```sh
export BONDARIES_LEXEMES_OUT="$BONDARIES_OUT/lexemes.jsonl"
python -m pytest -q
ruff check .
export BONDARIES_LEXER_JSONL="$BONDARIES_LEXEMES_OUT"
node tests/js/parity.test.mjs
python -m build --no-isolation
```

Tests include hand-confirmed SMILES and crime cases, alias and mirror contracts,
Python/JavaScript parity, and all 249,455 molecules in the pinned public ZINC250k
SMILES sample. Set `BONDARIES_CACHE` to a directory containing `zinc250k.csv` to use
an existing copy; otherwise the sample test retrieves the public file and verifies
its SHA-256. Tokenizer tests use the pinned Hugging Face snapshots; prepare them in
your cache before an offline run. Set `HF_HUB_OFFLINE=1` for offline operation.

Regenerate the browser manifest with `python -m bondaries.registry` and golden
references with `python -m bondaries.golden`. Golden generation retains excluded
parity candidates so their first mismatches remain reproducible;
`node tests/js/parity.test.mjs --all-candidates` includes them and is expected to
fail while a candidate differs. Mirrors can be regenerated with
`python -m bondaries.mirror --apache-license <Apache-2.0-text-file>`; all source
assets must already be cached at their pinned revisions. Serve `docs/` with any
static HTTP server. Browser smoke tests live in `tests/browser/smoke.py`.

## Licenses and attribution

bondaries is MIT licensed, © 2026 Jenny Luo. Tokenizer files retain their separate
upstream licenses; the browser table links each source at its pinned commit.
Qwen3.5 and gpt-oss mirrors include their upstream `LICENSE`; gpt-oss additionally
includes its verbatim `USAGE_POLICY`. Gemma 4 includes the Apache-2.0 license text and
[Google's Gemma 4 license link](https://ai.google.dev/gemma/docs/gemma_4_license).
Each mirror includes `SOURCE.md`, recording that JSON whitespace was removed without
changing content. No upstream NOTICE files were present for these three snapshots.
The Python wheel contains no tokenizer mirrors.

The interface uses RDKit.js 2026.9.1 (BSD-3-Clause), JSME 2024.4.29 (BSD-3-Clause),
and `@huggingface/tokenizers` 0.2.0 (Apache-2.0). Python analysis uses RDKit
(BSD-3-Clause), Transformers and Tokenizers (Apache-2.0). The lexer follows the
Schwaller-style SMILES token pattern with extended ring-closure and wildcard support.

Not affiliated with any model provider.
