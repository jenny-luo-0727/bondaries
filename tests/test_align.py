"""Ownership, notebook rendering, and lossy-tokenization contracts."""

import json
from pathlib import Path

import pytest

from bondaries import analyze, draw
from bondaries.align import greedy_owners


class CharacterTokenizer:
    """A small tokenizer fixture with visible offsets and optional BOS."""

    all_special_ids = [0]
    unk_token_id = 0

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False,
                 return_special_tokens_mask=True):
        ids = [ord(character) for character in text]
        offsets = [(index, index + 1) for index in range(len(text))]
        mask = [0] * len(text)
        if add_special_tokens:
            ids.insert(0, 0)
            offsets.insert(0, (0, 0))
            mask.insert(0, 1)
        result = {"input_ids": ids, "special_tokens_mask": mask}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def decode(self, ids, **kwargs):
        return "".join(chr(value) for value in ids if value)

    def convert_ids_to_tokens(self, ids):
        return [chr(value) if value else "<bos>" for value in ids]


class SlowTokenizer(CharacterTokenizer):
    def __call__(self, text, **kwargs):
        if kwargs.get("return_offsets_mapping"):
            raise NotImplementedError("Offsets unavailable")
        return super().__call__(text, **kwargs)


@pytest.mark.parametrize("tokenizer", [CharacterTokenizer(), SlowTokenizer()])
@pytest.mark.parametrize("prefix", ["", " ", "SMILES: ", "🧪 α  "])
@pytest.mark.parametrize("special", [False, True])
def test_lossless_ownership_with_and_without_offsets(tokenizer, prefix, special):
    result = analyze("BrCCl", tokenizer, prefix, special)
    assert result.valid and result.lossless and result.alignment_ok
    assert result.owners == list(range(len(prefix) + special, len(prefix) + special + 5))
    assert result.atom_to_token == [result.owners[index] for index in (0, 2, 3)]
    assert result.token_count == len(prefix) + 5
    assert "smiles" not in result.to_dict()


def test_greedy_skips_unknown_empty_and_unmatched_pieces():
    assert greedy_owners("BrCCCl", ["B", "", None, "x", "CC", "C"]) == [0, -1, 4, 4, 5, -1]


def test_decoder_added_leading_space_is_unambiguous():
    class LeadingSpace(SlowTokenizer):
        def decode(self, ids, **kwargs):
            return " " + super().decode(ids, **kwargs) if ids else ""
    result = analyze("CO", LeadingSpace())
    assert result.lossless and result.alignment_ok
    assert result.owners == [1, 2]


def test_unstable_prefix_decode_disables_colors():
    class Unstable(CharacterTokenizer):
        def decode(self, ids, **kwargs):
            return "?" if len(ids) == 1 else super().decode(ids, **kwargs)
    result = analyze("CO", Unstable())
    assert result.lossless and not result.alignment_ok
    assert "decoded prefixes" in result.error
    assert "#2A78D6" not in draw(result)


@pytest.mark.parametrize("smiles", ["C1CC", "C?O", "", "C C", "[CH5]"])
def test_invalid_smiles_never_draws_atoms(smiles):
    result = analyze(smiles, CharacterTokenizer())
    assert not result.valid
    assert "<svg" in draw(result)
    assert "atom-0" not in draw(result)


def test_svg_and_analysis_support_notebook_rendering():
    result = analyze("BrCCl", CharacterTokenizer())
    assert "atom-0" in draw(result)
    assert "<svg" in result._repr_svg_()
    assert "<svg" in draw(result)._repr_svg_()
    with pytest.raises(ValueError):
        draw(result, width=0)


def test_chemberta_true_character_positions():
    from transformers import AutoTokenizer

    records = json.loads((Path(__file__).parents[1] / "bondaries" / "registry.json").read_text())
    record = next(row for row in records if "ChemBERTa-77M-MTR" in row["repo"])
    tokenizer = AutoTokenizer.from_pretrained(record["repo"], revision=record["sha"],
                                              local_files_only=True, trust_remote_code=False)
    result = analyze("BrC1=CC=CC=C1Cl", tokenizer)
    assert not result.lossless and result.alignment_ok and result.valid
    assert result.crimes["dropped_character"] == [
        {"index": 1, "character": "r"}, {"index": 14, "character": "l"}]
    assert result.owners == [1, -1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -1]


def test_literal_special_token_owns_no_characters():
    class LiteralSpecial(CharacterTokenizer):
        all_special_ids = [0, ord("X")]

        def decode(self, ids, **kwargs):
            return super().decode([value for value in ids if value != ord("X")], **kwargs)

    result = analyze("CO", LiteralSpecial(), prefix="X", add_special_tokens=False)
    assert result.special_tokens_mask == [1, 0, 0]
    assert result.text_owners == [-1, 1, 2]
    assert result.owners == [1, 2]
    assert not result.lossless and result.alignment_ok
    assert result.token_count == 2


def test_trimmed_offsets_do_not_hide_a_context_merge():
    """A token encoding the prefix's trailing space plus SMILES is a context merge.

    Some tokenizers report that token's span without its leading space; ownership
    comes from decoding, so the trimmed offsets cannot hide the merge.
    """

    class SpaceMergingTokenizer(CharacterTokenizer):
        def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False,
                     return_special_tokens_mask=True):
            boundary = text.index(" B")
            pieces = list(text[:boundary]) + [" B"] + list(text[boundary + 2:])
            ids = [ord(piece) if len(piece) == 1 else 1 for piece in pieces]
            offsets, position = [], 0
            for piece in pieces:
                start = position + (1 if piece == " B" else 0)
                position += len(piece)
                offsets.append((start, position))
            result = {"input_ids": ids, "special_tokens_mask": [0] * len(ids)}
            if return_offsets_mapping:
                result["offset_mapping"] = offsets
            return result

        def decode(self, ids, **kwargs):
            return "".join(" B" if value == 1 else chr(value) for value in ids if value)

        def convert_ids_to_tokens(self, ids):
            return ["\u2581B" if value == 1 else chr(value) for value in ids]

    result = analyze("BrCCl", SpaceMergingTokenizer(), "SMILES: ", add_special_tokens=False)
    assert result.lossless and result.alignment_ok
    merged = len("SMILES:")
    assert result.text_owners[merged:merged + 2] == [merged, merged]
    assert result.crimes["context_merge"] == [{"token_index": merged}]
