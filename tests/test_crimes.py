"""Hand-confirmed token-boundary diagnostics."""

from bondaries.crimes import detect_crimes
from bondaries.lexer import lex


def test_split_atom_excludes_dropped_characters():
    result = detect_crimes("BrCl", lex("BrCl"), [0, 1, 2, -1], [0, 1, 2, -1], 0)
    assert result["split_atom"] == [{"atom_index": 0, "start": 0, "end": 2, "owners": [0, 1]}]
    assert result["dropped_character"] == [{"index": 3, "character": "l"}]


def test_ring_uses_nearest_preceding_atom_across_bond():
    smiles = "C=1CCCCC1"
    owners = [0, 1, 1, 2, 2, 2, 2, 3, 3]
    result = detect_crimes(smiles, lex(smiles), owners, owners, 0)
    assert result["detached_ring_closure"] == [
        {"start": 2, "end": 3, "atom_index": 0, "atom_owner": 0, "owners": [1]}]


def test_multichar_ring_and_context_merge():
    smiles = "C%10CC%10"
    owners = [2, 2, 3, 3, 4, 5, 5, 5, 5]
    result = detect_crimes(smiles, lex(smiles), owners, [0, 2] + owners, 2)
    assert result["context_merge"] == [{"token_index": 2}]
    assert result["detached_ring_closure"] == [
        {"start": 1, "end": 4, "atom_index": 0, "atom_owner": 2, "owners": [2, 3]}]


def test_detached_ring_after_branch_uses_lexical_rule():
    smiles = "C(C)1CC1"
    owners = list(range(len(smiles)))
    result = detect_crimes(smiles, lex(smiles), owners, owners, 0)
    assert result["detached_ring_closure"][0]["atom_index"] == 1


def test_empty_input_has_no_crimes():
    assert detect_crimes("", [], [], [], 0) == {
        "split_atom": [], "detached_ring_closure": [], "context_merge": [], "dropped_character": []}
