"""Input-order and atom-metadata contracts for the SMILES lexer."""

import csv
import hashlib
import json
import os
import urllib.request
from pathlib import Path

import pytest
from rdkit import Chem

from bondaries.lexer import lex

SAMPLE_URL = (
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/"
    "master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
)
SAMPLE_SHA256 = "35e3f1a52b1badc0697e373d73a18ad773f415936ff992f4c6baa2e067b3e6ae"
EDGE_CASES = ["BrCCl", "c1cc[nH]c1", "N[C@@H](C)C(=O)O", "[13C]", "[O-]C=O",
              "[H]C([H])([H])[H]", "C%10CCCCC%10", "C%(123)CCCCC%(123)",
              "C=1CCCCC1", "CC.O", "c1cc[se]c1", "*C", "F/C=C\\F",
              "[NH4+]", "[Fe+2]", "[Fe++]", "[O-][N+](=O)O", "[2H]O[2H]"]


def assert_atom_parity(smiles):
    """Compare lexical atoms with RDKit before aromaticity perception."""
    parameters = Chem.SmilesParserParams()
    parameters.removeHs = False
    molecule = Chem.MolFromSmiles(smiles, parameters)
    assert molecule is not None, smiles
    parameters.sanitize = False
    raw = Chem.MolFromSmiles(smiles, parameters)
    items = lex(smiles)
    atoms = [item for item in items if item["kind"] == "atom"]
    assert len(atoms) == molecule.GetNumAtoms(), smiles
    assert [(item["element"], item["aromatic"], item["formal_charge"]) for item in atoms] == [
        (atom.GetSymbol(), atom.GetIsAromatic(), atom.GetFormalCharge()) for atom in raw.GetAtoms()
    ], smiles
    assert "".join(item["text"] for item in items) == smiles
    return items


@pytest.mark.parametrize("smiles", EDGE_CASES)
def test_input_atom_metadata(smiles):
    assert_atom_parity(smiles)


@pytest.mark.parametrize("smiles", ["C?C", "C C", "[C", "C>>O", "[foo]"])
def test_unrecognized_syntax_is_explicit(smiles):
    with pytest.raises(ValueError, match="Unrecognized"):
        lex(smiles)


def test_spans_and_ring_types():
    items = lex("BrC%(123)C=1.Cl")
    assert [(item["text"], item["start"], item["end"]) for item in items] == [
        ("Br", 0, 2), ("C", 2, 3), ("%(123)", 3, 9), ("C", 9, 10),
        ("=", 10, 11), ("1", 11, 12), (".", 12, 13), ("Cl", 13, 15)]
    assert items[2]["kind"] == "ring_closure"
    assert items[4]["kind"] == "bond"
    assert items[7]["atom_index"] == 3


def test_full_public_molecule_sample():
    """Verify every cached public molecule and optionally export JS parity input."""
    cache = Path(os.environ.get("BONDARIES_CACHE", Path.home() / ".cache" / "bondaries"))
    sample = cache / "zinc250k.csv"
    if not sample.exists():
        cache.mkdir(parents=True, exist_ok=True)
        temporary = cache / "zinc250k.csv.download"
        urllib.request.urlretrieve(SAMPLE_URL, temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != SAMPLE_SHA256:
            temporary.unlink()
            pytest.fail("Public molecule sample SHA256 mismatch")
        temporary.replace(sample)
    assert hashlib.sha256(sample.read_bytes()).hexdigest() == SAMPLE_SHA256
    export = os.environ.get("BONDARIES_LEXEMES_OUT")
    output = open(export, "w", encoding="utf-8") if export else None
    count = 0
    try:
        with sample.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                smiles = row["smiles"].strip()
                items = assert_atom_parity(smiles)
                if output:
                    output.write(json.dumps({"smiles": smiles, "lexemes": items},
                                            separators=(",", ":")) + "\n")
                count += 1
    finally:
        if output:
            output.close()
    assert count == 249455
    print(f"{count:,}/249,455 molecules matched RDKit")
