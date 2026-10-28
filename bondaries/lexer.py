"""Lex SMILES without changing atom order or the original character spans."""

import re

_PATTERN = re.compile(r"\[[^\[\]]+\]|Br|Cl|[BCNOPSFIbcnops*]|%\([0-9]{1,5}\)|%[0-9]{2}|[0-9]|[-=#$:/\\~]|[().]")
_BRACKET_ATOM = re.compile(r"^\[(?:[0-9]+)?(\*|[A-Z][a-z]?|se|as|[bcnops])")
_CHARGE = re.compile(r"([+-])([0-9]+|\1*)")


def lex(smiles: str) -> list[dict]:
    """Return typed lexemes with Unicode character spans and input atom indices.

    This is a lexer, not a molecular validity check. Unsupported syntax raises
    ValueError; chemical validity is checked separately by :func:`analyze`.
    Aromaticity describes the input spelling, before aromaticity perception.
    """
    lexemes = []
    cursor = 0
    atom_index = 0
    while cursor < len(smiles):
        match = _PATTERN.match(smiles, cursor)
        if match is None:
            raise ValueError(f"Unrecognized SMILES syntax at character {cursor}.")
        token = match.group()
        item = {"kind": "atom", "text": token, "start": cursor,
                "end": match.end(), "atom_index": None}
        if token == "(":
            item["kind"] = "branch_open"
        elif token == ")":
            item["kind"] = "branch_close"
        elif token == ".":
            item["kind"] = "dot"
        elif token[0].isdigit() or token.startswith("%"):
            item["kind"] = "ring_closure"
        elif token in ("-", "=", "#", "$", ":", "/", "\\", "~"):
            item["kind"] = "bond"
        else:
            symbol = token
            charge = 0
            if token.startswith("["):
                atom_match = _BRACKET_ATOM.match(token)
                if atom_match is None:
                    raise ValueError(f"Unrecognized bracket atom at character {cursor}.")
                symbol = atom_match.group(1)
                charge_match = _CHARGE.search(token[atom_match.end():])
                if charge_match:
                    sign, amount = charge_match.groups()
                    charge = int(amount) if amount.isdigit() else len(amount) + 1
                    if sign == "-":
                        charge = -charge
            item.update(atom_index=atom_index, element=symbol.capitalize(),
                        aromatic=symbol.islower(), formal_charge=charge)
            atom_index += 1
        lexemes.append(item)
        cursor = match.end()
    return lexemes
