"""Public analysis and notebook drawing API."""

from dataclasses import asdict, dataclass
from html import escape

from rdkit import Chem, rdBase
from rdkit.Chem.Draw import rdMolDraw2D

from .align import align
from .crimes import detect_crimes
from .lexer import lex

_PALETTE = ("2a78d6", "eb6834", "1baf7a", "eda100", "e87ba4", "008300", "6250d6", "e34948")


@dataclass
class Analysis:
    """Token ownership and diagnostics for an exact SMILES input."""

    smiles: str
    prefix: str
    add_special_tokens: bool
    ids: list[int]
    tokens: list[str]
    special_tokens_mask: list[int]
    owners: list[int]
    text_owners: list[int]
    lexemes: list[dict]
    atom_to_token: list[int]
    crimes: dict
    lossless: bool
    alignment_ok: bool
    valid: bool
    error: str | None
    token_count: int
    atoms_per_token: float | None
    tokens_per_atom: float | None

    def to_dict(self) -> dict:
        """Return the shared JSON result shape, excluding input parameters."""
        result = asdict(self)
        for key in ("smiles", "prefix", "add_special_tokens"):
            result.pop(key)
        return result

    def _repr_svg_(self) -> str:
        """Render atom ownership when displayed in a notebook."""
        return str(draw(self))


class SVG(str):
    """An SVG string with rich notebook display support."""

    def _repr_svg_(self) -> str:
        return str(self)


def _molecule(smiles: str, sanitize: bool = True):
    parameters = Chem.SmilesParserParams()
    parameters.removeHs = False
    parameters.sanitize = sanitize
    parameters.parseName = False
    with rdBase.BlockLogs():
        return Chem.MolFromSmiles(smiles, parameters)


def analyze(smiles: str, tokenizer, prefix: str = "", add_special_tokens: bool = True) -> Analysis:
    """Inspect how a Hugging Face tokenizer splits an exact SMILES string.

    The tokenizer is supplied by the caller; no model code is downloaded or
    executed. Invalid molecules and unsafe alignments retain their token data
    but are never colored by :func:`draw`.
    """
    if not isinstance(smiles, str) or not isinstance(prefix, str):
        raise TypeError("smiles and prefix must be strings")
    data = align(prefix + smiles, tokenizer, add_special_tokens)
    owners = data["text_owners"][len(prefix):]
    error = data["error"]
    try:
        lexemes = lex(smiles)
    except ValueError as exc:
        lexemes = []
        error = str(exc)
    molecule = _molecule(smiles) if smiles and lexemes else None
    valid = molecule is not None
    if not valid:
        error = error or "Invalid SMILES."
    atoms = [item for item in lexemes if item["kind"] == "atom"]
    if valid:
        raw = _molecule(smiles, sanitize=False)
        expected = [(item["element"], item["aromatic"], item["formal_charge"]) for item in atoms]
        actual = [(atom.GetSymbol(), atom.GetIsAromatic(), atom.GetFormalCharge())
                  for atom in raw.GetAtoms()]
        if expected != actual:
            data["alignment_ok"] = False
            error = "Alignment failed: lexer and molecule atom order differ."
    if data["lossless"] and -1 in owners:
        data["alignment_ok"] = False
        error = "Alignment failed: tokenizer offsets do not cover SMILES."
    atom_to_token = [owners[item["start"]] for item in atoms]
    token_count = sum(not special for special in data["special_tokens_mask"])
    data["error"] = error
    return Analysis(smiles=smiles, prefix=prefix, add_special_tokens=add_special_tokens,
                    owners=owners, lexemes=lexemes, atom_to_token=atom_to_token,
                    crimes=detect_crimes(smiles, lexemes, owners, data["text_owners"], len(prefix)),
                    valid=valid, token_count=token_count,
                    atoms_per_token=len(atoms) / token_count if token_count else None,
                    tokens_per_atom=token_count / len(atoms) if atoms else None, **data)


def _rgb(color: str) -> tuple[float, float, float]:
    return tuple(int(color[start:start + 2], 16) / 255 for start in (0, 2, 4))


def draw(result: Analysis, width: int = 600, height: int = 350) -> SVG:
    """Return an RDKit SVG, coloring safe atom ownership by full token index.

    An atom owned by several tokens (a split atom) is striped in each of their colors.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Drawing dimensions must be positive.")
    if not result.valid:
        message = escape(result.error or "Invalid SMILES.")
        return SVG(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
                   f'<text x="16" y="32">{message}</text></svg>')
    molecule = _molecule(result.smiles)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    # Multi-color highlighting does not reliably honor useBWAtomPalette; blacken every element.
    drawer.drawOptions().setAtomPalette({z: (0.0, 0.0, 0.0) for z in range(-1, 119)})
    colors = {}
    if result.alignment_ok:
        # Color slot by position among real tokens, so a leading special token does not shift the scheme.
        slots, seen = [], 0
        for special in result.special_tokens_mask:
            slots.append(-1 if special else seen % len(_PALETTE))
            seen += not special
        atoms = [item for item in result.lexemes if item["kind"] == "atom"]
        for index, atom in enumerate(atoms):
            owners = list(dict.fromkeys(k for k in result.owners[atom["start"]:atom["end"]] if k >= 0))
            if owners:
                # RDKit lays the first color rightmost; reverse so stripes read left to right.
                colors[index] = [_rgb(_PALETTE[slots[owner]]) for owner in reversed(owners)]
    drawer.DrawMoleculeWithHighlights(molecule, "", colors, {}, {index: 0.34 for index in colors}, {})
    drawer.FinishDrawing()
    return SVG(drawer.GetDrawingText())
