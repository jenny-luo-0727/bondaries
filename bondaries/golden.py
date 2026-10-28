"""Generate deterministic Python reference cases for browser parity checks."""

import json
from pathlib import Path

from .registry import load_tokenizer, test_models

SMILES = ['BrC1=CC=CC=C1Cl', 'c1ccc(Br)cc1', 'CC(=O)Oc1ccccc1C(=O)O', 'Cn1cnc2c1c(=O)n(C)c(=O)n2C', 'c1ccc2[nH]ccc2c1', 'C[C@@H](N)C(=O)O', 'N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O', '[13CH4]', 'C[N+](C)(C)C.[Cl-]', '[O-][N+](=O)c1ccccc1', '[H]C([H])([H])[H]', 'C%10CCCCCCCCC%10', 'C1CC%11CCC1CC%11', 'C=1CCCCC=1', 'c1cc[se]c1', '*c1ccccc1', 'CC(C)(C)OC(=O)N1CCC[C@H]1C(=O)O', 'F/C=C/F', 'F/C=C\\F', 'ClC(Cl)(Cl)Cl', 'OB(O)c1ccccc1', 'CC(C)Cc1ccc(cc1)[C@@H](C)C(O)=O', 'C1=CC=C2C(=C1)C=CC=C2', '[Na+].[Cl-]', 'CCO.CC(=O)O', 'C#N', '[2H]C([2H])([2H])O', 'c1ccc2c(c1)oc1ccccc12', 'C[Si](C)(C)C', 'P(=O)(O)(O)O', 'CS(=O)(=O)N', 'I', 'C1CC1', 'O=C1N(C)C(=O)N(C)c2ncn(C)c21', 'Clc1ccc(Br)c(I)c1F', 'C%(123)CCCCC%(123)', '[13C]', '[C@@H](F)(Cl)Br']
PREFIXES = ("", " ", "SMILES: ")


def golden_cases():
    """Yield the complete curated molecule, prefix and special-token matrix."""
    for smiles in SMILES:
        for prefix in PREFIXES:
            for special in (True, False):
                yield {"smiles": smiles, "prefix": prefix, "add_special_tokens": special}


def generate(directory="tests/golden"):
    """Write one versioned reference document per candidate tokenizer."""
    from .api import analyze

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    counts = {}
    for model in test_models():
        tokenizer = load_tokenizer(model["id"], local_files_only=True)
        cases = [{**case, "result": analyze(tokenizer=tokenizer, **case).to_dict()}
                 for case in golden_cases()]
        payload = {"schema_version": 1, "model": model, "cases": cases}
        (target / (model["id"] + ".json")).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        counts[model["id"]] = len(cases)
        print(f"{model['id']}: {len(cases)} cases", flush=True)
    return counts


if __name__ == "__main__":
    generate()
