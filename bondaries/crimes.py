"""Deterministic token-boundary diagnostics over SMILES lexemes."""


def detect_crimes(smiles: str, lexemes: list[dict], owners: list[int],
                  text_owners: list[int], prefix_length: int) -> dict:
    """Find split atoms, detached rings, context merges, and dropped characters."""
    crimes = {"split_atom": [], "detached_ring_closure": [],
              "context_merge": [], "dropped_character": []}
    preceding_atom = None
    for item in lexemes:
        start, end = item["start"], item["end"]
        token_owners = sorted({owner for owner in owners[start:end] if owner >= 0})
        if item["kind"] == "atom":
            preceding_atom = item
            if len(token_owners) > 1:
                crimes["split_atom"].append({"atom_index": item["atom_index"],
                                             "start": start, "end": end,
                                             "owners": token_owners})
        elif item["kind"] == "ring_closure" and preceding_atom is not None:
            atom_owner = owners[preceding_atom["start"]]
            if any(owner != atom_owner for owner in token_owners):
                crimes["detached_ring_closure"].append({
                    "start": start, "end": end, "atom_index": preceding_atom["atom_index"],
                    "atom_owner": atom_owner, "owners": token_owners})
    before = set(text_owners[:prefix_length]) - {-1}
    after = set(text_owners[prefix_length:]) - {-1}
    crimes["context_merge"] = [{"token_index": index} for index in sorted(before & after)]
    crimes["dropped_character"] = [
        {"index": index, "character": character}
        for index, (character, owner) in enumerate(zip(smiles, owners)) if owner == -1
    ]
    return crimes
