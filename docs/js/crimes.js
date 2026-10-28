/** Classify boundary events using code-point ownership and typed SMILES lexemes. */
export function classifyCrimes(smiles, lexemes, owners, textOwners = owners, prefixLength = 0) {
  const crimes = { split_atom: [], detached_ring_closure: [], context_merge: [], dropped_character: [] };
  let atom = null;
  const distinct = (start, end) => [...new Set(owners.slice(start, end).filter((owner) => owner >= 0))].sort((a, b) => a - b);
  for (const item of lexemes) {
    const tokenOwners = distinct(item.start, item.end);
    if (item.kind === "atom") {
      atom = item;
      if (tokenOwners.length > 1) crimes.split_atom.push({ atom_index: item.atom_index, start: item.start, end: item.end, owners: tokenOwners });
    } else if (item.kind === "ring_closure" && atom !== null && tokenOwners.some((owner) => owner !== owners[atom.start])) {
      crimes.detached_ring_closure.push({ start: item.start, end: item.end, atom_index: atom.atom_index, atom_owner: owners[atom.start], owners: tokenOwners });
    }
  }
  const prefixOwners = new Set(textOwners.slice(0, prefixLength).filter((owner) => owner >= 0));
  const merged = [...new Set(owners.filter((owner) => prefixOwners.has(owner)))].sort((a, b) => a - b);
  crimes.context_merge = merged.map((token_index) => ({ token_index }));
  Array.from(smiles).forEach((character, index) => {
    if (owners[index] === -1) crimes.dropped_character.push({ index, character });
  });
  return crimes;
}
