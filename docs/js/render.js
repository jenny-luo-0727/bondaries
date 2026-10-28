/** Render token ownership only after molecular and alignment checks succeed. */
const CRIME_LABELS = {
  split_atom: "Split atoms",
  detached_ring_closure: "Detached rings",
  context_merge: "Context merges",
  dropped_character: "Dropped characters",
};
// Atomic numbers -1 (default) through 118, all black, so labels stay legible over highlights.
const BLACK_ATOMS = Object.fromEntries(Array.from({ length: 120 }, (_, i) => [i - 1, [0, 0, 0]]));

export function element(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== "") node.textContent = text;
  return node;
}

function palette() {
  const css = getComputedStyle(document.documentElement);
  return Array.from({ length: 8 }, (_, k) => {
    const hex = css.getPropertyValue(`--token-${k}`).trim();
    return [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  });
}

// Color slot by position among real (non-special) tokens, so a leading <bos> does not shift
// the scheme: every tokenizer's first real token gets the first color.
function colorSlots(result) {
  let seen = 0;
  return result.special_tokens_mask.map((special) => (special ? -1 : seen++ % 8));
}

function linkHover(body, result) {
  const chips = [...body.querySelectorAll(".chip")];
  const atoms = result.lexemes.filter((item) => item.kind === "atom");
  const contributions = atoms.map((atom) => new Set(result.owners.slice(atom.start, atom.end).filter((k) => k >= 0)));
  const shapes = [...body.querySelectorAll(".molecule [class]")].map((node) => ({
    node,
    atoms: [...node.classList].filter((name) => /^atom-\d+$/.test(name)).map((name) => Number(name.slice(5))),
  })).filter((item) => item.atoms.length);
  const clear = () => {
    body.querySelector(".chips").classList.remove("has-active");
    chips.forEach((chip) => chip.classList.remove("is-active"));
    shapes.forEach(({ node }) => node.classList.remove("atom-active"));
  };
  const activate = (tokens, atomIndices) => {
    clear();
    if (!tokens.size) return;
    body.querySelector(".chips").classList.add("has-active");
    chips.forEach((chip) => chip.classList.toggle("is-active", tokens.has(Number(chip.dataset.tokenIndex))));
    shapes.forEach(({ node, atoms: indices }) => node.classList.toggle("atom-active", indices.some((i) => atomIndices.has(i))));
  };
  chips.forEach((chip) => {
    const show = () => {
      const index = Number(chip.dataset.tokenIndex);
      const linked = new Set(contributions.flatMap((tokens, i) => tokens.has(index) ? [i] : []));
      activate(new Set([index]), linked);
    };
    chip.addEventListener("mouseenter", show);
    chip.addEventListener("mouseleave", clear);
    chip.addEventListener("focus", show);
    chip.addEventListener("blur", clear);
  });
  shapes.forEach(({ node, atoms: indices }) => {
    node.dataset.atomIndices = indices.join(",");
    node.addEventListener("mouseenter", () => activate(new Set(indices.flatMap((i) => [...(contributions[i] || [])])), new Set(indices)));
    node.addEventListener("mouseleave", clear);
  });
}

/** Replace a panel's analysis body; metadata remains owned by the application. */
export function renderPanel(panel, result, smiles, rdkit, tokenizer = null) {
  const body = element("div", "panel-body");
  panel.querySelector(".panel-body")?.replaceWith(body);
  if (!body.isConnected) panel.append(body);
  panel.analysisResult = result;
  panel.dataset.smiles = smiles;
  const atomCount = result.lexemes.filter((item) => item.kind === "atom").length;
  let molecule = null;
  let message = result.error || "";
  let safe = Boolean(result.alignment_ok && result.valid);
  try {
    molecule = rdkit.get_mol(smiles, JSON.stringify({ removeHs: false }));
    if (!molecule || !molecule.is_valid() || !smiles) {
      safe = false;
      message = "Invalid SMILES. Check the molecule before coloring atoms.";
    } else {
      const actualCount = JSON.parse(molecule.get_json()).molecules[0].atoms.length;
      if (actualCount !== atomCount) {
        safe = false;
        message = `Atom-count mismatch (${atomCount} text atoms, ${actualCount} structure atoms). Atom colors are disabled.`;
      }
    }
  } catch {
    safe = false;
    message = "The molecule could not be parsed. Atom colors are disabled.";
  }
  if (!result.alignment_ok && !message) message = "Alignment failed. Atom colors are disabled.";
  panel.dataset.colored = String(safe);
  const chips = element("div", "chips");
  chips.setAttribute("aria-label", "Tokens; focus or hover to highlight atoms");
  const slots = colorSlots(result);
  result.ids.forEach((id, k) => {
    const special = result.special_tokens_mask[k] === 1;
    let piece = result.tokens[k];
    if (tokenizer && !special) {
      piece = tokenizer.decode([id], { skip_special_tokens: false, clean_up_tokenization_spaces: false }) || piece;
    }
    const chip = element("button", `chip${special || !safe ? " special" : ""}`, piece.replaceAll(" ", "␠").replaceAll("\n", "↵").replaceAll("\t", "⇥"));
    chip.type = "button";
    chip.dataset.tokenIndex = String(k);
    if (!special) {
      chip.dataset.slot = String(slots[k]);
      chip.style.setProperty("--token-color", `var(--token-${slots[k]})`);
    }
    chip.title = `Token ${k} · id ${id} · raw ${result.tokens[k]}${special ? " · special token" : ""}`;
    chip.setAttribute("aria-label", chip.title);
    chips.append(chip);
  });
  body.append(chips);
  if (!safe) {
    const banner = element("p", "banner alignment-banner", message || "Alignment failed. Atom colors are disabled.");
    banner.setAttribute("role", "status");
    body.append(banner);
  }
  if (molecule) {
    try {
      if (molecule.is_valid()) {
        const drawing = element("div", "molecule");
        drawing.setAttribute("aria-label", safe ? "Molecule colored by token ownership" : "Molecule without token colors");
        const colors = palette();
        // Every token owning part of an atom, in character order: a split atom is striped in all of them.
        const atomOwners = result.lexemes.filter((item) => item.kind === "atom")
          .map((atom) => [...new Set(result.owners.slice(atom.start, atom.end).filter((k) => k >= 0))]);
        const ownedAtoms = safe ? atomOwners.flatMap((owners, i) => owners.length ? [i] : []) : [];
        drawing.innerHTML = molecule.get_svg_with_highlights(JSON.stringify({
          width: 560, height: 300, atoms: ownedAtoms, bonds: [],
          // RDKit lays the first color rightmost; reverse so stripes read left to right like the chips.
          highlightAtomMultipleColors: Object.fromEntries(ownedAtoms.map((i) => [i, atomOwners[i].map((k) => colors[slots[k]]).reverse()])),
          highlightRadii: Object.fromEntries(ownedAtoms.map((i) => [i, 0.34])),
          // Multi-color highlighting ignores useBWAtomPalette, so blacken every element explicitly.
          useBWAtomPalette: true, atomColourPalette: BLACK_ATOMS, clearBackground: false, baseFontSize: 0.75,
        }));
        body.append(drawing);
      }
    } finally {
      molecule.delete();
    }
  }
  const metrics = element("p", "metrics");
  const count = element("strong", "", String(result.token_count));
  metrics.append(count, ` content tokens · ${atomCount} atoms`);
  if (result.tokens_per_atom !== null) metrics.append(` · ${result.tokens_per_atom.toFixed(2)} tokens/atom`);
  if (result.atoms_per_token !== null) metrics.append(` · ${result.atoms_per_token.toFixed(2)} atoms/token`);
  body.append(metrics);
  const crimes = element("div", "crimes");
  for (const [key, label] of Object.entries(CRIME_LABELS)) {
    const count = result.crimes[key].length;
    const badge = element("span", `crime${count ? " has-crime" : ""}`, label);
    badge.dataset.crime = key;
    badge.dataset.count = String(count);
    badge.append(element("strong", "", String(count)));
    crimes.append(badge);
  }
  body.append(crimes);
  if (result.crimes.dropped_character.length) {
    const dropped = element("p", "dropped");
    dropped.append("Dropped: ");
    const positions = new Set(result.crimes.dropped_character.map((item) => item.index));
    [...smiles].forEach((character, i) => {
      const node = element(positions.has(i) ? "mark" : "span", "", character);
      if (positions.has(i)) node.title = `Dropped character at index ${i}`;
      dropped.append(node);
    });
    body.append(dropped);
  }
  if (safe) linkHover(body, result);
  panel.dataset.state = "ready";
}
