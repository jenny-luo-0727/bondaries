/** Lex SMILES without changing input atom order or character spans. */
const PATTERN = /\[[^\[\]]+\]|Br|Cl|[BCNOPSFIbcnops*]|%\([0-9]{1,5}\)|%[0-9]{2}|[0-9]|[-=#$:/\\~]|[().]/uy;
const BRACKET_ATOM = /^\[(?:[0-9]+)?(\*|[A-Z][a-z]?|se|as|[bcnops])/;
const CHARGE = /([+-])([0-9]+|\1*)/;

/** Return typed lexemes with Unicode code-point spans; invalid syntax throws. */
export function lex(smiles) {
  if (typeof smiles !== "string") throw new TypeError("smiles must be a string");
  const lexemes = [];
  let cursor = 0;
  let offset = 0;
  let atomIndex = 0;
  while (offset < smiles.length) {
    PATTERN.lastIndex = offset;
    const match = PATTERN.exec(smiles);
    if (!match) throw new Error(`Unrecognized SMILES syntax at character ${cursor}.`);
    const token = match[0];
    const end = cursor + Array.from(token).length;
    const item = { kind: "atom", text: token, start: cursor, end, atom_index: null };
    if (token === "(") item.kind = "branch_open";
    else if (token === ")") item.kind = "branch_close";
    else if (token === ".") item.kind = "dot";
    else if (/^[0-9%]/.test(token)) item.kind = "ring_closure";
    else if (["-", "=", "#", "$", ":", "/", "\\", "~"].includes(token)) item.kind = "bond";
    else {
      let symbol = token;
      let charge = 0;
      if (token.startsWith("[")) {
        const atomMatch = BRACKET_ATOM.exec(token);
        if (!atomMatch) throw new Error(`Unrecognized bracket atom at character ${cursor}.`);
        symbol = atomMatch[1];
        const chargeMatch = CHARGE.exec(token.slice(atomMatch[0].length));
        if (chargeMatch) {
          charge = /^[0-9]+$/.test(chargeMatch[2]) ? Number(chargeMatch[2]) : chargeMatch[2].length + 1;
          if (chargeMatch[1] === "-") charge = -charge;
        }
      }
      Object.assign(item, { atom_index: atomIndex++, element: symbol[0].toUpperCase() + symbol.slice(1).toLowerCase(), aromatic: /^[a-z]+$/.test(symbol), formal_charge: charge });
    }
    lexemes.push(item);
    offset = PATTERN.lastIndex;
    cursor = end;
  }
  return lexemes;
}
