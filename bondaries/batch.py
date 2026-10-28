"""CSV analysis with explicit denominators and distinct-SMILES collapse groups."""

import csv
import json
from collections import defaultdict
from pathlib import Path

CRIME_NAMES = ("split_atom", "detached_ring_closure", "context_merge", "dropped_character")


def _fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def batch(csv_path, tokenizer, *, col="smiles", out, prefix="", add_special_tokens=True,
          tokenizer_name=None, revision=None):
    """Analyze one CSV column and write complete molecule, crime and collapse tables.

    Row counts include invalid SMILES, which are explicitly flagged. Character
    fractions use the sum of input SMILES lengths. Collapse groups require two
    or more distinct strings; repeated copies of one string alone never qualify.
    Only the requested column is retained, and terminal CSV newlines are removed.
    The returned summary is also saved as ``summary.json``. Output paths that
    refer to the input file are rejected before analysis or writes.
    """
    from .api import analyze

    source = Path(csv_path).resolve()
    target = Path(out).resolve()
    for filename in ("molecules.csv", "crimes.csv", "collapse_groups.csv", "summary.json"):
        output = target / filename
        if output.resolve() == source or (output.exists() and source.samefile(output)):
            raise ValueError(f"Output file would overwrite the input CSV: {output}")

    rows = []
    sequences = defaultdict(lambda: defaultdict(list))
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if col not in (reader.fieldnames or []):
            raise ValueError(f"CSV has no column named {col!r}")
        for index, row in enumerate(reader, start=1):
            if row.get(col) is None:
                raise ValueError(f"CSV row {index} has no value for {col!r}")
            smiles = row[col].rstrip("\r\n")
            result = analyze(smiles, tokenizer, prefix=prefix,
                             add_special_tokens=add_special_tokens)
            content_ids = [token_id for token_id, special in
                           zip(result.ids, result.special_tokens_mask) if not special]
            counts = {name: len(result.crimes[name]) for name in CRIME_NAMES}
            rows.append({"row": index, "smiles": smiles, "valid": result.valid,
                         "alignment_ok": result.alignment_ok, "error": result.error,
                         "token_count": result.token_count,
                         "character_count": len(smiles), "atom_count": len(result.atom_to_token),
                         "atoms_per_token": result.atoms_per_token,
                         "tokens_per_atom": result.tokens_per_atom,
                         "lossless": result.lossless, **counts,
                         "dropped_character_fraction": _fraction(
                             counts["dropped_character"], len(smiles)),
                         "content_ids": json.dumps(content_ids, separators=(",", ":"))})
            sequences[tuple(content_ids)][smiles].append(index)
    n = len(rows)
    characters = sum(row["character_count"] for row in rows)
    dropped = sum(row["dropped_character"] for row in rows)
    crime_rows = [{"crime": name, "n": n,
                   "molecules_with_crime": sum(row[name] > 0 for row in rows),
                   "event_count": sum(row[name] for row in rows),
                   "molecule_fraction": _fraction(sum(row[name] > 0 for row in rows), n)}
                  for name in CRIME_NAMES]
    collapse_rows = []
    group_count = 0
    for ids, strings in sequences.items():
        if len(strings) < 2:
            continue
        group_count += 1
        group_rows = sum(map(len, strings.values()))
        for smiles, indices in strings.items():
            collapse_rows.append({"group": group_count, "n": n,
                                  "group_rows": group_rows, "distinct_smiles": len(strings),
                                  "smiles": smiles, "occurrences": len(indices),
                                  "rows": json.dumps(indices),
                                  "content_ids": json.dumps(ids, separators=(",", ":"))})
    summary = {"n": n, "tokenizer": tokenizer_name, "revision": revision,
               "prefix": prefix, "add_special_tokens": add_special_tokens,
               "valid_molecules": sum(row["valid"] for row in rows),
               "invalid_molecules": sum(not row["valid"] for row in rows),
               "total_tokens": sum(row["token_count"] for row in rows),
               "mean_tokens_per_molecule": _fraction(sum(row["token_count"] for row in rows), n),
               "total_characters": characters, "dropped_characters": dropped,
               "dropped_character_fraction": _fraction(dropped, characters),
               "molecules_with_dropped_characters": sum(row["dropped_character"] > 0 for row in rows),
               "molecule_dropped_fraction": _fraction(
                   sum(row["dropped_character"] > 0 for row in rows), n),
               "collapse_group_count": group_count,
               "collapse_distinct_smiles": len(collapse_rows),
               "collapse_rows": sum(row["occurrences"] for row in collapse_rows),
               "crimes": crime_rows}
    target.mkdir(parents=True, exist_ok=True)
    molecule_fields = ["row", "n", "smiles", "valid", "alignment_ok", "error", "token_count",
                       "character_count", "atom_count", "atoms_per_token", "tokens_per_atom",
                       "lossless", *CRIME_NAMES, "dropped_character_fraction", "content_ids"]
    tables = [("molecules.csv", molecule_fields, [{**row, "n": n} for row in rows]),
              ("crimes.csv", ["crime", "n", "molecules_with_crime", "event_count",
                              "molecule_fraction"], crime_rows),
              ("collapse_groups.csv", ["group", "n", "group_rows", "distinct_smiles", "smiles",
                                       "occurrences", "rows", "content_ids"], collapse_rows)]
    for filename, fields, values in tables:
        with (target / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)
    (target / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary
