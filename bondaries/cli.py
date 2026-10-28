"""Command-line entry points for SMILES tokenizer analysis."""

import argparse
import json


def main(argv=None):
    """Run a CSV analysis with a Hugging Face tokenizer or local tokenizer directory."""
    parser = argparse.ArgumentParser(prog="bondaries", description="Inspect how tokenizers split SMILES.")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("batch", help="Analyze every SMILES in a CSV column")
    command.add_argument("csv", help="Input CSV file")
    command.add_argument("--col", default="smiles", help="SMILES column (default: smiles)")
    command.add_argument("--tokenizer", required=True, help="Hugging Face ID or local directory")
    command.add_argument("--revision", help="Tokenizer commit or revision; registered IDs default to pinned commits")
    command.add_argument("--out", required=True, help="Output directory for tables and summary")
    command.add_argument("--prefix", default="", help="Text placed before every SMILES")
    command.add_argument("--no-special-tokens", action="store_true", help="Disable special tokens")
    command.add_argument("--local-files-only", action="store_true", help="Use cached files only")
    args = parser.parse_args(argv)
    from .batch import batch
    from .registry import load_tokenizer, tokenizer_source

    try:
        source, revision = tokenizer_source(args.tokenizer, args.revision)
        tokenizer = load_tokenizer(args.tokenizer, args.revision,
                                   local_files_only=args.local_files_only)
        summary = batch(args.csv, tokenizer, col=args.col, out=args.out,
                        prefix=args.prefix, add_special_tokens=not args.no_special_tokens,
                        tokenizer_name=source, revision=revision)
    except (OSError, ValueError) as error:
        parser.exit(2, f"bondaries: {error}\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0
