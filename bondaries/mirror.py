"""Reproduce the browser's licensed tokenizer mirrors from pinned cached files."""

import json
import shutil
from pathlib import Path

from .registry import registry, snapshot_file

GEMMA_LICENSE_LINK = "https://ai.google.dev/gemma/docs/gemma_4_license"


def mirror_tokenizer(model, root="docs/tokenizers", *, apache_license=None):
    """Copy tokenizer assets and notices, changing only tokenizer JSON whitespace.

    Gemma's repository supplies no license file. Its Apache-2.0 license text must
    therefore be provided explicitly; the upstream license link is appended.
    """
    if not model["mirrored"]:
        raise ValueError("Tokenizer is not marked for mirroring")
    target = Path(root) / model["id"] / model["sha"]
    target.mkdir(parents=True, exist_ok=True)
    source = json.loads(snapshot_file(model, "tokenizer.json").read_text())
    packed = json.dumps(source, separators=(",", ":"), ensure_ascii=False)
    expected_bytes = model.get("mirrored_tokenizer_json_bytes")
    if expected_bytes is not None and len(packed.encode("utf-8")) != expected_bytes:
        raise ValueError("Minified tokenizer size differs from the pinned registry metadata")
    (target / "tokenizer.json").write_text(packed, encoding="utf-8")
    if json.loads((target / "tokenizer.json").read_text()) != source:
        raise ValueError("Minified tokenizer content changed")
    shutil.copyfile(snapshot_file(model, "tokenizer_config.json"), target / "tokenizer_config.json")
    if model["id"] == "gemma4":
        if apache_license is None:
            raise ValueError("Gemma mirror requires an Apache-2.0 license text")
        text = Path(apache_license).read_text()
        (target / "LICENSE").write_text(text.rstrip() + "\n\n" + GEMMA_LICENSE_LINK + "\n")
    else:
        shutil.copyfile(snapshot_file(model, "LICENSE"), target / "LICENSE")
    if model["id"] == "gpt-oss":
        shutil.copyfile(snapshot_file(model, "USAGE_POLICY"), target / "USAGE_POLICY")
    (target / "SOURCE.md").write_text(
        "Minified (JSON whitespace removed, content unchanged) from "
        f"https://huggingface.co/{model['repo']} at commit {model['sha']}.\n"
    )
    return target


def mirror_all(root="docs/tokenizers", *, apache_license=None):
    """Write every enabled mirror using only the local Hugging Face cache."""
    return [mirror_tokenizer(model, root, apache_license=apache_license)
            for model in registry() if model["mirrored"]]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apache-license", required=True)
    parser.add_argument("--out", default="docs/tokenizers")
    args = parser.parse_args()
    mirror_all(args.out, apache_license=args.apache_license)
