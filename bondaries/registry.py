"""Pinned tokenizer metadata and safe Hugging Face loading."""

import json
from importlib.resources import files
from pathlib import Path

SITE_ROLES = {"default", "dropdown"}


def registry():
    """Return independent records from the packaged tokenizer registry."""
    return json.loads(files("bondaries").joinpath("registry.json").read_text(encoding="utf-8"))


def site_models():
    """Return the tokenizers enabled for the browser."""
    return [model for model in registry() if model["role"] in SITE_ROLES]


def test_models():
    """Return site candidates and the dropped-character reference tokenizer."""
    return [model for model in registry() if model["role"] in SITE_ROLES | {"test-only"} or model.get("parity_exclusion")]


def get_model(identifier):
    """Resolve a registry identifier or repository name."""
    for model in registry():
        if identifier in (model["id"], model["repo"]):
            return model
    raise KeyError(f"Unknown tokenizer: {identifier}")


def tokenizer_source(identifier, revision=None):
    """Resolve registered models to their pinned revision; preserve local paths."""
    if Path(identifier).is_dir():
        return str(identifier), revision
    try:
        model = get_model(identifier)
    except KeyError:
        return str(identifier), revision
    return model["repo"], revision or model["sha"]


def load_tokenizer(identifier, revision=None, *, local_files_only=False):
    """Load a tokenizer without executing repository code.

    Registered identifiers and repository names use the recorded immutable revision
    unless explicitly overridden. Other Hugging Face repositories and local paths
    are accepted. Hugging Face's offline environment settings are respected.
    """
    from transformers import AutoTokenizer

    source, revision = tokenizer_source(identifier, revision)
    return AutoTokenizer.from_pretrained(
        source, revision=revision, trust_remote_code=False,
        local_files_only=local_files_only, use_fast=True,
    )


def snapshot_file(model, filename):
    """Find an already cached file at the exact registered revision, without downloads."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        model["repo"], filename, revision=model["sha"], local_files_only=True,
    ))


def write_manifest(path="docs/models.json"):
    """Write the browser manifest from the package's source of truth."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(site_models(), indent=2, ensure_ascii=False) + "\n")
    return target


def compare_aliases(first, second, cases):
    """Compare complete encodings, including special tokens, on all supplied cases."""
    left, right = load_tokenizer(first), load_tokenizer(second)
    count = 0
    for case in cases:
        text = case["prefix"] + case["smiles"]
        special = case["add_special_tokens"]
        a = left(text, add_special_tokens=special)["input_ids"]
        b = right(text, add_special_tokens=special)["input_ids"]
        if a != b:
            return {"matched": False, "cases": count, "first_mismatch": case,
                    "first_ids": a, "second_ids": b}
        count += 1
    return {"matched": True, "cases": count, "first_mismatch": None}


if __name__ == "__main__":
    write_manifest()
