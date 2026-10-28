"""Mirrored tokenizer equality, provenance and license preservation contracts."""

import json
from pathlib import Path

from bondaries.mirror import GEMMA_LICENSE_LINK
from bondaries.registry import get_model, registry, site_models, snapshot_file

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_matches_registry():
    assert json.loads((ROOT / "docs/models.json").read_text()) == site_models()


def test_mirrors_are_minified_json_equal_and_preserve_notices():
    for model in registry():
        if not model["mirrored"]:
            continue
        target = ROOT / "docs/tokenizers" / model["id"] / model["sha"]
        original = json.loads(snapshot_file(model, "tokenizer.json").read_text())
        data = (target / "tokenizer.json").read_text()
        assert (target / "tokenizer.json").stat().st_size == model["mirrored_tokenizer_json_bytes"]
        assert snapshot_file(model, "tokenizer.json").stat().st_size == model["tokenizer_json_bytes"]
        assert json.loads(data) == original
        assert data == json.dumps(original, separators=(",", ":"), ensure_ascii=False)
        assert (target / "tokenizer_config.json").read_bytes() == snapshot_file(
            model, "tokenizer_config.json").read_bytes()
        assert (target / "SOURCE.md").read_text() == (
            "Minified (JSON whitespace removed, content unchanged) from "
            f"https://huggingface.co/{model['repo']} at commit {model['sha']}.\n")
        if model["id"] == "gemma4":
            apache = snapshot_file(get_model("qwen3.5"), "LICENSE").read_text()
            assert (target / "LICENSE").read_text() == apache.rstrip() + "\n\n" + GEMMA_LICENSE_LINK + "\n"
        else:
            assert (target / "LICENSE").read_bytes() == snapshot_file(model, "LICENSE").read_bytes()
        if model["id"] == "gpt-oss":
            assert (target / "USAGE_POLICY").read_bytes() == snapshot_file(model, "USAGE_POLICY").read_bytes()
