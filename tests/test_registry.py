"""Pinned source metadata and complete alias-encoding contracts."""

import re

from bondaries.golden import golden_cases
from bondaries.registry import compare_aliases, get_model, registry, site_models, tokenizer_source


def test_registry_pins_and_site_scope():
    models = registry()
    assert len({model["id"] for model in models}) == len(models)
    assert all(re.fullmatch(r"[0-9a-f]{40}", model["sha"]) for model in models)
    assert all(model["tokenizer_json_bytes"] > 0 for model in models)
    assert "chemberta" not in {model["id"] for model in site_models()}
    assert sum(model["role"] == "default" for model in site_models()) == 5


def test_registered_repository_is_pinned_and_override_is_explicit(tmp_path):
    model = get_model("qwen2.5")
    assert tokenizer_source(model["repo"]) == (model["repo"], model["sha"])
    assert tokenizer_source("qwen2.5", "custom") == (model["repo"], "custom")
    assert tokenizer_source(str(tmp_path)) == (str(tmp_path), None)


def test_aliases_match_every_case_with_special_tokens():
    cases = list(golden_cases())
    assert any(case["add_special_tokens"] for case in cases)
    for first, second in [("qwen2.5", "qwen3"), ("olmo2", "olmo3")]:
        result = compare_aliases(first, second, cases)
        assert result == {"matched": True, "cases": len(cases), "first_mismatch": None}
