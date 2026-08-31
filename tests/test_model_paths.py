from __future__ import annotations

from pathlib import Path

from lumina.models.transformers_mllm import _resolve_snapshot, _thinking_template_strategy


def test_huggingface_cache_directory_resolves_to_main_snapshot(tmp_path: Path) -> None:
    repository = tmp_path / "models--Qwen--Example"
    snapshot = repository / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (repository / "refs").mkdir()
    (repository / "refs" / "main").write_text("abc123\n", encoding="utf-8")
    assert _resolve_snapshot(str(repository)) == str(snapshot)


def test_thinking_toggle_is_forwarded_when_template_supports_it() -> None:
    assert _thinking_template_strategy("{% if enable_thinking %}", False) == "toggle"


def test_dedicated_thinking_template_uses_close_prefill() -> None:
    escaped = r"{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n{% endif %}"
    rendered = "<|im_start|>assistant\n<think>\n"
    assert _thinking_template_strategy(escaped, False) == "close_prefill"
    assert _thinking_template_strategy(rendered, False) == "close_prefill"


def test_dedicated_thinking_template_keeps_default_reasoning() -> None:
    template = "{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n{% endif %}"
    assert _thinking_template_strategy(template, None) == "default"
    assert _thinking_template_strategy(template, True) == "default"
