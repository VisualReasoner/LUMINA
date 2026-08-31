from __future__ import annotations

from pathlib import Path

from lumina.models.transformers_mllm import _resolve_snapshot


def test_huggingface_cache_directory_resolves_to_main_snapshot(tmp_path: Path) -> None:
    repository = tmp_path / "models--Qwen--Example"
    snapshot = repository / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (repository / "refs").mkdir()
    (repository / "refs" / "main").write_text("abc123\n", encoding="utf-8")
    assert _resolve_snapshot(str(repository)) == str(snapshot)
