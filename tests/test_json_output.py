from lumina.models.base import extract_json_object


def test_last_json_object_is_used_after_thinking_trace() -> None:
    raw = '<think>Intermediate scratch object: {"candidate": "wrong"}</think>\n{"leading_label": "right"}'
    assert extract_json_object(raw) == {"leading_label": "right"}
