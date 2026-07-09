from unittest.mock import patch

from uchi.tool_calling import ToolCall, default_registry


def test_web_search_not_registered_by_default():
    registry = default_registry()
    assert not registry.has("web_search")


def test_web_search_registered_when_enabled():
    registry = default_registry(enable_web_search=True)
    assert registry.has("web_search")


def test_web_search_tool_success():
    registry = default_registry(enable_web_search=True)
    with patch("uchi.web_search.perform_web_search", return_value="Paris is the capital of France."):
        entry = registry.dispatch(ToolCall(name="web_search", args={"query": "capital of France"}, raw="x"))
    assert entry.ok
    assert "Paris" in entry.result


def test_web_search_tool_empty_result_is_a_failure():
    """No results found must be a dispatch failure (not a silent empty
    success) so the loop guard/HitL machinery can see it and react."""
    registry = default_registry(enable_web_search=True)
    with patch("uchi.web_search.perform_web_search", return_value=""):
        entry = registry.dispatch(ToolCall(name="web_search", args={"query": "asdkjqwoieuqwoiue"}, raw="x"))
    assert not entry.ok
    assert "no web results" in entry.error
