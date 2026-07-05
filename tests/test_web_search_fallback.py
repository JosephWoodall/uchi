from unittest.mock import patch

from uchi.generate_and_ground import GenerateAndGround


class _FakeIndex:
    """Minimal stand-in for SemanticIndex: starts with no evidence, then
    returns real evidence once build_from_corpus() has been called --
    exactly the shape the web-search fallback needs to exercise."""

    def __init__(self):
        self.learned = []
        self._has_evidence = False

    def build_from_corpus(self, text):
        self.learned.append(text)
        self._has_evidence = True

    def retrieve(self, query, k=10):
        if self._has_evidence:
            return [("Paris is the capital of France.", 0.95)]
        return []

    # Needed by _known_fraction()
    @property
    def w2i(self):
        return {w: 1 for w in "what is the capital of france".split()}


def test_web_search_fallback_disabled_by_default_still_abstains():
    pipeline = GenerateAndGround(index=_FakeIndex())
    with patch("uchi.web_search.perform_web_search") as mock_search:
        result = pipeline.answer("What is the capital of France?")
    mock_search.assert_not_called()
    assert "don't have grounded knowledge" in result


def test_web_search_fallback_enabled_learns_and_retries():
    pipeline = GenerateAndGround(index=_FakeIndex(), web_search_enabled=True)
    with patch(
        "uchi.web_search.perform_web_search",
        return_value="Paris is the capital of France.",
    ) as mock_search:
        result = pipeline.answer("What is the capital of France?")
    mock_search.assert_called_once()
    assert pipeline.index.learned == ["Paris is the capital of France."]
    assert "Paris" in result


def test_web_search_fallback_still_abstains_if_web_search_finds_nothing():
    pipeline = GenerateAndGround(index=_FakeIndex(), web_search_enabled=True)
    with patch("uchi.web_search.perform_web_search", return_value=""):
        result = pipeline.answer("What is the capital of France?")
    assert "don't have grounded knowledge" in result


def test_web_search_fallback_swallows_search_errors():
    pipeline = GenerateAndGround(index=_FakeIndex(), web_search_enabled=True)
    with patch("uchi.web_search.perform_web_search", side_effect=RuntimeError("network down")):
        result = pipeline.answer("What is the capital of France?")  # must not raise
    assert "don't have grounded knowledge" in result
