import pytest
from uchi.plugins.web import fetch_web_context
import requests

def mock_get(*args, **kwargs):
    class MockResponse:
        status_code = 200
        text = "<html><body><p>This is a simulated Wikipedia truth.</p></body></html>"
    return MockResponse()

def test_fetch_web_context(monkeypatch):
    monkeypatch.setattr(requests, "get", mock_get)
    result = fetch_web_context("mock_query")
    assert "This is a simulated Wikipedia truth." in result
