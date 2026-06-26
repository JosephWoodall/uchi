import pytest
from unittest.mock import patch

# Patch _bootstrap_knowledge before api_server imports OmniRouter at module level.
# The bootstrap runs the full stdlib walk which is too slow for unit tests.
with patch("uchi.omni_router.OmniRouter._bootstrap_knowledge"):
    from uchi.api_server import app

from fastapi.testclient import TestClient


def test_metrics_returns_online():
    with TestClient(app) as client:
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert "memory_records" in data


def test_chat_returns_reply():
    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "hello"})
        assert response.status_code == 200
        data = response.json()
        assert "reply" in data
        assert isinstance(data["reply"], str)
        assert len(data["reply"]) > 0


def test_chat_empty_message_returns_400():
    with TestClient(app) as client:
        response = client.post("/chat", json={"message": ""})
        assert response.status_code == 400
