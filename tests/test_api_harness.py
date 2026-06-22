import pytest
from fastapi.testclient import TestClient
from uchi.api import app

def test_api_stream():
    with TestClient(app) as client:
        response = client.post("/stream", json={"tokens": ["hello", "world"]})
        assert response.status_code == 200
        assert response.json() == {"status": "success", "processed": 2}

def test_api_query():
    with TestClient(app) as client:
        client.post("/stream", json={"tokens": ["hello", "world"]})
        response = client.post("/query", json={"tokens": ["hello"]})
        assert response.status_code == 200
        assert "answer" in response.json()

def test_api_predict():
    with TestClient(app) as client:
        client.post("/stream", json={"tokens": ["hello", "world"]})
        response = client.post("/predict", json={"context": ["hello"], "steps": 1})
        assert response.status_code == 200
        assert "prediction" in response.json()
