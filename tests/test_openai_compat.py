from fastapi.testclient import TestClient

from uchi import api_server


def test_openai_chat_completions_returns_openai_shape():
    api_server._router = api_server.Core()
    client = TestClient(api_server.app)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "uchi",
            "messages": [{"role": "user", "content": "What is the boiling point of water?"}],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "uchi"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(body["choices"][0]["message"]["content"], str)
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] > 0
    assert body["id"].startswith("chatcmpl-")


def test_openai_chat_completions_uses_last_user_message():
    api_server._router = api_server.Core()
    client = TestClient(api_server.app)

    api_server._router.swarm.answer = lambda q, callback=None: f"echo: {q}"
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": "be nice"},
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "some reply"},
                {"role": "user", "content": "second question"},
            ]
        },
    )
    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    assert "second question" in content
    assert "first question" not in content


def test_openai_chat_completions_requires_a_user_message():
    api_server._router = api_server.Core()
    client = TestClient(api_server.app)
    resp = client.post("/v1/chat/completions", json={"messages": [{"role": "system", "content": "hi"}]})
    assert resp.status_code == 400


def test_openai_chat_completions_503_before_startup():
    api_server._router = None
    client = TestClient(api_server.app)
    resp = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 503
