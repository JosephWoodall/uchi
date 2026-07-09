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


def test_openai_chat_completions_uses_last_user_message_as_the_question():
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
    assert "Question: second question" in content


def test_openai_chat_completions_threads_prior_turns_as_context():
    """0.4.0 Item 17 fix: earlier turns must not be silently discarded --
    they were, before this fix (question = user_messages[-1].content and
    nothing else). Now they reach the pipeline as conversation context."""
    api_server._router = api_server.Core()
    client = TestClient(api_server.app)

    api_server._router.swarm.answer = lambda q, callback=None: f"echo: {q}"
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "some reply"},
                {"role": "user", "content": "second question"},
            ]
        },
    )
    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    assert "first question" in content
    assert "some reply" in content
    assert "second question" in content


def test_openai_chat_completions_does_not_pollute_shared_episodic_memory():
    """The REST server's _router is one global Core shared by every
    caller. Client-supplied history must be threaded through per-request
    without ever being written into (or read from) the shared instance's
    own episodic_memory -- otherwise one client's conversation would leak
    into another client's next request."""
    api_server._router = api_server.Core()
    client = TestClient(api_server.app)
    api_server._router.swarm.answer = lambda q, callback=None: f"echo: {q}"

    before = len(api_server._router.episodic_memory.history)
    client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "client A's secret question"},
                {"role": "assistant", "content": "client A's secret answer"},
                {"role": "user", "content": "client A's follow-up"},
            ]
        },
    )
    assert len(api_server._router.episodic_memory.history) == before

    resp2 = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "client B's unrelated question"}]},
    )
    content2 = resp2.json()["choices"][0]["message"]["content"]
    assert "client A" not in content2


def test_core_ask_conversation_context_overrides_episodic_memory():
    """Unit-level check of the underlying Core.ask() mechanism itself,
    independent of the REST layer."""
    from uchi.simple import Core
    core = Core()
    core.swarm.answer = lambda q, callback=None: q  # echo the augmented prompt back

    result = core.ask("real question", conversation_context="--- injected context ---")
    assert "injected context" in result
    assert "real question" in result
    # External context must not have been written into episodic_memory.
    assert len(core.episodic_memory.history) == 0

    # Without conversation_context, normal episodic-memory behavior is untouched.
    core.ask("a normal question")
    assert len(core.episodic_memory.history) == 1


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
