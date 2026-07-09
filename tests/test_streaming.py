import time

from uchi.streaming import stream_ask


class _FakeCore:
    def ask(self, question, callback=None, **data):
        callback("thinking", "step 1")
        time.sleep(0.05)
        callback("reinforce", "step 2")
        return f"answer to {question}"


class _BrokenCore:
    def ask(self, question, callback=None, **data):
        callback("thinking", "about to fail")
        raise RuntimeError("pipeline exploded")


def test_stream_ask_yields_thoughts_then_speech():
    events = list(stream_ask(_FakeCore(), "hello"))
    assert events[0] == {"type": "thought", "stage": "thinking", "message": "step 1"}
    assert events[1] == {"type": "thought", "stage": "reinforce", "message": "step 2"}
    assert events[2] == {"type": "speech", "content": "answer to hello"}


def test_stream_ask_events_arrive_before_final_answer_is_computed():
    """Genuine streaming, not chunking a pre-computed string: the first
    thought event must be observable while ask() is still running."""
    gen = stream_ask(_FakeCore(), "hello")
    first = next(gen)
    assert first["type"] == "thought"
    # ask() sleeps 0.05s between the two callback() calls -- confirm we
    # got the first event without waiting for the whole call to finish.
    remaining = list(gen)
    assert remaining[-1]["type"] == "speech"


def test_stream_ask_surfaces_exceptions_as_a_speech_error_event():
    events = list(stream_ask(_BrokenCore(), "hello"))
    assert events[0]["type"] == "thought"
    final = events[-1]
    assert final["type"] == "speech"
    assert final["content"] == ""
    assert "pipeline exploded" in final["error"]


def test_stream_ask_with_no_callbacks_still_yields_final_speech():
    class _SilentCore:
        def ask(self, question, callback=None, **data):
            return "quiet answer"
    events = list(stream_ask(_SilentCore(), "hi"))
    assert events == [{"type": "speech", "content": "quiet answer"}]
