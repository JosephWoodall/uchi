from uchi.tool_calling import (
    ToolRegistry,
    default_registry,
    format_tool_result,
    parse_tool_call,
    run_with_tools,
)


def test_parse_tool_call_basic():
    text = 'before <|tool_call|> write_file(path="a.txt", content="hi") <|end_tool|> after'
    call = parse_tool_call(text)
    assert call is not None
    assert call.name == "write_file"
    assert call.args == {"path": "a.txt", "content": "hi"}


def test_parse_tool_call_handles_commas_inside_quoted_strings():
    text = '<|tool_call|> write_file(path="a.txt", content="one, two, three") <|end_tool|>'
    call = parse_tool_call(text)
    assert call.args == {"path": "a.txt", "content": "one, two, three"}


def test_parse_tool_call_no_match_returns_none():
    assert parse_tool_call("just plain text, no tool call here") is None


def test_parse_tool_call_no_args():
    text = "<|tool_call|> list_files() <|end_tool|>"
    call = parse_tool_call(text)
    assert call.name == "list_files"
    assert call.args == {}


def test_registry_dispatch_success():
    registry = ToolRegistry()
    registry.register("add", lambda a, b: a + b)
    from uchi.tool_calling import ToolCall
    entry = registry.dispatch(ToolCall(name="add", args={"a": 2, "b": 3}, raw="x"))
    assert entry.ok
    assert entry.result == "5"
    assert registry.log == [entry]


def test_registry_dispatch_unknown_tool():
    registry = ToolRegistry()
    from uchi.tool_calling import ToolCall
    entry = registry.dispatch(ToolCall(name="nope", args={}, raw="x"))
    assert not entry.ok
    assert "unknown tool" in entry.error


def test_registry_dispatch_captures_exception():
    registry = ToolRegistry()
    def boom():
        raise ValueError("kaboom")
    registry.register("boom", boom)
    from uchi.tool_calling import ToolCall
    entry = registry.dispatch(ToolCall(name="boom", args={}, raw="x"))
    assert not entry.ok
    assert "kaboom" in entry.error


def test_format_tool_result_success_and_error():
    from uchi.tool_calling import ToolCallLogEntry
    ok_entry = ToolCallLogEntry("t", {}, "result!", None)
    err_entry = ToolCallLogEntry("t", {}, None, "boom")
    assert format_tool_result(ok_entry) == "<|tool_result|> t -> result! <|end_result|>"
    assert format_tool_result(err_entry) == "<|tool_error|> t -> boom <|end_result|>"


def test_run_with_tools_splices_result_into_text():
    registry = ToolRegistry()
    registry.register("add", lambda a, b: a + b)
    text = 'Let me compute that. <|tool_call|> add(a=2, b=3) <|end_tool|> done.'
    out = run_with_tools(text, registry)
    assert "<|tool_call|>" not in out
    assert "<|tool_result|> add -> 5 <|end_result|>" in out
    assert "Let me compute that." in out
    assert "done." in out


def test_run_with_tools_no_call_is_noop():
    registry = ToolRegistry()
    text = "nothing to do here"
    assert run_with_tools(text, registry) == text


def test_run_with_tools_respects_max_hops():
    # A tool whose result text itself looks like another tool call — bounded
    # so this can't loop forever (full loop-prevention lands in Item 6).
    registry = ToolRegistry()
    registry.register(
        "echo_call", lambda: "<|tool_call|> echo_call() <|end_tool|>"
    )
    text = "<|tool_call|> echo_call() <|end_tool|>"
    out = run_with_tools(text, registry, max_hops=2)
    assert out.count("<|tool_call|>") <= 1


def test_default_registry_has_workspace_and_scratchpad_tools(tmp_path):
    registry = default_registry()
    for name in ("read_file", "write_file", "list_files", "delete_file", "run_python"):
        assert registry.has(name), f"missing default tool: {name}"


def test_default_registry_write_then_read_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    registry = default_registry()
    from uchi.tool_calling import ToolCall
    write_entry = registry.dispatch(
        ToolCall(name="write_file", args={"path": "note.txt", "content": "hi"}, raw="x")
    )
    assert write_entry.ok
    read_entry = registry.dispatch(
        ToolCall(name="read_file", args={"path": "note.txt"}, raw="x")
    )
    assert read_entry.ok
    assert read_entry.result == "hi"


def test_default_registry_run_python(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    registry = default_registry()
    from uchi.tool_calling import ToolCall
    entry = registry.dispatch(
        ToolCall(name="run_python", args={"code": "print(2 + 2)"}, raw="x")
    )
    assert entry.ok
    assert "4" in entry.result
