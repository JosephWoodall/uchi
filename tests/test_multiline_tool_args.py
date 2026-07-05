from uchi.tool_calling import ToolRegistry, parse_tool_call, run_with_tools


def test_multiline_code_argument_parses_correctly():
    """Regression test: a raw embedded newline inside a quoted string
    argument used to make ast.parse raise SyntaxError, silently caught
    by _parse_kwargs and turned into an empty {} args dict -- breaking
    every multi-line run_python/lint_python call."""
    text = '<|tool_call|> run_python(code="x = 1\ny = 2\nprint(x + y)") <|end_tool|>'
    call = parse_tool_call(text)
    assert call is not None
    assert call.args == {"code": "x = 1\ny = 2\nprint(x + y)"}


def test_multiline_argument_with_escaped_quote_inside():
    text = '<|tool_call|> run_python(code="print(\\"hi\\")\nprint(1)") <|end_tool|>'
    call = parse_tool_call(text)
    assert call.args == {"code": 'print("hi")\nprint(1)'}


def test_multiline_argument_with_multiple_kwargs():
    text = '<|tool_call|> write_file(path="a.txt", content="line1\nline2\nline3") <|end_tool|>'
    call = parse_tool_call(text)
    assert call.args == {"path": "a.txt", "content": "line1\nline2\nline3"}


def test_newline_between_args_outside_strings_still_works():
    text = '<|tool_call|> add(\n  a=1,\n  b=2\n) <|end_tool|>'
    call = parse_tool_call(text)
    assert call.args == {"a": 1, "b": 2}


def test_multiline_run_python_actually_executes_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from uchi.tool_calling import default_registry
    registry = default_registry()
    text = '<|tool_call|> run_python(code="total = 0\nfor i in range(5):\n    total += i\nprint(total)") <|end_tool|>'
    out = run_with_tools(text, registry)
    assert "10" in out
    assert "<|tool_call|>" not in out
