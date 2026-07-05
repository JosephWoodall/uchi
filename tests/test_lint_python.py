from uchi.scratchpad import lint_python
from uchi.tool_calling import ToolCall, default_registry


def test_clean_code_reports_no_issues(tmp_path):
    result = lint_python("def add(a, b):\n    return a + b\n", root=str(tmp_path))
    assert result == "No issues found."


def test_unused_import_is_flagged(tmp_path):
    result = lint_python("import os\nx = 1\n", root=str(tmp_path))
    assert "unused" in result.lower() or "F401" in result


def test_syntax_error_is_flagged(tmp_path):
    result = lint_python("def broken(:\n    pass\n", root=str(tmp_path))
    assert result != "No issues found."


def test_lint_python_registered_as_a_tool(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    registry = default_registry()
    assert registry.has("lint_python")
    entry = registry.dispatch(ToolCall(name="lint_python", args={"code": "import os\nx=1"}, raw="x"))
    assert entry.ok  # findings are informative output, not a failed dispatch
    assert "unused" in entry.result.lower() or "F401" in entry.result


def test_scratch_file_cleaned_up_after_lint(tmp_path):
    lint_python("x = 1\n", root=str(tmp_path))
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith("_lint_")]
    assert leftovers == []
