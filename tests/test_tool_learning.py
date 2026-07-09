import os

from uchi.tool_calling import ToolCall, ToolRegistry
from uchi.tool_learning import learn_tools

SAMPLE_MODULE = '''
def add(a, b):
    """Add two numbers together."""
    return a + b


def greet(name: str) -> str:
    """Return a friendly greeting for *name*."""
    return f"Hello, {name}!"


class NotATopLevelFunction:
    def method_should_not_register(self):
        return "nope"


_private_import_style = add  # aliasing shouldn't create a second registration
'''


def _write_module(tmp_path):
    path = tmp_path / "sample_tools.py"
    path.write_text(SAMPLE_MODULE)
    return str(path)


def test_learn_tools_registers_top_level_functions(tmp_path):
    path = _write_module(tmp_path)
    registry = ToolRegistry()
    learned = learn_tools(path, registry)
    names = {t.name for t in learned}
    assert names == {"add", "greet"}
    assert registry.has("add")
    assert registry.has("greet")


def test_learn_tools_skips_methods_and_aliases(tmp_path):
    path = _write_module(tmp_path)
    registry = ToolRegistry()
    learn_tools(path, registry)
    assert not registry.has("method_should_not_register")
    assert not registry.has("_private_import_style")


def test_learned_tool_captures_signature_and_docstring(tmp_path):
    path = _write_module(tmp_path)
    registry = ToolRegistry()
    learned = {t.name: t for t in learn_tools(path, registry)}
    assert learned["add"].signature == "(a, b)"
    assert learned["add"].docstring == "Add two numbers together."
    assert learned["greet"].signature == "(name: str) -> str"


def test_learned_tool_is_actually_callable_through_the_registry(tmp_path):
    path = _write_module(tmp_path)
    registry = ToolRegistry()
    learn_tools(path, registry)
    entry = registry.dispatch(ToolCall(name="add", args={"a": 2, "b": 3}, raw="x"))
    assert entry.ok
    assert entry.result == "5"

    entry2 = registry.dispatch(ToolCall(name="greet", args={"name": "Uchi"}, raw="x"))
    assert entry2.ok
    assert entry2.result == "Hello, Uchi!"


def test_as_knowledge_rendering(tmp_path):
    path = _write_module(tmp_path)
    registry = ToolRegistry()
    learned = {t.name: t for t in learn_tools(path, registry)}
    text = learned["add"].as_knowledge()
    assert text == "Tool available: add(a, b) — Add two numbers together."


def test_empty_file_learns_nothing(tmp_path):
    path = tmp_path / "empty.py"
    path.write_text("x = 1\n")
    registry = ToolRegistry()
    learned = learn_tools(str(path), registry)
    assert learned == []
