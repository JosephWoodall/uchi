"""tool_calling.py — formalized tool-calling grammar (0.4.0 Items 4 & 7).

Uchi can emit a token sequence:

    <|tool_call|> tool_name(arg1=value1, arg2="value2") <|end_tool|>

This halts the response, dispatches to a registered Python function, and
the result is spliced back into the text as
``<|tool_result|> tool_name -> ... <|end_result|>`` (or ``<|tool_error|>``
on failure) before the caller resumes. Initial tools are the Item 2
filesystem operations and the Item 3 Python scratchpad.

A second grammar element, ``<|tool_call_async|> ... <|end_tool|>`` (Item 7),
lets several independent calls in the same response fire concurrently
instead of one at a time — every async call in a text is dispatched via a
thread pool and results land back in their original positions once all
complete, rather than blocking on each one serially.

Every dispatch is recorded on ``ToolRegistry.log`` (tool, args, result or
error, timestamp) — logging from day one rather than retrofitting a trace
format once Items 9/14 exist.
"""
from __future__ import annotations

import ast
import concurrent.futures
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from .goal_state import GoalState

_TOOL_CALL_RE = re.compile(
    r"<\|tool_call\|>\s*(\w+)\((.*?)\)\s*<\|end_tool\|>",
    re.DOTALL,
)
_ASYNC_TOOL_CALL_RE = re.compile(
    r"<\|tool_call_async\|>\s*(\w+)\((.*?)\)\s*<\|end_tool\|>",
    re.DOTALL,
)


@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]
    raw: str


@dataclass
class ToolCallLogEntry:
    name: str
    args: Dict[str, Any]
    result: Optional[str]
    error: Optional[str]
    timestamp: float = field(default_factory=time.time)
    duration_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


def parse_tool_call(text: str) -> Optional[ToolCall]:
    """Find the first ``<|tool_call|> name(...) <|end_tool|>`` in *text*."""
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None
    name, argstr = m.group(1), m.group(2)
    return ToolCall(name=name, args=_parse_kwargs(argstr), raw=m.group(0))


def parse_async_tool_calls(text: str) -> List[ToolCall]:
    """Find every ``<|tool_call_async|> name(...) <|end_tool|>`` in *text*."""
    calls = []
    for m in _ASYNC_TOOL_CALL_RE.finditer(text):
        name, argstr = m.group(1), m.group(2)
        calls.append(ToolCall(name=name, args=_parse_kwargs(argstr), raw=m.group(0)))
    return calls


def _escape_embedded_newlines(argstr: str) -> str:
    """Escape raw newlines that fall inside a quoted string literal.

    Tool-call arguments legitimately contain multi-line code (``run_python``,
    ``lint_python``) — but a raw, unescaped newline inside a single/double
    -quoted Python string literal is a ``SyntaxError``. Left unhandled,
    ``_parse_kwargs``'s broad ``except`` would silently swallow that and
    return ``{}``, turning any multi-line ``code="..."`` argument into a
    missing-argument error instead of running the code. Track quote state
    char-by-char (respecting backslash-escapes) and escape only the
    newlines that fall inside an open string.
    """
    out = []
    in_string = None
    escape_next = False
    for ch in argstr:
        if in_string:
            if escape_next:
                out.append(ch)
                escape_next = False
            elif ch == "\\":
                out.append(ch)
                escape_next = True
            elif ch == in_string:
                in_string = None
                out.append(ch)
            elif ch == "\n":
                out.append("\\n")
            else:
                out.append(ch)
        else:
            if ch in ("'", '"'):
                in_string = ch
            out.append(ch)
    return "".join(out)


def _parse_kwargs(argstr: str) -> Dict[str, Any]:
    """Parse ``key=value, key2=value2`` via ast so quoted strings containing
    commas/parens are handled correctly, not split naively on ','."""
    argstr = argstr.strip()
    if not argstr:
        return {}
    try:
        safe_argstr = _escape_embedded_newlines(argstr)
        node = ast.parse(f"f({safe_argstr})", mode="eval").body
        return {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
    except Exception:
        return {}


class ToolRegistry:
    """Name -> callable registry with execution logging and loop guarding."""

    def __init__(self) -> None:
        self._tools: Dict[str, Callable[..., Any]] = {}
        self.log: List[ToolCallLogEntry] = []
        from .loop_guard import LoopGuard
        self.loop_guard = LoopGuard()

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        self._tools[name] = fn

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> List[str]:
        return sorted(self._tools)

    @staticmethod
    def _signature(call: ToolCall) -> str:
        return f"{call.name}({sorted(call.args.items())!r})"

    def dispatch(self, call: ToolCall) -> ToolCallLogEntry:
        fn = self._tools.get(call.name)
        if fn is None:
            entry = ToolCallLogEntry(call.name, call.args, None, f"unknown tool: {call.name!r}")
            self.log.append(entry)
            return entry

        sig = self._signature(call)
        if self.loop_guard.is_penalized(sig):
            n = self.loop_guard.failure_count(sig)
            entry = ToolCallLogEntry(
                call.name, call.args, None,
                f"blocked: an identical call to {call.name!r} with the same arguments "
                f"already failed {n} time(s) in a row — try a structurally different approach",
            )
            self.log.append(entry)
            return entry

        t0 = time.perf_counter()
        try:
            result = fn(**call.args)
            elapsed = time.perf_counter() - t0
            entry = ToolCallLogEntry(call.name, call.args, str(result), None, duration_seconds=elapsed)
            self.loop_guard.record_success(sig)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            entry = ToolCallLogEntry(
                call.name, call.args, None, f"{type(e).__name__}: {e}", duration_seconds=elapsed
            )
            self.loop_guard.record_failure(sig)
        self.log.append(entry)
        return entry


def format_tool_result(entry: ToolCallLogEntry) -> str:
    """Render a dispatched tool call as text tokens to splice back into context."""
    if entry.ok:
        return f"<|tool_result|> {entry.name} -> {entry.result} <|end_result|>"
    return f"<|tool_error|> {entry.name} -> {entry.error} <|end_result|>"


def run_with_tools(
    text: str,
    registry: ToolRegistry,
    max_hops: int = 3,
    goal_state: Optional["GoalState"] = None,
) -> str:
    """Execute every tool call found in *text*, splicing results back in.

    Bounded to *max_hops* replacements — a tool call producing text that
    itself parses as another tool call could otherwise loop forever;
    strict loop prevention on repeated-failure sequences is Item 6.

    If *goal_state* is given (Item 5), every dispatched entry is recorded
    into it and compaction is checked after each hop, so a long tool-call
    chain never lets raw logs balloon past the compaction threshold.
    """
    for _ in range(max_hops):
        call = parse_tool_call(text)
        if call is None:
            break
        entry = registry.dispatch(call)
        if goal_state is not None:
            goal_state.record(entry)
            goal_state.maybe_compact()
        text = text.replace(call.raw, format_tool_result(entry), 1)
    return text


def run_async_tool_calls(
    text: str,
    registry: ToolRegistry,
    goal_state: Optional["GoalState"] = None,
    max_workers: int = 8,
) -> str:
    """Dispatch every ``<|tool_call_async|>`` in *text* concurrently.

    All calls fire at once via a thread pool (matching the pattern already
    proven in ``swarm.py``'s parallel sub-question dispatch) instead of
    blocking on each one serially — the caller's internal monologue isn't
    held up waiting on, say, 5 web pages one at a time. Each result is
    spliced back into its original position once every call has completed.
    """
    matches = list(_ASYNC_TOOL_CALL_RE.finditer(text))
    if not matches:
        return text

    def _dispatch(m: "re.Match") -> ToolCallLogEntry:
        call = ToolCall(name=m.group(1), args=_parse_kwargs(m.group(2)), raw=m.group(0))
        entry = registry.dispatch(call)
        if goal_state is not None:
            goal_state.record(entry)
            goal_state.maybe_compact()
        return entry

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(matches))) as pool:
        entries = list(pool.map(_dispatch, matches))

    result_text = text
    for m, entry in sorted(zip(matches, entries), key=lambda pair: pair[0].start(), reverse=True):
        result_text = result_text[:m.start()] + format_tool_result(entry) + result_text[m.end():]
    return result_text


def _run_python_tool(code: str, timeout: float = 5.0) -> str:
    """Registry-facing wrapper around ``scratchpad.run_python``.

    Raises when the *executed* code failed (nonzero exit / timeout) so
    ``ToolRegistry.dispatch``'s exception-based success/failure detection
    — which drives both the loop guard (Item 6) and HitL auto-escalation
    (Item 10) — actually sees a scratchpad failure as a failure, not a
    successful call that merely reports an error as its result text.
    """
    from .scratchpad import run_python
    result = run_python(code, timeout=timeout)
    if not result.ok:
        raise RuntimeError(result.as_text())
    return result.as_text()


def _web_search_tool(query: str, sources=None) -> str:
    """Registry-facing wrapper around ``web_search.perform_web_search``
    (0.4.0 Item 11). Static, structured retrieval (DuckDuckGo, Wikipedia,
    arXiv, News RSS) exposed as a tool call — scoped down from full
    browser automation per the 0.4.0 audit, since this alone covers most
    "look something up" goals without a Chromium dependency.
    """
    from .web_search import perform_web_search
    result = perform_web_search(query, sources=sources)
    if not result:
        raise RuntimeError(f"no web results found for query: {query!r}")
    return result


def default_registry(enable_web_search: bool = False) -> ToolRegistry:
    """Registry with the Item 2 filesystem tools, Item 3 scratchpad, and
    (if *enable_web_search*) the Item 11 web search tool wired in.

    *enable_web_search* mirrors ``Core(web_search=...)`` — off by default
    so the brain runs fully offline unless a caller opts in.
    """
    from . import workspace

    registry = ToolRegistry()
    from .scratchpad import lint_python

    registry.register("read_file", workspace.read_file)
    registry.register("write_file", workspace.write_file)
    registry.register("list_files", workspace.list_files)
    registry.register("delete_file", workspace.delete_file)
    registry.register("run_python", _run_python_tool)
    registry.register("lint_python", lint_python)
    if enable_web_search:
        registry.register("web_search", _web_search_tool)
    return registry
