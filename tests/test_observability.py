import json

from uchi.observability import entry_to_span, export_spans
from uchi.tool_calling import ToolCallLogEntry


def test_entry_to_span_success_shape():
    entry = ToolCallLogEntry("add", {"a": 1, "b": 2}, "3", None, timestamp=1000.0, duration_seconds=0.05)
    span = entry_to_span(entry, trace_id="trace123")
    assert span["name"] == "add"
    assert span["trace_id"] == "trace123"
    assert span["attributes"]["args"] == {"a": 1, "b": 2}
    assert span["status"] == {"code": "OK"}
    assert span["end_time_unix_nano"] == int(1000.0 * 1e9)
    assert span["start_time_unix_nano"] == int(1000.0 * 1e9) - int(0.05 * 1e9)


def test_entry_to_span_error_shape():
    entry = ToolCallLogEntry("run_python", {}, None, "ValueError: boom", timestamp=1000.0)
    span = entry_to_span(entry, trace_id="t")
    assert span["status"] == {"code": "ERROR", "message": "ValueError: boom"}


def test_span_ids_are_unique():
    entry = ToolCallLogEntry("a", {}, "ok", None)
    s1 = entry_to_span(entry, trace_id="t")
    s2 = entry_to_span(entry, trace_id="t")
    assert s1["span_id"] != s2["span_id"]


def test_export_spans_returns_one_span_per_log_entry():
    log = [
        ToolCallLogEntry("a", {}, "ok", None),
        ToolCallLogEntry("b", {}, None, "boom"),
    ]
    spans = export_spans(log)
    assert len(spans) == 2
    assert spans[0]["name"] == "a"
    assert spans[1]["name"] == "b"
    # all spans from one export share a trace_id
    assert spans[0]["trace_id"] == spans[1]["trace_id"]


def test_export_spans_writes_valid_jsonl(tmp_path):
    log = [ToolCallLogEntry("a", {}, "ok", None), ToolCallLogEntry("b", {}, "ok", None)]
    out_path = str(tmp_path / "spans.jsonl")
    export_spans(log, out_path=out_path)

    lines = open(out_path).read().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)  # must not raise


def test_export_spans_appends_across_calls(tmp_path):
    out_path = str(tmp_path / "spans.jsonl")
    export_spans([ToolCallLogEntry("a", {}, "ok", None)], out_path=out_path)
    export_spans([ToolCallLogEntry("b", {}, "ok", None)], out_path=out_path)
    lines = open(out_path).read().strip().splitlines()
    assert len(lines) == 2
