"""observability.py — Enterprise Observability exporter (0.4.0 Item 16.8).

Extends the Glass Brain trace (``ToolRegistry.log``, already built in
Item 4) with an OpenTelemetry-*shaped* span exporter — without adding the
``opentelemetry`` SDK as a dependency or requiring a running collector.
Each tool call becomes a span dict following the OTel data model (name,
trace/span IDs, start/end timestamps, attributes, status), written as
JSON Lines. That's a format most log pipelines (Datadog, Grafana Loki,
LangSmith) already ingest directly, and it's a trivial adapter away from
a real OTLP exporter if a team wants to add the full SDK later — this
gives the structured trace without committing to a heavy dependency that
can't be verified against a real collector in this environment.
"""
from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .tool_calling import ToolCallLogEntry


def entry_to_span(entry: "ToolCallLogEntry", trace_id: str) -> Dict[str, Any]:
    """Render one ``ToolCallLogEntry`` as an OTel-shaped span dict."""
    end_ns = int(entry.timestamp * 1e9)
    start_ns = end_ns - int(entry.duration_seconds * 1e9)
    return {
        "name": entry.name,
        "trace_id": trace_id,
        "span_id": uuid.uuid4().hex[:16],
        "start_time_unix_nano": start_ns,
        "end_time_unix_nano": end_ns,
        "attributes": {"args": entry.args},
        "status": (
            {"code": "OK"}
            if entry.ok
            else {"code": "ERROR", "message": entry.error}
        ),
    }


def export_spans(
    log: List["ToolCallLogEntry"],
    out_path: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convert *log* into OTel-shaped spans, optionally writing them as
    JSON Lines to *out_path* (appends). Returns the span dicts either way."""
    trace_id = trace_id or uuid.uuid4().hex
    spans = [entry_to_span(entry, trace_id) for entry in log]

    if out_path:
        with open(out_path, "a", encoding="utf-8") as fh:
            for span in spans:
                fh.write(json.dumps(span) + "\n")

    return spans
