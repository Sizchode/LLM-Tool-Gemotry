from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_jsonl
from .schema import Decision, Tool, Trace


def _tool(row: dict[str, Any]) -> Tool:
    return Tool(str(row["tool_id"]), str(row["name"]), str(row.get("description", "")), dict(row.get("schema", {})), str(row.get("source", "unknown")))
def _decision(row: dict[str, Any]) -> Decision:
    return Decision(str(row["decision_id"]), str(row["query"]), list(row["candidate_tool_ids"]), row.get("gold_tool_id"), row.get("chosen_tool_id"), str(row.get("source", "unknown")))
def _trace(row: dict[str, Any]) -> Trace:
    return Trace(str(row["trace_id"]), list(row["tool_ids"]), str(row.get("source", "unknown")))

def load_normalized(path: str | Path) -> tuple[list[Tool], list[Decision], list[Trace]]:
    root = Path(path)
    if root.is_dir():
        tools = [_tool(row) for row in read_jsonl(root / "tools.jsonl")]
        decisions = [_decision(row) for row in read_jsonl(root / "decisions.jsonl")]
        traces_path = root / "traces.jsonl"
        traces = [_trace(row) for row in read_jsonl(traces_path)] if traces_path.exists() else []
        return tools, decisions, traces
    rows = read_jsonl(root)
    return ([_tool(row) for row in rows if "tool_id" in row and "name" in row],
            [_decision(row) for row in rows if "decision_id" in row],
            [_trace(row) for row in rows if "trace_id" in row])

def validate(tools: list[Tool], decisions: list[Decision], traces: list[Trace]) -> list[str]:
    ids = {tool.tool_id for tool in tools}; errors: list[str] = []
    if len(ids) != len(tools): errors.append("tool_id values must be unique")
    for item in decisions:
        unknown = set(item.candidate_tool_ids) - ids
        if unknown: errors.append(f"{item.decision_id}: unknown candidates {sorted(unknown)}")
        if item.gold_tool_id and item.gold_tool_id not in ids: errors.append(f"{item.decision_id}: unknown gold")
    for item in traces:
        unknown = set(item.tool_ids) - ids
        if unknown: errors.append(f"{item.trace_id}: unknown trace tools {sorted(unknown)}")
    return errors
