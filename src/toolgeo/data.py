from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_jsonl
from .schema import Decision, Tool, Trace


def _tool(row: dict[str, Any]) -> Tool:
    return Tool(str(row["tool_id"]), str(row["name"]), str(row.get("description", "")), dict(row.get("schema", {})), str(row.get("source", "unknown")))
def _decision(row: dict[str, Any]) -> Decision:
    candidates = list(row["candidate_tool_ids"])
    gold = row.get("gold_tool_id")
    chosen = row.get("chosen_tool_id")
    return Decision(
        str(row["decision_id"]), str(row["query"]), candidates, gold, chosen,
        str(row.get("source", "unknown")),
        row.get("gold_position", candidates.index(gold) if gold in candidates else None),
        row.get("chosen_position", candidates.index(chosen) if chosen in candidates else None),
        row.get("menu_order_seed"), row.get("menu_variant_id"),
        int(row.get("gold_call_count", 1)),
    )
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
    decision_ids = {item.decision_id for item in decisions}
    trace_ids = {item.trace_id for item in traces}
    if not tools: errors.append("dataset must contain at least one tool")
    if not decisions: errors.append("dataset must contain at least one decision")
    if len(ids) != len(tools): errors.append("tool_id values must be unique")
    if len(decision_ids) != len(decisions): errors.append("decision_id values must be unique")
    if len(trace_ids) != len(traces): errors.append("trace_id values must be unique")
    for item in decisions:
        if not item.query.strip(): errors.append(f"{item.decision_id}: query is empty")
        if len(item.candidate_tool_ids) < 2: errors.append(f"{item.decision_id}: risk set has fewer than two candidates")
        if item.gold_call_count < 1: errors.append(f"{item.decision_id}: gold_call_count must be positive")
        if len(item.candidate_tool_ids) != len(set(item.candidate_tool_ids)):
            errors.append(f"{item.decision_id}: duplicate candidates")
        unknown = set(item.candidate_tool_ids) - ids
        if unknown: errors.append(f"{item.decision_id}: unknown candidates {sorted(unknown)}")
        if item.gold_tool_id and item.gold_tool_id not in ids: errors.append(f"{item.decision_id}: unknown gold")
        if item.gold_tool_id and item.gold_tool_id not in item.candidate_tool_ids:
            errors.append(f"{item.decision_id}: gold is absent from risk set")
        if item.chosen_tool_id and item.chosen_tool_id not in item.candidate_tool_ids:
            errors.append(f"{item.decision_id}: choice is absent from risk set")
        if item.gold_position is not None and (item.gold_position < 0 or item.gold_position >= len(item.candidate_tool_ids) or item.candidate_tool_ids[item.gold_position] != item.gold_tool_id):
            errors.append(f"{item.decision_id}: gold_position does not match ordered risk set")
        if item.chosen_position is not None and (item.chosen_position < 0 or item.chosen_position >= len(item.candidate_tool_ids) or item.candidate_tool_ids[item.chosen_position] != item.chosen_tool_id):
            errors.append(f"{item.decision_id}: chosen_position does not match ordered risk set")
    for item in traces:
        unknown = set(item.tool_ids) - ids
        if unknown: errors.append(f"{item.trace_id}: unknown trace tools {sorted(unknown)}")
    return errors
