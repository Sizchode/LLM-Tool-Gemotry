"""Official ToolHop -> benchmark-provided step decisions and traces."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .io import write_json, write_jsonl
from .schema import Decision, Tool, Trace, record
from .sources import TOOLHOP_REVISION, download

_URL = f"https://huggingface.co/datasets/bytedance-research/ToolHop/resolve/{TOOLHOP_REVISION}/data/ToolHop.json"


def _slug(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_").lower()
    return clean or "unnamed"


def export(output: Path, input_file: Path | None = None) -> tuple[int, int, int]:
    source = input_file or output / "official" / "ToolHop.json"
    if input_file is None and not source.exists():
        download(_URL, source)
    with source.open(encoding="utf-8") as handle:
        rows: list[dict[str, Any]] = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError("ToolHop source must be a JSON array")

    tools: list[Tool] = []
    decisions: list[Decision] = []
    traces: list[Trace] = []
    executables: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    seen_rows: set[str] = set()
    for row in rows:
        row_id = str(row["id"])
        if row_id in seen_rows:
            raise ValueError(f"duplicate ToolHop row ID: {row_id}")
        seen_rows.add(row_id)
        subtasks = list(row["sub_task"].items())
        specifications = list(row["tools"].items())
        functions = list(row.get("functions", []))
        if len(subtasks) != len(specifications) or len(subtasks) != len(functions):
            raise ValueError(f"toolhop_{row_id}: sub_task/tools/functions lengths differ")

        row_tool_ids: list[str] = []
        rendered_names: list[str] = []
        raw_names = [str(specification[1]["name"]) for specification in specifications]
        duplicate_names = {name for name in raw_names if raw_names.count(name) > 1}
        for step, (_, specification) in enumerate(specifications):
            raw_name = str(specification["name"])
            rendered_name = f"{raw_name}__step_{step + 1}" if raw_name in duplicate_names else raw_name
            identifier = f"toolhop.{row_id}.step_{step + 1}.{_slug(raw_name)}"
            schema = dict(specification.get("parameters", {}))
            if rendered_name != raw_name:
                schema = {**schema, "x-toolhop-original-name": raw_name}
            tools.append(Tool(
                identifier, rendered_name, str(specification.get("description", "")),
                schema, "toolhop",
            ))
            row_tool_ids.append(identifier)
            rendered_names.append(rendered_name)
            executables.append({
                "tool_id": identifier, "python_source": str(functions[step]),
                "source": "toolhop", "execution_policy": "untrusted_not_executed_by_importer",
            })

        if len(rendered_names) != len(set(rendered_names)):
            raise ValueError(f"toolhop_{row_id}: rendered candidate names remain ambiguous")
        for step, ((subtask, _), _) in enumerate(zip(subtasks, specifications)):
            gold = row_tool_ids[step]
            decisions.append(Decision(
                f"toolhop_{row_id}_step_{step + 1}", str(subtask), list(row_tool_ids),
                gold, None, "toolhop", step, None, None, "original", 1,
            ))
        traces.append(Trace(f"toolhop_{row_id}", list(row_tool_ids), "toolhop"))
        trajectories.append({
            "trace_id": f"toolhop_{row_id}", "query": str(row["question"]),
            "answer": row.get("answer"), "domain": row.get("domain"),
            "answer_type": row.get("answer_type"),
            "previous_answer_type": row.get("previous_answer_type"),
            "steps": [
                {"step": step + 1, "subtask": str(subtask), "answer": answer, "tool_id": row_tool_ids[step]}
                for step, (subtask, answer) in enumerate(subtasks)
            ],
        })

    write_jsonl(output / "tools.jsonl", (record(item) for item in tools))
    write_jsonl(output / "decisions.jsonl", (record(item) for item in decisions))
    write_jsonl(output / "traces.jsonl", (record(item) for item in traces))
    write_jsonl(output / "executables.jsonl", executables)
    write_jsonl(output / "trajectories.jsonl", trajectories)
    write_json(output / "source_manifest.json", {
        "dataset": "ToolHop", "source_mode": "pinned_download" if input_file is None else "local_file",
        "revision": TOOLHOP_REVISION if input_file is None else None,
        "data_source": _URL if input_file is None else str(input_file),
        "rows": len(rows), "tools": len(tools), "decisions": len(decisions), "traces": len(traces),
        "decision_unit": "benchmark_provided_sub_task",
    })
    return len(tools), len(decisions), len(traces)
