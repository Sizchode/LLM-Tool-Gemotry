"""Official Seal-Tools → normalized Paper-1 tables adapter."""
from __future__ import annotations

import ast
import re
from pathlib import Path

from .io import write_jsonl
from .schema import Decision, Tool

_APIS = re.compile(r"api_list\s*=\s*(\[.*?\])\s*\ntask_instruction\s*=\s*\"(.*?)\"\s*\nOutput:", re.S)

def export(split: str, output: Path, limit: int | None = None) -> tuple[int, int]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install toolgeo[datasets] for Seal-Tools import.") from exc
    dataset = load_dataset("casey-martin/Seal-Tools", split=split)
    tools: dict[str, Tool] = {}; decisions: list[Decision] = []
    for position, row in enumerate(dataset):
        if limit is not None and position >= limit: break
        human = next(item["value"] for item in row["conversations"] if item["from"] == "human")
        answer = next(item["value"] for item in row["conversations"] if item["from"] == "gpt")
        match = _APIS.search(human)
        if not match: continue
        apis, query = ast.literal_eval(match.group(1)), match.group(2)
        candidates = []
        for api in apis:
            identifier = "seal." + api["api_name"]
            candidates.append(identifier)
            tools.setdefault(identifier, Tool(identifier, api["api_name"], api.get("api_description", ""), {"type":"object", "properties":api.get("parameters", {}), "required":api.get("required", [])}, "seal_tools"))
        calls = ast.literal_eval(answer)
        gold = calls[0]["api"] if calls else None
        gold_id = "seal." + gold if gold else None
        # Gold is a benchmark label, never an observed model behaviour.
        decisions.append(Decision(str(row["id"]), query, candidates, gold_id, None, "seal_tools"))
    write_jsonl(output / "tools.jsonl", (tool.__dict__ for tool in tools.values()))
    write_jsonl(output / "decisions.jsonl", (decision.__dict__ for decision in decisions))
    write_jsonl(output / "traces.jsonl", [])
    return len(tools), len(decisions)
