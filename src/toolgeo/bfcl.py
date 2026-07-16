"""Official BFCL v4 live_multiple -> normalized Paper-1 tables."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import write_json, write_jsonl
from .schema import Decision, Tool, record
from .sources import BFCL_REVISION, download

_ROOT = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/"
    f"{BFCL_REVISION}/berkeley-function-call-leaderboard/bfcl_eval/data"
)
_DATA_URL = f"{_ROOT}/BFCL_v4_live_multiple.json"
_ANSWER_URL = f"{_ROOT}/possible_answer/BFCL_v4_live_multiple.json"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _slug(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_").lower()
    return clean or "unnamed"


def _query(row: dict[str, Any]) -> str:
    turns = row.get("question", [])
    messages = [message for turn in turns for message in turn]
    user = [str(message.get("content", "")) for message in messages if message.get("role") == "user"]
    if not user:
        raise ValueError(f"{row.get('id')}: BFCL row has no user message")
    return "\n".join(user)


def export(output: Path, input_file: Path | None = None, answer_file: Path | None = None) -> tuple[int, int]:
    source = input_file or output / "official" / "BFCL_v4_live_multiple.json"
    answers_source = answer_file or output / "official" / "possible_answer" / "BFCL_v4_live_multiple.json"
    if input_file is None and not source.exists():
        download(_DATA_URL, source)
    if answer_file is None and not answers_source.exists():
        download(_ANSWER_URL, answers_source)

    rows = _jsonl(source)
    answers = _jsonl(answers_source)
    answer_by_id = {str(row["id"]): row for row in answers}
    if len(answer_by_id) != len(answers):
        raise ValueError("BFCL possible-answer IDs are not unique")

    # BFCL reuses names while varying descriptions and schemas.  Exact
    # definitions are one representation object; variants receive stable
    # human-readable ordinals rather than being silently collapsed by name.
    definitions: dict[str, dict[str, Any]] = {}
    variants: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        for function in row["function"]:
            canonical = json.dumps(function, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            definitions[canonical] = function
            variants[_slug(str(function["name"]))].add(canonical)
    tool_id: dict[str, str] = {}
    for slug, values in variants.items():
        for number, canonical in enumerate(sorted(values), 1):
            tool_id[canonical] = f"bfcl_v4.{slug}.v{number:03d}"

    tools = [
        Tool(
            tool_id[canonical], str(function["name"]), str(function.get("description", "")),
            dict(function.get("parameters", {})), "bfcl_v4_live_multiple",
        )
        for canonical, function in sorted(definitions.items(), key=lambda item: tool_id[item[0]])
    ]
    decisions: list[Decision] = []
    gold_calls: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in rows:
        row_id = str(row["id"])
        if row_id in seen_ids:
            raise ValueError(f"duplicate BFCL query ID: {row_id}")
        seen_ids.add(row_id)
        if row_id not in answer_by_id:
            raise ValueError(f"{row_id}: missing BFCL possible answer")
        ground_truth = answer_by_id[row_id].get("ground_truth", [])
        if len(ground_truth) != 1 or len(ground_truth[0]) != 1:
            raise ValueError(f"{row_id}: expected exactly one gold function call")
        gold_name = str(next(iter(ground_truth[0])))
        functions = row["function"]
        names = [str(function["name"]) for function in functions]
        if len(names) != len(set(names)):
            raise ValueError(f"{row_id}: candidate function names are ambiguous")
        if gold_name not in names:
            raise ValueError(f"{row_id}: gold function {gold_name!r} is absent from menu")
        candidates = [
            tool_id[json.dumps(function, ensure_ascii=False, sort_keys=True, separators=(",", ":"))]
            for function in functions
        ]
        gold = candidates[names.index(gold_name)]
        decisions.append(Decision(
            row_id, _query(row), candidates, gold, None, "bfcl_v4_live_multiple",
            candidates.index(gold), None, None, "original", 1,
        ))
        gold_calls.append({"decision_id": row_id, "ground_truth": ground_truth})

    if set(answer_by_id) != seen_ids:
        raise ValueError("BFCL input and possible-answer ID sets differ")
    write_jsonl(output / "tools.jsonl", (record(item) for item in tools))
    write_jsonl(output / "decisions.jsonl", (record(item) for item in decisions))
    write_jsonl(output / "traces.jsonl", [])
    write_jsonl(output / "gold_calls.jsonl", gold_calls)
    write_json(output / "source_manifest.json", {
        "dataset": "BFCL_v4_live_multiple",
        "source_mode": "pinned_download" if input_file is None and answer_file is None else "local_files",
        "revision": BFCL_REVISION if input_file is None and answer_file is None else None,
        "data_source": _DATA_URL if input_file is None else str(input_file),
        "answer_source": _ANSWER_URL if answer_file is None else str(answer_file),
        "rows": len(rows), "tools": len(tools), "decisions": len(decisions),
    })
    return len(tools), len(decisions)
