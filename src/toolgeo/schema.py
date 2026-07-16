from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Tool:
    tool_id: str
    name: str
    description: str
    schema: dict[str, Any]
    source: str

@dataclass(frozen=True)
class Decision:
    decision_id: str
    query: str
    candidate_tool_ids: list[str]
    gold_tool_id: str | None
    chosen_tool_id: str | None
    source: str
    # candidate_tool_ids is the complete benchmark-provided ordered menu.
    gold_position: int | None = None
    chosen_position: int | None = None
    menu_order_seed: int | None = None
    menu_variant_id: str | None = None
    gold_call_count: int = 1

@dataclass(frozen=True)
class Trace:
    trace_id: str
    tool_ids: list[str]
    source: str

def record(item: Tool | Decision | Trace) -> dict[str, Any]:
    return asdict(item)
