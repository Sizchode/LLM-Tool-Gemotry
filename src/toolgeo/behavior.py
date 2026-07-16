from __future__ import annotations

import numpy as np

from .schema import Decision, Trace, Tool

def matrices(tools: list[Tool], decisions: list[Decision], traces: list[Trace]) -> dict[str, np.ndarray]:
    n = len(tools); index = {tool.tool_id: i for i, tool in enumerate(tools)}
    confusion = np.zeros((n, n)); cooccur = np.zeros((n, n)); order = np.zeros((n, n)); substitute = np.zeros((n, n))
    for decision in decisions:
        if not decision.gold_tool_id: continue
        gold = index[decision.gold_tool_id]
        if decision.chosen_tool_id and decision.chosen_tool_id in index and decision.chosen_tool_id != decision.gold_tool_id:
            confusion[gold, index[decision.chosen_tool_id]] += 1
        candidates = [index[item] for item in decision.candidate_tool_ids if item in index]
        for left in candidates:
            for right in candidates:
                if left != right: substitute[left, right] += 1
    for trace in traces:
        values = [index[item] for item in trace.tool_ids if item in index]
        for left in values:
            for right in values:
                if left != right: cooccur[left, right] += 1
        for left, right in zip(values, values[1:]): order[left, right] += 1
    return {"confusion": confusion, "cooccurrence": cooccur, "order": order, "substitutability": substitute}
