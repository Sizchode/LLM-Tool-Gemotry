"""Behavioral summaries; choice-model inference lives in :mod:`analysis`."""
from __future__ import annotations

import numpy as np

from .schema import Decision, Trace, Tool


def matrices(tools: list[Tool], decisions: list[Decision], traces: list[Trace]) -> dict[str, np.ndarray]:
    """Return exposure-aware descriptive matrices.

    `substitutability` is intentionally absent.  It requires paired
    counterfactual menu replacements and cannot be recovered from co-exposure.
    """
    n = len(tools)
    index = {tool.tool_id: position for position, tool in enumerate(tools)}
    exposure = np.zeros((n, n), dtype=np.int64)
    confusion_count = np.zeros((n, n), dtype=np.int64)
    trace_cooccurrence = np.zeros((n, n), dtype=np.int64)
    trace_order = np.zeros((n, n), dtype=np.int64)
    for decision in decisions:
        if not decision.gold_tool_id or decision.gold_tool_id not in index:
            continue
        gold = index[decision.gold_tool_id]
        for candidate_id in decision.candidate_tool_ids:
            if candidate_id != decision.gold_tool_id and candidate_id in index:
                exposure[gold, index[candidate_id]] += 1
        if decision.chosen_tool_id and decision.chosen_tool_id != decision.gold_tool_id:
            confusion_count[gold, index[decision.chosen_tool_id]] += 1
    for trace in traces:
        values = [index[item] for item in trace.tool_ids if item in index]
        for left in values:
            for right in values:
                if left != right:
                    trace_cooccurrence[left, right] += 1
        for left, right in zip(values, values[1:]):
            trace_order[left, right] += 1
    confusion_rate = np.divide(
        confusion_count, exposure, out=np.full((n, n), np.nan), where=exposure > 0,
    )
    symmetric = np.full((n, n), np.nan)
    asymmetric = np.full((n, n), np.nan)
    for left in range(n):
        for right in range(left + 1, n):
            if exposure[left, right] and exposure[right, left]:
                symmetric[left, right] = symmetric[right, left] = (confusion_rate[left, right] + confusion_rate[right, left]) / 2
                asymmetric[left, right] = (confusion_rate[left, right] - confusion_rate[right, left]) / 2
                asymmetric[right, left] = -asymmetric[left, right]
    return {
        "risk_set_exposure": exposure,
        "confusion_count": confusion_count,
        "confusion_rate": confusion_rate,
        "confusion_symmetric": symmetric,
        "confusion_asymmetric": asymmetric,
        "trace_cooccurrence": trace_cooccurrence,
        "trace_order": trace_order,
    }
