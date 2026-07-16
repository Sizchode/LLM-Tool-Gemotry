from __future__ import annotations

import numpy as np

def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2
        start = end
    return result
def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left, right = _rank(left), _rank(right)
    left -= left.mean(); right -= right.mean()
    return float((left @ right) / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-12))

def evaluate(feature_cosines: dict[str, np.ndarray], behavior: dict[str, np.ndarray], heldout_fraction: float, seed: int) -> dict:
    n = next(iter(feature_cosines.values())).shape[0]
    pairs = np.array([(i, j) for i in range(n) for j in range(n) if i != j])
    rng = np.random.default_rng(seed); test = rng.random(len(pairs)) < heldout_fraction
    report: dict[str, dict[str, float]] = {}
    for target, matrix in behavior.items():
        values = matrix[pairs[:, 0], pairs[:, 1]]
        if not np.any(values):
            report[target] = {"status": "unavailable_no_observations"}
            continue
        report[target] = {name: spearman(cosine[pairs[test, 0], pairs[test, 1]], values[test]) for name, cosine in feature_cosines.items()}
        report[target]["n_heldout_pairs"] = int(test.sum())
        report[target]["n_heldout_nonzero"] = int((values[test] != 0).sum())
        report[target]["prevalence"] = float((values[test] != 0).mean())
        report[target]["winner"] = max(feature_cosines, key=lambda name: report[target][name])
    return report
