"""Risk-set-aware discrete-choice analysis for Paper 1."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from .schema import Decision, Tool


@dataclass
class ChoiceSet:
    decision_id: str
    query_group: str
    gold_index: int
    chosen: int
    candidate_indices: np.ndarray
    positions: np.ndarray


def _bucket(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}|{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def tool_partitions(tools: list[Tool], validation_fraction: float, test_fraction: float, seed: int) -> dict[str, set[int]]:
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction and test_fraction must be positive and sum to less than one")
    result = {"train": set(), "validation": set(), "test": set()}
    for index, tool in enumerate(tools):
        value = _bucket(tool.tool_id, seed)
        split = "test" if value < test_fraction else "validation" if value < test_fraction + validation_fraction else "train"
        result[split].add(index)
    if any(not values for values in result.values()):
        raise ValueError("Tool-disjoint split produced an empty partition; use more tools or a different seed")
    return result


def choice_sets(tools: list[Tool], decisions: list[Decision]) -> list[ChoiceSet]:
    index = {tool.tool_id: position for position, tool in enumerate(tools)}
    result = []
    for decision in decisions:
        # Seal-Tools may contain multi-call answers.  Paper 1's estimand is a
        # single categorical choice, so first-call projection would be invalid.
        if decision.gold_call_count != 1:
            continue
        if not decision.gold_tool_id or not decision.chosen_tool_id:
            continue
        if decision.gold_tool_id not in index or decision.chosen_tool_id not in decision.candidate_tool_ids:
            continue
        candidates = np.array([index[item] for item in decision.candidate_tool_ids], dtype=np.int32)
        result.append(ChoiceSet(
            decision.decision_id, decision.decision_id.split("::", 1)[0], index[decision.gold_tool_id],
            decision.candidate_tool_ids.index(decision.chosen_tool_id), candidates,
            np.arange(len(candidates), dtype=np.float64) / max(len(candidates) - 1, 1),
        ))
    return result


def split_choice_sets(sets: list[ChoiceSet], partitions: dict[str, set[int]]) -> dict[str, list[ChoiceSet]]:
    train_tools, validation_tools, test_tools = (partitions[name] for name in ("train", "validation", "test"))
    # No held-out tool appears anywhere in an earlier split's menu.  Validation
    # and test are keyed by held-out gold identity, so entire directed rows are
    # unseen during fitting/selection.
    return {
        "train": [item for item in sets if set(item.candidate_indices) <= train_tools],
        "validation": [item for item in sets if item.gold_index in validation_tools and not (set(item.candidate_indices) & test_tools)],
        "test": [item for item in sets if item.gold_index in test_tools],
        "refit": [item for item in sets if not (set(item.candidate_indices) & test_tools)],
    }


def _design(item: ChoiceSet, matrices: dict[str, np.ndarray], directional: dict[str, np.ndarray]) -> np.ndarray:
    n = len(item.candidate_indices)
    is_gold = (item.candidate_indices == item.gold_index).astype(np.float64)
    columns = [is_gold, item.positions]
    for matrix in matrices.values():
        values = matrix[item.gold_index, item.candidate_indices].astype(np.float64)
        # Self-similarity is always one and would trivially identify the gold.
        # is_gold absorbs correctness; geometry only ranks distractors.
        columns.append(np.where(is_gold > 0, 0.0, values))
    for values in directional.values():
        columns.append(values[item.candidate_indices] - values[item.gold_index])
    return np.column_stack(columns).reshape(n, -1)


def _prepare(sets: list[ChoiceSet], matrices: dict[str, np.ndarray], directional: dict[str, np.ndarray]) -> list[np.ndarray]:
    return [_design(item, matrices, directional) for item in sets]


def _fit(designs: list[np.ndarray], sets: list[ChoiceSet], l2: float) -> dict[str, Any]:
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise RuntimeError("Install toolgeo[analysis] for conditional-logit analysis.") from exc
    if not sets:
        raise ValueError("No decisions are available for this split")
    stacked = np.vstack(designs)
    mean = stacked.mean(0)
    scale = stacked.std(0)
    # Binary is_gold and position are interpretable without centering, but all
    # columns receive a fixed train-derived scale for numerical conditioning.
    scale[scale < 1e-8] = 1.0
    normalized = [(value - mean) / scale for value in designs]

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        loss = 0.5 * l2 * float(beta @ beta)
        gradient = l2 * beta
        for design, item in zip(normalized, sets):
            utility = design @ beta
            utility -= utility.max()
            probability = np.exp(utility)
            probability /= probability.sum()
            loss -= np.log(max(probability[item.chosen], 1e-15))
            gradient += design.T @ probability - design[item.chosen]
        return loss, gradient

    result = minimize(lambda beta: objective(beta), np.zeros(stacked.shape[1]), jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"Conditional logit failed: {result.message}")
    return {"beta": result.x, "mean": mean, "scale": scale, "optimization": str(result.message)}


def _score(model: dict[str, Any], designs: list[np.ndarray], sets: list[ChoiceSet]) -> dict[str, Any]:
    losses, correct, probabilities = [], [], []
    for design, item in zip(designs, sets):
        normalized = (design - model["mean"]) / model["scale"]
        utility = normalized @ model["beta"]
        utility -= utility.max()
        probability = np.exp(utility); probability /= probability.sum()
        losses.append(-np.log(max(probability[item.chosen], 1e-15)))
        correct.append(int(np.argmax(probability) == item.chosen))
        probabilities.append(probability)
    return {
        "mean_negative_log_likelihood": float(np.mean(losses)),
        "choice_accuracy": float(np.mean(correct)),
        "per_decision_nll": np.array(losses),
        "probabilities": probabilities,
    }


def _fit_score(train: list[ChoiceSet], test: list[ChoiceSet], matrices: dict[str, np.ndarray], directional: dict[str, np.ndarray], l2: float) -> tuple[dict[str, Any], dict[str, Any]]:
    train_design = _prepare(train, matrices, directional)
    model = _fit(train_design, train, l2)
    score = _score(model, _prepare(test, matrices, directional), test)
    return model, score


def _similarity(value: np.ndarray, n_tools: int) -> np.ndarray:
    if value.ndim != 2 or value.shape[0] != n_tools:
        raise ValueError("Geometry view must be [tool, hidden]")
    normalized = value.astype(np.float32)
    normalized /= np.clip(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-12, None)
    return normalized @ normalized.T


def _bootstrap_delta(groups: list[str], full_nll: np.ndarray, baseline_nll: np.ndarray, samples: int, seed: int) -> dict[str, Any]:
    unique = sorted(set(groups))
    members = {group: np.flatnonzero(np.array(groups) == group) for group in unique}
    rng = np.random.default_rng(seed)
    observed = float(np.mean(baseline_nll - full_nll))
    draws = []
    for _ in range(samples):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([members[group] for group in chosen])
        draws.append(float(np.mean(baseline_nll[indices] - full_nll[indices])))
    low, high = np.quantile(draws, [0.025, 0.975]) if draws else (np.nan, np.nan)
    return {"delta_nll": observed, "ci95": [float(low), float(high)], "samples": samples, "cluster": "query"}


def evaluate_choice_geometry(
    tools: list[Tool], decisions: list[Decision], geometry: dict[str, np.ndarray],
    baselines: dict[str, np.ndarray], stability: dict[str, float],
    directional: dict[str, np.ndarray] | None = None, validation_fraction: float = 0.2,
    test_fraction: float = 0.2, stability_threshold: float = 0.8,
    bootstrap_samples: int = 500, permutation_samples: int = 200,
    l2: float = 1e-4, seed: int = 17,
) -> dict[str, Any]:
    directional = directional or {}
    sets = choice_sets(tools, decisions)
    partitions = tool_partitions(tools, validation_fraction, test_fraction, seed)
    split = split_choice_sets(sets, partitions)
    if min(len(split[name]) for name in ("train", "validation", "test")) < 2:
        raise ValueError({name: len(value) for name, value in split.items()})
    _, validation_surface = _fit_score(split["train"], split["validation"], baselines, directional, l2)
    candidates: dict[str, dict[str, float]] = {}
    eligible = [name for name in geometry if stability.get(name, 0.0) >= stability_threshold]
    if not eligible:
        raise ValueError(f"No geometry view passed preregistered stability threshold {stability_threshold}")
    for name in eligible:
        pairwise = _similarity(geometry[name], len(tools))
        _, score = _fit_score(split["train"], split["validation"], {**baselines, name: pairwise}, directional, l2)
        candidates[name] = {
            "validation_nll": score["mean_negative_log_likelihood"],
            "delta_nll_over_surface": validation_surface["mean_negative_log_likelihood"] - score["mean_negative_log_likelihood"],
            "card_stability": stability[name],
        }
    selected = min(candidates, key=lambda name: candidates[name]["validation_nll"])
    surface_model, surface_test = _fit_score(split["refit"], split["test"], baselines, directional, l2)
    full_matrices = {**baselines, selected: _similarity(geometry[selected], len(tools))}
    full_model, full_test = _fit_score(split["refit"], split["test"], full_matrices, directional, l2)
    bootstrap = _bootstrap_delta(
        [item.query_group for item in split["test"]], full_test["per_decision_nll"],
        surface_test["per_decision_nll"], bootstrap_samples, seed,
    )
    # Menu-local randomization keeps the risk set, choice, gold indicator,
    # surface covariates, and order fixed.  Only the selected geometric values
    # are reassigned among distractors.  This directly targets the incremental
    # test statistic under the null of no geometry-to-choice correspondence.
    rng = np.random.default_rng(seed + 1)
    observed = bootstrap["delta_nll"]
    null = []
    geometry_column = 2 + len(baselines)
    full_designs = _prepare(split["test"], full_matrices, directional)
    for _ in range(permutation_samples):
        permuted = []
        for design, item in zip(full_designs, split["test"]):
            value = design.copy()
            distractors = np.flatnonzero(item.candidate_indices != item.gold_index)
            value[distractors, geometry_column] = rng.permutation(value[distractors, geometry_column])
            permuted.append(value)
        permuted_score = _score(full_model, permuted, split["test"])
        null.append(float(surface_test["mean_negative_log_likelihood"] - permuted_score["mean_negative_log_likelihood"]))
    p_value = (1 + sum(value >= observed for value in null)) / (1 + len(null))
    names = ["is_gold", "menu_position", *baselines.keys(), selected, *directional.keys()]
    return {
        "analysis": "conditional_logit_with_query_risk_sets",
        "identification": "standalone-card geometry predicts in-context menu choices",
        "split_unit": "tool_id", "selection_protocol": "validation selects layer/pooling; test evaluated once",
        "tool_counts": {name: len(values) for name, values in partitions.items()},
        "decision_counts": {name: len(values) for name, values in split.items()},
        "stability_threshold": stability_threshold, "selected_geometry_view": selected,
        "validation_candidates": candidates,
        "surface_test": {key: value for key, value in surface_test.items() if key not in ("per_decision_nll", "probabilities")},
        "surface_plus_geometry_test": {key: value for key, value in full_test.items() if key not in ("per_decision_nll", "probabilities")},
        "query_bootstrap": bootstrap,
        "within_menu_permutation": {"samples": permutation_samples, "p_value_one_sided": float(p_value), "null_mean_delta_nll": float(np.mean(null)) if null else None},
        "coefficients": {name: float(value) for name, value in zip(names, full_model["beta"])},
        "directionality_note": "Symmetric similarities explain the symmetric component; position and candidate-minus-gold priors model asymmetric residual structure.",
    }


def evaluate_cross_dataset_transfer(
    source_tools: list[Tool], source_decisions: list[Decision], source_geometry: dict[str, np.ndarray],
    source_baselines: dict[str, np.ndarray], source_stability: dict[str, float], source_directional: dict[str, np.ndarray],
    target_tools: list[Tool], target_decisions: list[Decision], target_geometry: dict[str, np.ndarray],
    target_baselines: dict[str, np.ndarray], target_directional: dict[str, np.ndarray],
    validation_fraction: float = 0.2, stability_threshold: float = 0.8,
    bootstrap_samples: int = 500, l2: float = 1e-4, seed: int = 17,
) -> dict[str, Any]:
    """Fit coefficients on one dataset and evaluate untouched target queries."""
    if list(source_baselines) != list(target_baselines):
        raise ValueError("Source and target baseline feature families/order must match")
    if list(source_directional) != list(target_directional):
        raise ValueError("Source and target directional feature families/order must match")
    shared = [name for name in source_geometry if name in target_geometry and source_stability.get(name, 0) >= stability_threshold]
    if not shared:
        raise ValueError("No stable layer/pooling view is shared across source and target")
    source_sets = choice_sets(source_tools, source_decisions)
    # A source-only tool-disjoint validation split selects the view. The target
    # dataset is never consulted until the final transfer score.
    partitions = tool_partitions(source_tools, validation_fraction, validation_fraction, seed)
    split = split_choice_sets(source_sets, partitions)
    if min(len(split["train"]), len(split["validation"])) < 2:
        raise ValueError("Insufficient source choices for transfer view selection")
    _, surface_validation = _fit_score(split["train"], split["validation"], source_baselines, source_directional, l2)
    candidates = {}
    for name in shared:
        source_pairwise = _similarity(source_geometry[name], len(source_tools))
        _, score = _fit_score(
            split["train"], split["validation"], {**source_baselines, name: source_pairwise}, source_directional, l2,
        )
        candidates[name] = surface_validation["mean_negative_log_likelihood"] - score["mean_negative_log_likelihood"]
    selected = max(candidates, key=candidates.get)
    target_sets = choice_sets(target_tools, target_decisions)
    if len(target_sets) < 2:
        raise ValueError("Insufficient target choices for cross-dataset transfer")
    surface_model = _fit(_prepare(source_sets, source_baselines, source_directional), source_sets, l2)
    full_source = {**source_baselines, selected: _similarity(source_geometry[selected], len(source_tools))}
    full_target = {**target_baselines, selected: _similarity(target_geometry[selected], len(target_tools))}
    full_model = _fit(_prepare(source_sets, full_source, source_directional), source_sets, l2)
    surface_score = _score(surface_model, _prepare(target_sets, target_baselines, target_directional), target_sets)
    full_score = _score(full_model, _prepare(target_sets, full_target, target_directional), target_sets)
    bootstrap = _bootstrap_delta(
        [item.query_group for item in target_sets], full_score["per_decision_nll"],
        surface_score["per_decision_nll"], bootstrap_samples, seed,
    )
    return {
        "analysis": "cross_dataset_conditional_logit_transfer",
        "selection_data": "source_validation_only", "target_data_used_for_selection": False,
        "selected_geometry_view": selected, "source_validation_delta_nll": candidates,
        "n_source_choices": len(source_sets), "n_target_choices": len(target_sets),
        "target_surface_nll": surface_score["mean_negative_log_likelihood"],
        "target_surface_plus_geometry_nll": full_score["mean_negative_log_likelihood"],
        "target_query_bootstrap": bootstrap,
    }
