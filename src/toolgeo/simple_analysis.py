"""Direct, layerwise Paper-1 measurements without learned prediction models."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .io import read_jsonl, write_json, write_jsonl


def _normalise(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)


def _schema_features(schema: Any, prefix: str = "$") -> set[str]:
    result: set[str] = set()
    if isinstance(schema, dict):
        if "type" in schema:
            result.add(f"{prefix}:type={schema['type']}")
        required = set(schema.get("required", []))
        for key, value in sorted(schema.get("properties", {}).items()):
            child = f"{prefix}.{key}"
            result.add(f"{child}:required={key in required}")
            result |= _schema_features(value, child)
        if "items" in schema:
            result |= _schema_features(schema["items"], f"{prefix}[]")
    return result


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _mean_schema_jaccard(schema_sets: list[set[str]]) -> float | None:
    total = 0.0
    pairs = 0
    for left in range(len(schema_sets)):
        for right in range(left + 1, len(schema_sets)):
            total += _jaccard(schema_sets[left], schema_sets[right])
            pairs += 1
    return total / pairs if pairs else None


def _error_locality(rows: list[dict[str, Any]], similarities: list[list[float]]) -> dict[str, Any]:
    selected, gaps, percentiles = [], [], []
    hit = {1: 0, 5: 0, 10: 0}
    errors = 0
    for row, values in zip(rows, similarities):
        if row["chosen_tool_id"] == row["gold_tool_id"]:
            continue
        candidates = row["candidate_tool_ids"]
        gold = candidates.index(row["gold_tool_id"])
        chosen = candidates.index(row["chosen_tool_id"])
        distractors = [index for index in range(len(candidates)) if index != gold]
        others = [index for index in distractors if index != chosen]
        chosen_value = float(values[chosen])
        selected.append(chosen_value)
        if others:
            gaps.append(chosen_value - float(np.mean([values[index] for index in others])))
        ranked = sorted(distractors, key=lambda index: values[index], reverse=True)
        rank = ranked.index(chosen) + 1
        percentiles.append(1.0 - (rank - 1) / max(len(ranked) - 1, 1))
        for k in hit:
            hit[k] += int(rank <= min(k, len(ranked)))
        errors += 1
    return {
        "n_errors": errors,
        "n_errors_with_additional_distractors": len(gaps),
        "selected_wrong_cosine": float(np.mean(selected)) if selected else None,
        "selected_minus_other_distractors": float(np.mean(gaps)) if gaps else None,
        "selected_neighbor_percentile": float(np.mean(percentiles)) if percentiles else None,
        "hit_at_k": {str(k): (hit[k] / errors if errors else None) for k in hit},
    }


def _confusion_spearman(rows: list[dict[str, Any]], similarities: list[list[float]]) -> dict[str, Any]:
    from scipy.stats import spearmanr

    exposure: dict[tuple[str, str], int] = defaultdict(int)
    chosen: dict[tuple[str, str], int] = defaultdict(int)
    similarity_sum: dict[tuple[str, str], float] = defaultdict(float)
    for row, values in zip(rows, similarities):
        gold = row["gold_tool_id"]
        for candidate, value in zip(row["candidate_tool_ids"], values):
            if candidate == gold:
                continue
            pair = (gold, candidate)
            exposure[pair] += 1
            similarity_sum[pair] += float(value)
            chosen[pair] += int(row["chosen_tool_id"] == candidate)
    pairs = sorted(exposure)
    rates = np.array([chosen[pair] / exposure[pair] for pair in pairs])
    geometry = np.array([similarity_sum[pair] / exposure[pair] for pair in pairs])
    correlation, p_value = spearmanr(geometry, rates) if len(pairs) > 1 else (np.nan, np.nan)
    return {
        "n_exposed_directed_pairs": len(pairs),
        "spearman_r": float(correlation),
        "p_value": float(p_value),
    }


def _tool_centroids(values: np.ndarray, tool_indices: np.ndarray, n_tools: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sums = np.zeros((n_tools, values.shape[1]), dtype=np.float64)
    counts = np.bincount(tool_indices, minlength=n_tools)
    np.add.at(sums, tool_indices, values)
    centroids = _normalise(sums.astype(np.float32))
    return sums, counts, centroids


def _retrieval(values: np.ndarray, tool_indices: np.ndarray, sums: np.ndarray, counts: np.ndarray, centroids: np.ndarray) -> float | None:
    eligible = np.flatnonzero(counts[tool_indices] >= 2)
    if not len(eligible):
        return None
    correct = 0
    for start in range(0, len(eligible), 256):
        rows = eligible[start : start + 256]
        scores = values[rows] @ centroids.T
        own = tool_indices[rows]
        leave_one_out = sums[own] - values[rows]
        leave_one_out /= np.clip(np.linalg.norm(leave_one_out, axis=1, keepdims=True), 1e-12, None)
        scores[np.arange(len(rows)), own] = np.sum(values[rows] * leave_one_out, axis=1)
        correct += int(np.sum(np.argmax(scores, axis=1) == own))
    return correct / len(eligible)


def _within_between(values: np.ndarray, tool_indices: np.ndarray, n_tools: int) -> tuple[float | None, float | None]:
    sums = np.zeros((n_tools, values.shape[1]), dtype=np.float64)
    counts = np.bincount(tool_indices, minlength=n_tools)
    np.add.at(sums, tool_indices, values)
    squared_norms = np.sum(values.astype(np.float64) ** 2, axis=1)
    per_tool_squared_norms = np.zeros(n_tools, dtype=np.float64)
    np.add.at(per_tool_squared_norms, tool_indices, squared_norms)
    within_sum = float(np.sum((sums * sums).sum(1) - per_tool_squared_norms) / 2)
    within_pairs = int(np.sum(counts * (counts - 1) // 2))
    total_vector = values.astype(np.float64).sum(0)
    total_sum = float(((total_vector @ total_vector) - squared_norms.sum()) / 2)
    total_pairs = len(values) * (len(values) - 1) // 2
    between_pairs = total_pairs - within_pairs
    return (
        within_sum / within_pairs if within_pairs else None,
        (total_sum - within_sum) / between_pairs if between_pairs else None,
    )


def analyze(measurement_dir: str, semantic_path: str, output: str) -> dict[str, Any]:
    root = Path(measurement_dir)
    tools = read_jsonl(root / "tools.jsonl")
    contexts = read_jsonl(root / "context_index.jsonl")
    measurements = read_jsonl(root / "geometry_measurements.jsonl")
    index = np.load(root / "geometry_index.npz")
    layers = [int(value) for value in index["layers"].tolist()]
    shards = [str(value) for value in index["representation_shards"].tolist()]
    all_tool_ids = [str(row["tool_id"]) for row in tools]
    all_tool_lookup = {tool_id: position for position, tool_id in enumerate(all_tool_ids)}
    observed = {str(row["tool_id"]) for row in contexts}
    tool_ids = [tool_id for tool_id in all_tool_ids if tool_id in observed]
    tool_lookup = {tool_id: position for position, tool_id in enumerate(tool_ids)}
    context_tool_indices = np.array([tool_lookup[str(row["tool_id"])] for row in contexts], dtype=np.int32)
    original_context = np.array([row["context_variant"] == "original" for row in contexts], dtype=bool)
    active_tools = [tools[all_tool_lookup[tool_id]] for tool_id in tool_ids]
    schema_sets = [_schema_features(row.get("schema", {})) for row in active_tools]
    global_schema_jaccard = _mean_schema_jaccard(schema_sets)

    if len(layers) != len(shards):
        raise ValueError("Layer and representation-shard counts differ")
    if any([int(value) for value in row["layers"]] != layers for row in measurements):
        raise ValueError("Measurement rows do not share the geometry-index layer sequence")

    layer_reports = []
    neighbor_rows = []
    previous_neighbors: np.ndarray | None = None
    for layer_offset, (layer, shard) in enumerate(zip(layers, shards)):
        values = _normalise(np.load(root / shard, mmap_mode="r"))
        original_values = values[original_context]
        original_indices = context_tool_indices[original_context]
        sums, counts, centroids = _tool_centroids(original_values, original_indices, len(tools))
        natural_within, natural_between = _within_between(original_values, original_indices, len(tools))
        all_sums, all_counts, all_centroids = _tool_centroids(values, context_tool_indices, len(tools))
        all_within, all_between = _within_between(values, context_tool_indices, len(tools))
        ks = [value for value in (1, 5, 10) if value < len(tools)]
        max_k = max(ks)
        cosine = centroids @ centroids.T
        np.fill_diagonal(cosine, -np.inf)
        neighbors = np.argsort(-cosine, axis=1)[:, :max_k]
        schema_neighbor = {
            str(k): float(np.mean([
                _jaccard(schema_sets[index_], schema_sets[neighbor])
                for index_ in range(len(tools)) for neighbor in neighbors[index_, :k]
            ]))
            for k in ks
        }
        consistency = None
        if previous_neighbors is not None:
            consistency = {
                str(k): float(np.mean([
                    len(set(previous_neighbors[index_, :k]) & set(neighbors[index_, :k])) / k
                    for index_ in range(len(tools))
                ]))
                for k in ks
            }
        similarities = [row["gold_candidate_cosine_by_layer"][layer_offset] for row in measurements]
        report = {
            "layer": layer,
            "stability": {
                "natural_query_and_menu_contexts": {
                    "same_tool_retrieval_accuracy": _retrieval(
                        original_values, original_indices, sums, counts, centroids,
                    ),
                    "within_tool_cosine": natural_within,
                    "between_tool_cosine": natural_between,
                    "within_minus_between_cosine": (
                        natural_within - natural_between
                        if natural_within is not None and natural_between is not None else None
                    ),
                },
                "including_reverse_position_intervention": {
                    "same_tool_retrieval_accuracy": _retrieval(
                        values, context_tool_indices, all_sums, all_counts, all_centroids,
                    ),
                    "within_tool_cosine": all_within,
                    "between_tool_cosine": all_between,
                    "within_minus_between_cosine": (
                        all_within - all_between
                        if all_within is not None and all_between is not None else None
                    ),
                },
            },
            "schema_jaccard_at_k": schema_neighbor,
            "schema_neighbor_enrichment_over_all_pairs": {
                str(k): (schema_neighbor[str(k)] - global_schema_jaccard)
                if global_schema_jaccard is not None else None
                for k in ks
            },
            "neighbor_overlap_with_previous_layer": consistency,
            "error_locality": _error_locality(measurements, similarities),
            "confusion_geometry": _confusion_spearman(measurements, similarities),
        }
        layer_reports.append(report)
        for tool_index, neighbor_indices in enumerate(neighbors):
            neighbor_rows.append({
                "layer": layer,
                "tool_id": tool_ids[tool_index],
                "neighbors": [
                    {"tool_id": tool_ids[value], "cosine": float(cosine[tool_index, value])}
                    for value in neighbor_indices
                ],
            })
        previous_neighbors = neighbors

    semantic = np.load(semantic_path)
    semantic_ids = [str(value) for value in semantic["tool_ids"].tolist()]
    if semantic_ids != all_tool_ids:
        raise ValueError("Semantic baseline tool IDs do not match measurement tools")
    semantic_vectors = _normalise(semantic["card_embedding"])[
        [all_tool_lookup[tool_id] for tool_id in tool_ids]
    ]
    semantic_similarities = []
    schema_similarities = []
    for row in measurements:
        candidates = row["candidate_tool_ids"]
        indices = [tool_lookup[value] for value in candidates]
        gold = tool_lookup[row["gold_tool_id"]]
        semantic_similarities.append((semantic_vectors[indices] @ semantic_vectors[gold]).tolist())
        schema_similarities.append([_jaccard(schema_sets[gold], schema_sets[index_]) for index_ in indices])
    result = {
        "analysis": "direct_contextual_geometry_measurements",
        "pooling": str(index["pooling"]),
        "n_tools_in_dataset": len(tools),
        "n_tools_in_eligible_menus": len(tool_ids),
        "n_contextual_tool_occurrences": len(contexts),
        "n_decisions": len(measurements),
        "all_tool_pairs_schema_jaccard": global_schema_jaccard,
        "layerwise": layer_reports,
        "external_semantic_baseline": {
            "error_locality": _error_locality(measurements, semantic_similarities),
            "confusion_geometry": _confusion_spearman(measurements, semantic_similarities),
        },
        "schema_jaccard_baseline": {
            "error_locality": _error_locality(measurements, schema_similarities),
            "confusion_geometry": _confusion_spearman(measurements, schema_similarities),
        },
        "neighbor_graph": str(Path(output).with_name("tool_neighbors.jsonl")),
    }
    write_json(Path(output), result)
    write_jsonl(Path(output).with_name("tool_neighbors.jsonl"), neighbor_rows)
    return result
