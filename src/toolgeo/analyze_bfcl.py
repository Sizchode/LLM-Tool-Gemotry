"""Analyze BFCL cosine readout, separating replication from extensions."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cosine_argmax(
    state: torch.Tensor,
    prototypes: dict[str, torch.Tensor],
    candidates: list[str],
) -> str:
    missing = [name for name in candidates if name not in prototypes]
    if missing:
        raise ValueError(f"missing prototypes: {missing}")
    state = state.float()
    state_norm = torch.linalg.vector_norm(state)
    if state_norm == 0:
        raise ValueError("zero-norm decision state")
    scores = []
    for name in candidates:
        prototype = prototypes[name].float()
        norm = torch.linalg.vector_norm(prototype)
        if norm == 0:
            raise ValueError(f"zero-norm prototype: {name}")
        scores.append(torch.dot(state, prototype) / (state_norm * norm))
    return candidates[int(torch.stack(scores).argmax())]


def leave_out_prototypes(
    states: torch.Tensor,
    labels: list[str],
    excluded_indices: set[int],
) -> dict[str, torch.Tensor]:
    grouped: defaultdict[str, list[torch.Tensor]] = defaultdict(list)
    for index, label in enumerate(labels):
        if index not in excluded_indices:
            grouped[label].append(states[index].float())
    return {
        label: torch.stack(vectors).mean(dim=0)
        for label, vectors in grouped.items()
        if vectors
    }


def _state_rows(
    output_dir: Path,
) -> tuple[list[dict[str, Any]], torch.Tensor, list[int]]:
    records = _read_jsonl(output_dir / "examples.jsonl")
    payload = torch.load(output_dir / "states.pt", map_location="cpu", weights_only=True)
    keys_from_records = [(row["example_id"], row["order"]) for row in records]
    keys_from_states = list(zip(payload["example_ids"], payload["orders"]))
    if keys_from_records != keys_from_states:
        raise ValueError("examples.jsonl and states.pt row orders differ")
    return records, payload["states"], list(payload["decoder_layers"])


def _original_rows(
    records: list[dict[str, Any]], states: torch.Tensor
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    indices = [index for index, row in enumerate(records) if row["order"] == "original"]
    return [records[index] for index in indices], states[indices]


def _generation_map(output_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = _read_jsonl(output_dir / "generations.jsonl")
    result = {(row["example_id"], row["order"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("generation keys are not unique")
    return result


def _global_predictions(
    layer_states: torch.Tensor, labels: list[str]
) -> tuple[list[str | None], list[bool]]:
    tools = sorted(set(labels))
    tool_index = {tool: index for index, tool in enumerate(tools)}
    counts = Counter(labels)
    sums = torch.zeros((len(tools), layer_states.shape[1]), dtype=torch.float32)
    for index, label in enumerate(labels):
        sums[tool_index[label]] += layer_states[index].float()
    prototypes = sums / torch.tensor(
        [counts[tool] for tool in tools], dtype=torch.float32
    ).unsqueeze(1)
    state_norm = torch.nn.functional.normalize(layer_states.float(), dim=1)
    prototype_norm = torch.nn.functional.normalize(prototypes, dim=1)
    scores = state_norm @ prototype_norm.T

    eligible = [counts[label] > 1 for label in labels]
    predictions: list[str | None] = [None] * len(labels)
    for index, label in enumerate(labels):
        if not eligible[index]:
            continue
        position = tool_index[label]
        loo = (sums[position] - layer_states[index].float()) / (counts[label] - 1)
        scores[index, position] = torch.nn.functional.cosine_similarity(
            layer_states[index].float().unsqueeze(0), loo.unsqueeze(0)
        )[0]
        predictions[index] = tools[int(scores[index].argmax())]
    return predictions, eligible


def _menu_predictions(
    layer_states: torch.Tensor, rows: list[dict[str, Any]]
) -> tuple[list[str | None], list[bool]]:
    labels = [str(row["gold_tool"]) for row in rows]
    tools = sorted(set(labels))
    tool_index = {tool: index for index, tool in enumerate(tools)}
    counts = Counter(labels)
    sums = torch.zeros((len(tools), layer_states.shape[1]), dtype=torch.float32)
    indices_by_query: defaultdict[str, list[int]] = defaultdict(list)
    for index, (label, row) in enumerate(zip(labels, rows)):
        sums[tool_index[label]] += layer_states[index].float()
        indices_by_query[str(row["exact_query_key"])].append(index)

    predictions: list[str | None] = []
    eligible: list[bool] = []
    for index, row in enumerate(rows):
        excluded = indices_by_query[str(row["exact_query_key"])]
        removed_counts = Counter(labels[item] for item in excluded)
        candidates = [str(name) for name in row["candidate_tools"]]
        can_evaluate = all(
            name in counts and counts[name] - removed_counts[name] > 0
            for name in candidates
        )
        eligible.append(can_evaluate)
        if not can_evaluate:
            predictions.append(None)
            continue
        prototypes: dict[str, torch.Tensor] = {}
        for name in candidates:
            removed = torch.stack(
                [layer_states[item].float() for item in excluded if labels[item] == name]
            ).sum(dim=0) if removed_counts[name] else torch.zeros_like(sums[0])
            prototypes[name] = (
                sums[tool_index[name]] - removed
            ) / (counts[name] - removed_counts[name])
        predictions.append(
            cosine_argmax(layer_states[index], prototypes, candidates)
        )
    return predictions, eligible


def analyze(config: dict[str, Any]) -> list[Path]:
    output_dir = Path(config["run"]["output_dir"])
    records, states, decoder_layers = _state_rows(output_dir)
    rows, original_states = _original_rows(records, states)
    generations = _generation_map(output_dir)
    labels = [str(row["gold_tool"]) for row in rows]
    total = len(rows)

    replication_layer = int(config["analysis"]["replication_decoder_layer"])
    replication_offset = decoder_layers.index(replication_layer)
    replication_predictions, replication_eligible = _global_predictions(
        original_states[:, replication_offset], labels
    )
    replication_indices = [
        index for index, keep in enumerate(replication_eligible) if keep
    ]
    replication_row = {
        "decoder_layer": replication_layer,
        "candidate_scope": "global",
        "evaluated_examples": len(replication_indices),
        "total_examples": total,
        "readout_gold_accuracy": sum(
            replication_predictions[index] == labels[index]
            for index in replication_indices
        ) / len(replication_indices),
        "generation_gold_accuracy": sum(
            generations[(rows[index]["example_id"], "original")]["outcome"]
            == "correct"
            for index in replication_indices
        ) / len(replication_indices),
    }
    replication_path = output_dir / "bfcl_replication_results.csv"
    _write_csv(replication_path, [replication_row])

    result_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    for offset, layer in enumerate(decoder_layers):
        predictions, eligible = _menu_predictions(original_states[:, offset], rows)
        evaluated = [index for index, keep in enumerate(eligible) if keep]
        generation_correct = [
            index
            for index in evaluated
            if generations[(rows[index]["example_id"], "original")]["outcome"]
            == "correct"
        ]
        generation_wrong = [
            index
            for index in evaluated
            if generations[(rows[index]["example_id"], "original")]["outcome"]
            == "wrong_in_menu"
        ]
        generation_out_of_menu = [
            index
            for index in evaluated
            if generations[(rows[index]["example_id"], "original")]["outcome"]
            == "out_of_menu"
        ]
        generation_invalid = [
            index
            for index in evaluated
            if generations[(rows[index]["example_id"], "original")]["outcome"]
            == "invalid"
        ]
        result_rows.append(
            {
                "decoder_layer": layer,
                "candidate_scope": "benchmark_menu",
                "evaluated_examples": len(evaluated),
                "total_examples": total,
                "coverage": len(evaluated) / total,
                "readout_gold_accuracy": sum(
                    predictions[index] == labels[index] for index in evaluated
                ) / len(evaluated),
                "generation_gold_accuracy": sum(
                    generations[(rows[index]["example_id"], "original")]["outcome"]
                    == "correct"
                    for index in evaluated
                ) / len(evaluated),
                "chance_accuracy": sum(
                    1 / len(rows[index]["candidate_tools"]) for index in evaluated
                ) / len(evaluated),
            }
        )
        wrong_gold = sum(predictions[index] == labels[index] for index in generation_wrong)
        wrong_generated = sum(
            predictions[index]
            == generations[(rows[index]["example_id"], "original")]["action"]
            for index in generation_wrong
        )
        error_rows.append(
            {
                "decoder_layer": layer,
                "generation_correct_examples": len(generation_correct),
                "readout_gold_given_generation_correct": (
                    sum(predictions[index] == labels[index] for index in generation_correct)
                    / len(generation_correct)
                    if generation_correct
                    else None
                ),
                "generation_wrong_examples": len(generation_wrong),
                "readout_gold_given_generation_wrong": (
                    wrong_gold / len(generation_wrong) if generation_wrong else None
                ),
                "readout_generated_wrong_given_generation_wrong": (
                    wrong_generated / len(generation_wrong) if generation_wrong else None
                ),
                "readout_other_given_generation_wrong": (
                    (len(generation_wrong) - wrong_gold - wrong_generated)
                    / len(generation_wrong)
                    if generation_wrong
                    else None
                ),
                "generation_out_of_menu_examples": len(generation_out_of_menu),
                "generation_invalid_examples": len(generation_invalid),
            }
        )

    results_path = output_dir / "bfcl_results.csv"
    errors_path = output_dir / "bfcl_error_conditioning.csv"
    _write_csv(results_path, result_rows)
    _write_csv(errors_path, error_rows)

    original_generations = [
        generations[(row["example_id"], "original")] for row in rows
    ]
    outcome_counts = Counter(row["outcome"] for row in original_generations)
    generation_path = output_dir / "bfcl_generation_results.csv"
    _write_csv(
        generation_path,
        [
            {
                "examples": total,
                "accuracy": outcome_counts["correct"] / total,
                "correct": outcome_counts["correct"],
                "wrong_in_menu": outcome_counts["wrong_in_menu"],
                "out_of_menu": outcome_counts["out_of_menu"],
                "invalid": outcome_counts["invalid"],
            }
        ],
    )
    return [replication_path, results_path, errors_path, generation_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for path in analyze(config):
        print(path)


if __name__ == "__main__":
    main()
