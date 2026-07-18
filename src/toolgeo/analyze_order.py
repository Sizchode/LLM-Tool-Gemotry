"""Analyze original-versus-reverse BFCL menu-order matched pairs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import yaml


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(config: dict[str, Any]) -> list[Path]:
    output_dir = Path(config["run"]["output_dir"])
    records = _read_jsonl(output_dir / "examples.jsonl")
    payload = torch.load(output_dir / "states.pt", map_location="cpu", weights_only=True)
    generations = _read_jsonl(output_dir / "generations.jsonl")
    state_keys = list(zip(payload["example_ids"], payload["orders"]))
    record_keys = [(row["example_id"], row["order"]) for row in records]
    if state_keys != record_keys:
        raise ValueError("examples.jsonl and states.pt row orders differ")

    record_by_key = {key: row for key, row in zip(record_keys, records)}
    state_index_by_key = {key: i for i, key in enumerate(state_keys)}
    generation_by_key = {
        (row["example_id"], row["order"]): row for row in generations
    }
    example_ids = [
        row["example_id"] for row in records if row["order"] == "original"
    ]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("original-order example IDs are not unique")

    for example_id in example_ids:
        original = record_by_key[(example_id, "original")]
        reverse = record_by_key[(example_id, "reverse")]
        if original["messages"] != reverse["messages"]:
            raise ValueError(f"{example_id}: messages changed under menu reversal")
        if original["functions"] != list(reversed(reverse["functions"])):
            raise ValueError(f"{example_id}: reverse run changed more than tool order")

    original_indices = [
        state_index_by_key[(example_id, "original")] for example_id in example_ids
    ]
    reverse_indices = [
        state_index_by_key[(example_id, "reverse")] for example_id in example_ids
    ]
    layers = list(payload["decoder_layers"])
    similarities = torch.empty((len(example_ids), len(layers)), dtype=torch.float32)
    for layer_offset in range(len(layers)):
        similarities[:, layer_offset] = torch.nn.functional.cosine_similarity(
            payload["states"][original_indices, layer_offset].float(),
            payload["states"][reverse_indices, layer_offset].float(),
            dim=1,
        )
    representation_rows: list[dict[str, Any]] = []
    for example_offset, example_id in enumerate(example_ids):
        original = record_by_key[(example_id, "original")]
        for offset, layer in enumerate(layers):
            representation_rows.append(
                {
                    "example_id": example_id,
                    "gold_tool": str(original["gold_tool"]),
                    "decoder_layer": layer,
                    "original_reverse_cosine": float(
                        similarities[example_offset, offset]
                    ),
                }
            )

    pair_rows = [
        (
            generation_by_key[(example_id, "original")],
            generation_by_key[(example_id, "reverse")],
        )
        for example_id in example_ids
    ]
    total = len(pair_rows)
    original_correct = sum(left["outcome"] == "correct" for left, _ in pair_rows)
    reverse_correct = sum(right["outcome"] == "correct" for _, right in pair_rows)
    flips = sum(left["action"] != right["action"] for left, right in pair_rows)
    both_correct = sum(
        left["outcome"] == "correct" and right["outcome"] == "correct"
        for left, right in pair_rows
    )
    original_only = sum(
        left["outcome"] == "correct" and right["outcome"] != "correct"
        for left, right in pair_rows
    )
    reverse_only = sum(
        left["outcome"] != "correct" and right["outcome"] == "correct"
        for left, right in pair_rows
    )
    neither = total - both_correct - original_only - reverse_only
    behavior_rows = [
        {"metric": "original_accuracy", "value": original_correct / total, "count": original_correct, "pairs": total},
        {"metric": "reverse_accuracy", "value": reverse_correct / total, "count": reverse_correct, "pairs": total},
        {"metric": "action_flip_rate", "value": flips / total, "count": flips, "pairs": total},
        {"metric": "both_correct", "value": both_correct / total, "count": both_correct, "pairs": total},
        {"metric": "original_only_correct", "value": original_only / total, "count": original_only, "pairs": total},
        {"metric": "reverse_only_correct", "value": reverse_only / total, "count": reverse_only, "pairs": total},
        {"metric": "neither_correct", "value": neither / total, "count": neither, "pairs": total},
    ]

    behavior_path = output_dir / "order_results.csv"
    representation_path = output_dir / "order_representation_results.csv"
    _write_csv(behavior_path, behavior_rows)
    _write_csv(representation_path, representation_rows)
    return [behavior_path, representation_path]


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
