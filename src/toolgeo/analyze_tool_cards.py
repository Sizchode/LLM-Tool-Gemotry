"""Compute exact all-layer BFCL tool-card geometry measurements."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _pairwise_cosine_components(
    normalized: torch.Tensor, labels: torch.Tensor, class_count: int
) -> tuple[int, float, int, float]:
    """Return exact within- and between-class unordered-pair counts and means."""
    count = normalized.shape[0]
    total_pair_count = count * (count - 1) // 2
    total_sum = float(
        ((normalized.sum(dim=0).square().sum() - count) / 2).item()
    )

    class_sums = torch.zeros(
        class_count, normalized.shape[1], dtype=normalized.dtype
    )
    class_sums.index_add_(0, labels, normalized)
    class_sizes = torch.bincount(labels, minlength=class_count)
    within_pair_count = int(
        (class_sizes * (class_sizes - 1) // 2).sum().item()
    )
    within_sum = float(
        (
            (
                class_sums.square().sum(dim=1)
                - class_sizes.to(normalized.dtype)
            )
            / 2
        ).sum().item()
    )
    between_pair_count = total_pair_count - within_pair_count
    if within_pair_count == 0 or between_pair_count == 0:
        raise ValueError("within/between cosine requires both pair types")
    return (
        within_pair_count,
        within_sum / within_pair_count,
        between_pair_count,
        (total_sum - within_sum) / between_pair_count,
    )


def _leave_context_out_scores(
    train_states: torch.Tensor,
    test_states: torch.Tensor,
    labels: torch.Tensor,
    example_ids: list[str],
    class_count: int,
) -> torch.Tensor:
    """Score each test state against original-order prototypes excluding its menu."""
    sums = torch.zeros(class_count, train_states.shape[1], dtype=torch.float32)
    sums.index_add_(0, labels, train_states)
    counts = torch.bincount(labels, minlength=class_count)
    if bool((counts < 2).any()):
        raise ValueError("retrieval classes must occur in at least two menus")
    prototypes = F.normalize(sums / counts[:, None], dim=1)
    scores = F.normalize(test_states, dim=1) @ prototypes.T

    rows_by_example: defaultdict[str, list[int]] = defaultdict(list)
    for row, example_id in enumerate(example_ids):
        rows_by_example[example_id].append(row)
    for rows in rows_by_example.values():
        menu_labels = labels[rows]
        if len(set(menu_labels.tolist())) != len(rows):
            raise ValueError("an exact tool card occurs twice in one candidate menu")
        for removed_row, class_index in zip(rows, menu_labels.tolist()):
            adjusted = F.normalize(
                (sums[class_index] - train_states[removed_row])
                / (counts[class_index] - 1),
                dim=0,
            )
            scores[rows, class_index] = (
                F.normalize(test_states[rows], dim=1) @ adjusted
            )
    return scores


def analyze(config: dict[str, Any]) -> Path:
    output_dir = Path(config["run"]["output_dir"])
    metadata = _read_jsonl(output_dir / "tool_cards.jsonl")
    artifact = torch.load(
        output_dir / "tool_card_states.pt", map_location="cpu", weights_only=True
    )
    states = artifact["states"]
    if states.ndim != 3 or states.shape[0] != len(metadata):
        raise ValueError("tool-card state tensor does not match metadata")
    if artifact["row_indices"] != list(range(len(metadata))):
        raise ValueError("tool-card state rows are not in metadata order")

    by_key: dict[tuple[str, str, str], int] = {}
    for row in metadata:
        key = (row["example_id"], row["order"], row["card_json"])
        if key in by_key:
            raise ValueError("duplicate exact tool card within one ordered menu")
        by_key[key] = int(row["row_index"])

    original_rows = [
        int(row["row_index"]) for row in metadata if row["order"] == "original"
    ]
    reverse_rows = []
    for index in original_rows:
        row = metadata[index]
        key = (row["example_id"], "reverse", row["card_json"])
        if key not in by_key:
            raise ValueError("original tool card has no reverse-order match")
        reverse_rows.append(by_key[key])
    original_cards = [metadata[index]["card_json"] for index in original_rows]
    original_examples = [metadata[index]["example_id"] for index in original_rows]

    all_classes = sorted(set(original_cards))
    all_class_index = {card: index for index, card in enumerate(all_classes)}
    all_labels = torch.tensor(
        [all_class_index[card] for card in original_cards], dtype=torch.long
    )
    card_counts = Counter(original_cards)
    supported_cards = sorted(card for card, count in card_counts.items() if count > 1)
    if not supported_cards:
        raise ValueError("no exact tool card occurs in more than one menu")
    supported_index = {card: index for index, card in enumerate(supported_cards)}
    eligible_positions = [
        index for index, card in enumerate(original_cards) if card in supported_index
    ]
    retrieval_labels = torch.tensor(
        [supported_index[original_cards[index]] for index in eligible_positions],
        dtype=torch.long,
    )
    retrieval_examples = [original_examples[index] for index in eligible_positions]
    eligible_original_rows = [original_rows[index] for index in eligible_positions]
    eligible_reverse_rows = [reverse_rows[index] for index in eligible_positions]

    results: list[dict[str, Any]] = []
    decoder_layers = artifact["decoder_layers"]
    if decoder_layers != list(range(states.shape[1])):
        raise ValueError("decoder layer labels are not raw zero-based block indices")
    for layer in decoder_layers:
        original = states[original_rows, layer].float()
        reverse = states[reverse_rows, layer].float()
        normalized_original = F.normalize(original, dim=1)
        normalized_reverse = F.normalize(reverse, dim=1)
        within_count, within_mean, between_count, between_mean = (
            _pairwise_cosine_components(
                normalized_original, all_labels, len(all_classes)
            )
        )

        train = states[eligible_original_rows, layer].float()
        same_test = train
        reverse_test = states[eligible_reverse_rows, layer].float()
        same_scores = _leave_context_out_scores(
            train,
            same_test,
            retrieval_labels,
            retrieval_examples,
            len(supported_cards),
        )
        reverse_scores = _leave_context_out_scores(
            train,
            reverse_test,
            retrieval_labels,
            retrieval_examples,
            len(supported_cards),
        )
        same_correct = int((same_scores.argmax(dim=1) == retrieval_labels).sum())
        reverse_correct = int(
            (reverse_scores.argmax(dim=1) == retrieval_labels).sum()
        )
        matched_cosine = (normalized_original * normalized_reverse).sum(dim=1)
        results.append(
            {
                "decoder_layer": layer,
                "card_occurrences": len(original_rows),
                "exact_card_classes": len(all_classes),
                "repeated_card_classes": len(supported_cards),
                "retrieval_occurrences": len(eligible_positions),
                "same_order_correct": same_correct,
                "same_order_accuracy": same_correct / len(eligible_positions),
                "reverse_order_correct": reverse_correct,
                "reverse_order_accuracy": reverse_correct / len(eligible_positions),
                "within_pair_count": within_count,
                "within_cosine": within_mean,
                "between_pair_count": between_count,
                "between_cosine": between_mean,
                "within_minus_between_cosine": within_mean - between_mean,
                "matched_order_pairs": len(original_rows),
                "original_reverse_cosine": float(matched_cosine.mean().item()),
            }
        )
        print(f"analyzed decoder_layer={layer}", flush=True)

    output_path = output_dir / "tool_card_rq1_all_layers.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    print(analyze(config))


if __name__ == "__main__":
    main()
