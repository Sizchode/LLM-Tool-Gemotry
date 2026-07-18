"""Analyze matched menu-order choices and geometry of real in-menu errors."""
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


IN_MENU_OUTCOMES = {"correct", "wrong_in_menu"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pair_category(original: dict[str, Any], reverse: dict[str, Any]) -> str:
    left = original["outcome"]
    right = reverse["outcome"]
    if left == "correct" and right == "correct":
        return "both_correct"
    if left == "correct" and right == "wrong_in_menu":
        return "original_correct_reverse_wrong"
    if left == "wrong_in_menu" and right == "correct":
        return "original_wrong_reverse_correct"
    if left == "wrong_in_menu" and right == "wrong_in_menu":
        return (
            "both_wrong_same_tool"
            if original["parsed_tool"] == reverse["parsed_tool"]
            else "both_wrong_different_tools"
        )
    return f"{left}_to_{right}"


def analyze(config: dict[str, Any]) -> list[Path]:
    output_dir = Path(config["run"]["output_dir"])
    examples = _read_jsonl(output_dir / "examples.jsonl")
    generations = _read_jsonl(output_dir / "generations.jsonl")
    cards = _read_jsonl(output_dir / "tool_cards.jsonl")
    card_payload = torch.load(
        output_dir / "tool_card_states.pt", map_location="cpu", weights_only=True
    )
    states = card_payload["states"]
    if states.ndim != 3 or states.shape[0] != len(cards):
        raise ValueError("tool-card states do not match tool-card metadata")

    example_by_key = {
        (row["example_id"], row["order"]): row for row in examples
    }
    generation_by_key = {
        (row["example_id"], row["order"]): row for row in generations
    }
    if set(example_by_key) != set(generation_by_key):
        raise ValueError("generation and example key sets differ")
    card_by_key: dict[tuple[str, str, str], int] = {}
    for row in cards:
        key = (row["example_id"], row["order"], row["tool_name"])
        if key in card_by_key:
            raise ValueError("tool names are not unique inside a candidate menu")
        card_by_key[key] = int(row["row_index"])

    example_ids = [
        row["example_id"] for row in examples if row["order"] == "original"
    ]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("original-order example IDs are not unique")

    pair_rows: list[dict[str, Any]] = []
    transition_counts: Counter[tuple[str, str]] = Counter()
    category_counts: Counter[str] = Counter()
    for example_id in example_ids:
        original = generation_by_key[(example_id, "original")]
        reverse = generation_by_key[(example_id, "reverse")]
        original_example = example_by_key[(example_id, "original")]
        reverse_example = example_by_key[(example_id, "reverse")]
        if original_example["candidate_tools"] != list(
            reversed(reverse_example["candidate_tools"])
        ):
            raise ValueError("reverse condition changed more than menu order")
        category = _pair_category(original, reverse)
        transition_counts[(original["outcome"], reverse["outcome"])] += 1
        category_counts[category] += 1
        both_parsed = (
            original["parsed_tool"] is not None
            and reverse["parsed_tool"] is not None
        )
        pair_rows.append(
            {
                "example_id": example_id,
                "gold_tool": original["gold_tool"],
                "menu_size": len(original_example["candidate_tools"]),
                "original_outcome": original["outcome"],
                "reverse_outcome": reverse["outcome"],
                "original_tool": original["parsed_tool"],
                "reverse_tool": reverse["parsed_tool"],
                "both_parsed": both_parsed,
                "same_parsed_tool": (
                    original["parsed_tool"] == reverse["parsed_tool"]
                    if both_parsed
                    else ""
                ),
                "both_in_menu": (
                    original["outcome"] in IN_MENU_OUTCOMES
                    and reverse["outcome"] in IN_MENU_OUTCOMES
                ),
                "category": category,
            }
        )

    transition_rows = [
        {
            "original_outcome": left,
            "reverse_outcome": right,
            "count": count,
            "paired_decisions": len(example_ids),
        }
        for (left, right), count in sorted(transition_counts.items())
    ]
    category_rows = [
        {
            "category": category,
            "count": count,
            "paired_decisions": len(example_ids),
        }
        for category, count in sorted(category_counts.items())
    ]

    locality_rows: list[dict[str, Any]] = []
    decoder_layers = card_payload["decoder_layers"]
    for generation in generations:
        if generation["outcome"] != "wrong_in_menu":
            continue
        example_id = generation["example_id"]
        order = generation["order"]
        example = example_by_key[(example_id, order)]
        candidates = list(example["candidate_tools"])
        gold = str(example["gold_tool"])
        selected = str(generation["parsed_tool"])
        if gold == selected or gold not in candidates or selected not in candidates:
            raise ValueError("wrong_in_menu row is inconsistent with its menu")
        row_indices = [
            card_by_key[(example_id, order, tool)] for tool in candidates
        ]
        gold_index = candidates.index(gold)
        selected_index = candidates.index(selected)
        non_gold_indices = [
            index for index, tool in enumerate(candidates) if tool != gold
        ]
        other_indices = [
            index
            for index in non_gold_indices
            if index != selected_index
        ]
        for layer in decoder_layers:
            menu_states = F.normalize(states[row_indices, layer].float(), dim=1)
            similarities = menu_states @ menu_states[gold_index]
            selected_similarity = float(similarities[selected_index])
            other_non_gold = similarities[other_indices]
            strictly_closer = int((other_non_gold > selected_similarity).sum())
            equal_similarity = int((other_non_gold == selected_similarity).sum())
            if other_indices:
                other_mean = float(other_non_gold.mean())
                selected_minus_other = selected_similarity - other_mean
            else:
                other_mean = ""
                selected_minus_other = ""
            locality_rows.append(
                {
                    "example_id": example_id,
                    "order": order,
                    "decoder_layer": layer,
                    "gold_tool": gold,
                    "selected_wrong_tool": selected,
                    "menu_size": len(candidates),
                    "non_gold_candidates": len(non_gold_indices),
                    "other_unselected_distractors": len(other_indices),
                    "gold_selected_cosine": selected_similarity,
                    "strictly_closer_distractors": strictly_closer,
                    "equal_similarity_distractors": equal_similarity,
                    "selected_is_nearest_non_gold": strictly_closer == 0,
                    "other_distractor_mean_cosine": other_mean,
                    "selected_minus_other_mean_cosine": selected_minus_other,
                }
            )

    grouped: defaultdict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in locality_rows:
        grouped[(row["decoder_layer"], row["order"], row["menu_size"])].append(row)
    locality_summary: list[dict[str, Any]] = []
    for (layer, order, menu_size), rows in sorted(grouped.items()):
        with_other = [
            row for row in rows if row["other_unselected_distractors"] > 0
        ]
        locality_summary.append(
            {
                "decoder_layer": layer,
                "order": order,
                "menu_size": menu_size,
                "wrong_in_menu_errors": len(rows),
                "selected_nearest_count": sum(
                    bool(row["selected_is_nearest_non_gold"]) for row in rows
                ),
                "selected_nearest_fraction": sum(
                    bool(row["selected_is_nearest_non_gold"]) for row in rows
                )
                / len(rows),
                "mean_gold_selected_cosine": sum(
                    float(row["gold_selected_cosine"]) for row in rows
                )
                / len(rows),
                "errors_with_other_unselected_distractors": len(with_other),
                "mean_selected_minus_other_cosine": (
                    sum(
                        float(row["selected_minus_other_mean_cosine"])
                        for row in with_other
                    )
                    / len(with_other)
                    if with_other
                    else ""
                ),
            }
        )

    matched_geometry_rows: list[dict[str, Any]] = []
    for example_id in example_ids:
        original_generation = generation_by_key[(example_id, "original")]
        reverse_generation = generation_by_key[(example_id, "reverse")]
        category = _pair_category(original_generation, reverse_generation)
        if category == "original_correct_reverse_wrong":
            focal_wrong_tool = str(reverse_generation["parsed_tool"])
        elif category == "original_wrong_reverse_correct":
            focal_wrong_tool = str(original_generation["parsed_tool"])
        elif category == "both_wrong_same_tool":
            focal_wrong_tool = str(original_generation["parsed_tool"])
        else:
            continue
        gold = str(original_generation["gold_tool"])
        for order in ("original", "reverse"):
            candidates = example_by_key[(example_id, order)]["candidate_tools"]
            if gold not in candidates or focal_wrong_tool not in candidates:
                raise ValueError("matched focal tool is absent from an ordered menu")
        original_gold_row = card_by_key[(example_id, "original", gold)]
        original_focal_row = card_by_key[
            (example_id, "original", focal_wrong_tool)
        ]
        reverse_gold_row = card_by_key[(example_id, "reverse", gold)]
        reverse_focal_row = card_by_key[(example_id, "reverse", focal_wrong_tool)]
        for layer in decoder_layers:
            original_cosine = float(
                F.cosine_similarity(
                    states[original_gold_row, layer].float()[None],
                    states[original_focal_row, layer].float()[None],
                ).item()
            )
            reverse_cosine = float(
                F.cosine_similarity(
                    states[reverse_gold_row, layer].float()[None],
                    states[reverse_focal_row, layer].float()[None],
                ).item()
            )
            matched_geometry_rows.append(
                {
                    "example_id": example_id,
                    "decoder_layer": layer,
                    "category": category,
                    "gold_tool": gold,
                    "focal_wrong_tool": focal_wrong_tool,
                    "original_gold_focal_cosine": original_cosine,
                    "reverse_gold_focal_cosine": reverse_cosine,
                    "reverse_minus_original_cosine": (
                        reverse_cosine - original_cosine
                    ),
                }
            )

    matched_groups: defaultdict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in matched_geometry_rows:
        matched_groups[(row["decoder_layer"], row["category"])].append(row)
    matched_summary = []
    for (layer, category), rows in sorted(matched_groups.items()):
        matched_summary.append(
            {
                "decoder_layer": layer,
                "category": category,
                "matched_decisions": len(rows),
                "mean_original_gold_focal_cosine": sum(
                    float(row["original_gold_focal_cosine"]) for row in rows
                )
                / len(rows),
                "mean_reverse_gold_focal_cosine": sum(
                    float(row["reverse_gold_focal_cosine"]) for row in rows
                )
                / len(rows),
                "mean_reverse_minus_original_cosine": sum(
                    float(row["reverse_minus_original_cosine"]) for row in rows
                )
                / len(rows),
            }
        )

    outputs = [
        output_dir / "tool_choice_pairs.csv",
        output_dir / "tool_choice_transition_counts.csv",
        output_dir / "tool_choice_category_counts.csv",
        output_dir / "error_locality_all_layers.csv",
        output_dir / "error_locality_by_menu_size.csv",
        output_dir / "matched_choice_geometry_all_layers.csv",
        output_dir / "matched_choice_geometry_summary.csv",
    ]
    _write_csv(outputs[0], pair_rows, list(pair_rows[0]))
    _write_csv(outputs[1], transition_rows, list(transition_rows[0]))
    _write_csv(outputs[2], category_rows, list(category_rows[0]))
    _write_csv(outputs[3], locality_rows, list(locality_rows[0]))
    _write_csv(outputs[4], locality_summary, list(locality_summary[0]))
    _write_csv(
        outputs[5], matched_geometry_rows, list(matched_geometry_rows[0])
    )
    _write_csv(outputs[6], matched_summary, list(matched_summary[0]))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for output in analyze(config):
        print(output)


if __name__ == "__main__":
    main()
