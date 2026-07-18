"""Read matched original/reverse tool-call prefixes with a Jacobian Lens."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from jlens.hf import from_hf
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.vis import _ranks_of

from .model import load_model, load_tokenizer, model_family, text_tokenizer
from .read_tool_calls_with_jacobian_lens import _generated_name_tokens


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty result: {path}")
    if path.suffix == ".gz":
        handle_context = gzip.open(path, "wt", encoding="utf-8", newline="")
    else:
        handle_context = path.open("w", encoding="utf-8", newline="")
    with handle_context as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (left_id, right_id) in enumerate(zip(left, right, strict=False)):
        if left_id != right_id:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _opening_token_index(
    raw_output: str, offsets: list[tuple[int, int]]
) -> int:
    opening = "<tool_call>"
    char_start = raw_output.find(opening)
    if char_start < 0:
        raise ValueError("parsed tool call has no opening tool_call tag")
    char_end = char_start + len(opening)
    indices = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > char_start and start < char_end
    ]
    if not indices:
        raise ValueError("opening tool_call tag maps to no generated token")
    return indices[0]


def _pair_scope(
    *,
    original_ids: list[int],
    reverse_ids: list[int],
    original_name_indices: list[int],
    reverse_name_indices: list[int],
    original_call_start: int,
    reverse_call_start: int,
    same_tool: bool,
) -> dict[str, Any]:
    if original_call_start != reverse_call_start:
        return {
            "status": "tool_call_starts_at_different_positions",
            "start": None,
            "end": None,
            "first_output_difference": _first_difference(original_ids, reverse_ids),
            "first_name_difference": None,
        }

    output_difference = _first_difference(original_ids, reverse_ids)
    original_name_ids = [original_ids[index] for index in original_name_indices]
    reverse_name_ids = [reverse_ids[index] for index in reverse_name_indices]
    name_offset = _first_difference(original_name_ids, reverse_name_ids)
    name_difference = None
    if (
        output_difference is not None
        and output_difference in original_name_indices
        and output_difference in reverse_name_indices
    ):
        name_difference = output_difference

    if same_tool:
        if original_name_ids != reverse_name_ids:
            raise ValueError("identical parsed tool names have different token IDs")
        end = original_name_indices[-1]
        status = "same_tool_common_name_trajectory"
        if output_difference is not None and output_difference <= end:
            end = output_difference
            status = "same_tool_output_diverged_before_name_end"
    else:
        if name_offset is None:
            raise ValueError("different parsed tool names have identical token IDs")
        if output_difference is None:
            raise ValueError("different parsed tools have identical generated outputs")
        end = output_difference
        if output_difference < min(original_name_indices[0], reverse_name_indices[0]):
            status = "different_tools_output_diverged_before_name"
        else:
            status = "different_tools_diverge_at_name"

    if end is None or end < original_call_start:
        return {
            "status": status,
            "start": None,
            "end": None,
            "first_output_difference": output_difference,
            "first_name_difference": name_difference,
        }
    return {
        "status": status,
        "start": original_call_start,
        "end": end,
        "first_output_difference": output_difference,
        "first_name_difference": name_difference,
    }


def _readout_values(
    logits: torch.Tensor,
    targets: dict[str, torch.Tensor],
) -> dict[str, list[Any]]:
    log_probabilities = logits.float().log_softmax(dim=-1)
    values: dict[str, list[Any]] = {"top_ids": logits.argmax(dim=-1).tolist()}
    names = list(targets)
    target_matrix = torch.stack([targets[name] for name in names], dim=1)
    ranks = _ranks_of(logits.float(), target_matrix)
    target_log_probabilities = log_probabilities.gather(1, target_matrix)
    for column, name in enumerate(names):
        values[f"{name}_ranks"] = [int(value) for value in ranks[:, column].tolist()]
        values[f"{name}_log_probabilities"] = target_log_probabilities[
            :, column
        ].tolist()
    return values


def _condition_readouts(
    *,
    lens_model: Any,
    lens: JacobianLens,
    prompt_ids: list[int],
    generated_ids: list[int],
    positions: list[int],
    targets: dict[str, torch.Tensor],
) -> dict[tuple[int, str], dict[str, list[Any]]]:
    full_ids = torch.tensor(
        [prompt_ids + generated_ids], dtype=torch.long, device=lens_model.input_device
    )
    readout_positions = [len(prompt_ids) + position - 1 for position in positions]
    if min(readout_positions) < 0:
        raise ValueError("generated target has no preceding residual position")
    target_layer = lens_model.n_layers - 1
    outputs: dict[tuple[int, str], dict[str, list[Any]]] = {}
    with torch.inference_mode():
        with ActivationRecorder(
            lens_model.layers, at=[*lens.source_layers, target_layer]
        ) as recorder:
            lens_model.forward(full_ids)
        final_residual = recorder.activations[target_layer][0, readout_positions].float()
        final_logits = lens_model.unembed(final_residual).float()
        outputs[(target_layer, "final_model")] = _readout_values(
            final_logits, targets
        )
        del final_logits, final_residual
        for layer in lens.source_layers:
            residual = recorder.activations[layer][0, readout_positions].float()
            logits = lens_model.unembed(lens.transport(residual, layer)).float()
            outputs[(layer, "jacobian_lens")] = _readout_values(
                logits, targets
            )
            del logits, residual
    return outputs


def read_matched_trajectories(
    config: dict[str, Any], lens_path: Path, result_name: str
) -> list[Path]:
    if model_family(config["model"]) != "qwen3":
        raise ValueError("the current family launcher supports Qwen3 checkpoints")
    output_dir = Path(config["run"]["output_dir"])
    examples = _read_jsonl(output_dir / "examples.jsonl")
    generations = _read_jsonl(output_dir / "generations.jsonl")
    example_by_key = {(row["example_id"], row["order"]): row for row in examples}
    generation_by_key = {
        (row["example_id"], row["order"]): row for row in generations
    }
    if set(example_by_key) != set(generation_by_key):
        raise ValueError("generation and example key sets differ")
    example_ids = [
        row["example_id"] for row in examples if row["order"] == "original"
    ]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("original-order example IDs are not unique")

    interface = load_tokenizer(config["model"])
    tokenizer = text_tokenizer(interface)
    model = load_model(config["model"])
    lens_model = from_hf(model, tokenizer, compile=False, force_bos=False)
    lens = JacobianLens.load(str(lens_path))
    if lens.d_model != lens_model.d_model:
        raise ValueError("lens and model residual widths differ")
    if lens.source_layers != list(range(lens_model.n_layers - 1)):
        raise ValueError("lens must cover every block below the final block")

    trajectory_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for number, example_id in enumerate(example_ids, start=1):
        original_example = example_by_key[(example_id, "original")]
        reverse_example = example_by_key[(example_id, "reverse")]
        original_generation = generation_by_key[(example_id, "original")]
        reverse_generation = generation_by_key[(example_id, "reverse")]
        if original_example["candidate_tools"] != list(
            reversed(reverse_example["candidate_tools"])
        ):
            raise ValueError(f"{example_id}: reverse changed more than menu order")
        original_tool = original_generation["parsed_tool"]
        reverse_tool = reverse_generation["parsed_tool"]
        if original_tool is None or reverse_tool is None:
            pair_rows.append(
                {
                    "example_id": example_id,
                    "original_tool": original_tool,
                    "reverse_tool": reverse_tool,
                    "same_tool": "",
                    "status": "one_or_both_calls_unparsed",
                    "first_output_difference": "",
                    "first_name_difference": "",
                    "trajectory_positions": 0,
                }
            )
            continue

        original_ids, original_offsets, original_name_indices = _generated_name_tokens(
            tokenizer, original_generation["raw_output"], str(original_tool)
        )
        reverse_ids, reverse_offsets, reverse_name_indices = _generated_name_tokens(
            tokenizer, reverse_generation["raw_output"], str(reverse_tool)
        )
        scope = _pair_scope(
            original_ids=original_ids,
            reverse_ids=reverse_ids,
            original_name_indices=original_name_indices,
            reverse_name_indices=reverse_name_indices,
            original_call_start=_opening_token_index(
                original_generation["raw_output"], original_offsets
            ),
            reverse_call_start=_opening_token_index(
                reverse_generation["raw_output"], reverse_offsets
            ),
            same_tool=original_tool == reverse_tool,
        )
        if scope["start"] is None:
            positions: list[int] = []
        else:
            positions = list(range(int(scope["start"]), int(scope["end"]) + 1))
        pair_rows.append(
            {
                "example_id": example_id,
                "original_tool": original_tool,
                "reverse_tool": reverse_tool,
                "same_tool": original_tool == reverse_tool,
                "status": scope["status"],
                "first_output_difference": (
                    scope["first_output_difference"]
                    if scope["first_output_difference"] is not None
                    else ""
                ),
                "first_name_difference": (
                    scope["first_name_difference"]
                    if scope["first_name_difference"] is not None
                    else ""
                ),
                "trajectory_positions": len(positions),
            }
        )
        if not positions:
            continue
        if any(position >= len(original_ids) or position >= len(reverse_ids) for position in positions):
            raise ValueError(f"{example_id}: comparison position exceeds an output")
        for position in positions[:-1]:
            if original_ids[position] != reverse_ids[position]:
                raise ValueError(f"{example_id}: trajectory continued after output divergence")

        original_next_targets = torch.tensor(
            [original_ids[position] for position in positions],
            dtype=torch.long,
            device=lens_model.input_device,
        )
        reverse_next_targets = torch.tensor(
            [reverse_ids[position] for position in positions],
            dtype=torch.long,
            device=lens_model.input_device,
        )
        if original_tool == reverse_tool:
            original_branch_id = original_ids[original_name_indices[0]]
            reverse_branch_id = reverse_ids[reverse_name_indices[0]]
            branch_token_role = "shared_first_tool_name_token"
        else:
            branch_position = int(scope["end"])
            original_branch_id = original_ids[branch_position]
            reverse_branch_id = reverse_ids[branch_position]
            branch_token_role = "first_divergent_tool_name_token"
        original_branch_targets = torch.full_like(
            original_next_targets, original_branch_id
        )
        reverse_branch_targets = torch.full_like(
            reverse_next_targets, reverse_branch_id
        )
        targets = {
            "original_next": original_next_targets,
            "reverse_next": reverse_next_targets,
            "original_branch": original_branch_targets,
            "reverse_branch": reverse_branch_targets,
        }
        original_prompt_ids = tokenizer(
            original_example["prompt"], add_special_tokens=False
        )["input_ids"]
        reverse_prompt_ids = tokenizer(
            reverse_example["prompt"], add_special_tokens=False
        )["input_ids"]
        original_readouts = _condition_readouts(
            lens_model=lens_model,
            lens=lens,
            prompt_ids=original_prompt_ids,
            generated_ids=original_ids,
            positions=positions,
            targets=targets,
        )
        reverse_readouts = _condition_readouts(
            lens_model=lens_model,
            lens=lens,
            prompt_ids=reverse_prompt_ids,
            generated_ids=reverse_ids,
            positions=positions,
            targets=targets,
        )
        if set(original_readouts) != set(reverse_readouts):
            raise ValueError("original and reverse readouts cover different layers")
        for (layer, readout), original_values in sorted(original_readouts.items()):
            reverse_values = reverse_readouts[(layer, readout)]
            for offset, generated_position in enumerate(positions):
                original_target = int(original_next_targets[offset])
                reverse_target = int(reverse_next_targets[offset])
                trajectory_rows.append(
                    {
                        "example_id": example_id,
                        "status": scope["status"],
                        "original_tool": original_tool,
                        "reverse_tool": reverse_tool,
                        "same_tool": original_tool == reverse_tool,
                        "generated_position": generated_position,
                        "position_from_tool_call_start": offset,
                        "is_output_divergence": original_target != reverse_target,
                        "original_target_id": original_target,
                        "original_target": tokenizer.decode(
                            [original_target], skip_special_tokens=False
                        ),
                        "reverse_target_id": reverse_target,
                        "reverse_target": tokenizer.decode(
                            [reverse_target], skip_special_tokens=False
                        ),
                        "branch_token_role": branch_token_role,
                        "original_branch_id": original_branch_id,
                        "original_branch_token": tokenizer.decode(
                            [original_branch_id], skip_special_tokens=False
                        ),
                        "reverse_branch_id": reverse_branch_id,
                        "reverse_branch_token": tokenizer.decode(
                            [reverse_branch_id], skip_special_tokens=False
                        ),
                        "decoder_layer": layer,
                        "readout": readout,
                        "original_context_top_id": original_values["top_ids"][offset],
                        "original_context_top_token": tokenizer.decode(
                            [original_values["top_ids"][offset]],
                            skip_special_tokens=False,
                        ),
                        "original_context_original_next_rank": original_values[
                            "original_next_ranks"
                        ][offset],
                        "original_context_reverse_next_rank": original_values[
                            "reverse_next_ranks"
                        ][offset],
                        "original_context_original_next_log_probability": original_values[
                            "original_next_log_probabilities"
                        ][offset],
                        "original_context_reverse_next_log_probability": original_values[
                            "reverse_next_log_probabilities"
                        ][offset],
                        "original_context_original_branch_rank": original_values[
                            "original_branch_ranks"
                        ][offset],
                        "original_context_reverse_branch_rank": original_values[
                            "reverse_branch_ranks"
                        ][offset],
                        "original_context_original_branch_log_probability": original_values[
                            "original_branch_log_probabilities"
                        ][offset],
                        "original_context_reverse_branch_log_probability": original_values[
                            "reverse_branch_log_probabilities"
                        ][offset],
                        "reverse_context_top_id": reverse_values["top_ids"][offset],
                        "reverse_context_top_token": tokenizer.decode(
                            [reverse_values["top_ids"][offset]],
                            skip_special_tokens=False,
                        ),
                        "reverse_context_original_next_rank": reverse_values[
                            "original_next_ranks"
                        ][offset],
                        "reverse_context_reverse_next_rank": reverse_values[
                            "reverse_next_ranks"
                        ][offset],
                        "reverse_context_original_next_log_probability": reverse_values[
                            "original_next_log_probabilities"
                        ][offset],
                        "reverse_context_reverse_next_log_probability": reverse_values[
                            "reverse_next_log_probabilities"
                        ][offset],
                        "reverse_context_original_branch_rank": reverse_values[
                            "original_branch_ranks"
                        ][offset],
                        "reverse_context_reverse_branch_rank": reverse_values[
                            "reverse_branch_ranks"
                        ][offset],
                        "reverse_context_original_branch_log_probability": reverse_values[
                            "original_branch_log_probabilities"
                        ][offset],
                        "reverse_context_reverse_branch_log_probability": reverse_values[
                            "reverse_branch_log_probabilities"
                        ][offset],
                    }
                )
        if number % 25 == 0:
            print(f"processed_pairs={number}/{len(example_ids)}", flush=True)

    outputs = [
        output_dir / f"menu_order_jacobian_{result_name}_pairs.csv",
        output_dir / f"menu_order_jacobian_{result_name}_trajectories.csv.gz",
    ]
    _write_csv(outputs[0], pair_rows)
    _write_csv(outputs[1], trajectory_rows)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--lens", required=True)
    parser.add_argument("--result-name", required=True)
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for output in read_matched_trajectories(
        config, Path(args.lens), args.result_name
    ):
        print(output)


if __name__ == "__main__":
    main()
