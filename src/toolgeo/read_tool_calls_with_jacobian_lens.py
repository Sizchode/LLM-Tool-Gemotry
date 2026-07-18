"""Read generated BFCL tool-name tokens with a fitted Jacobian Lens."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml
from jlens.hf import from_hf
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.vis import _ranks_of

from .model import load_model, load_tokenizer, model_family, text_tokenizer


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty result: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _top_level_name_span(raw_output: str, parsed_name: str) -> tuple[int, int]:
    """Return the character span inside the top-level JSON ``name`` value."""
    opening = "<tool_call>"
    closing = "</tool_call>"
    start = raw_output.find(opening)
    end = raw_output.find(closing, start + len(opening))
    if start < 0 or end < 0:
        raise ValueError("parsed Qwen call has no complete tool_call block")
    payload_start = start + len(opening)
    payload = raw_output[payload_start:end].strip()
    parsed = json.loads(payload)
    if not isinstance(parsed, dict) or parsed.get("name") != parsed_name:
        raise ValueError("saved parsed name differs from the generated JSON")

    key = json.dumps("name")
    key_start = payload.find(key)
    if key_start < 0:
        raise ValueError("generated JSON has no top-level name key")
    colon = payload.find(":", key_start + len(key))
    if colon < 0:
        raise ValueError("generated JSON name key has no value")
    value_start = colon + 1
    while value_start < len(payload) and payload[value_start].isspace():
        value_start += 1
    encoded_name = json.dumps(parsed_name, ensure_ascii=False)
    if not payload.startswith(encoded_name, value_start):
        raise ValueError("top-level name value is not the parsed JSON string")

    leading_space = len(raw_output[payload_start:end]) - len(
        raw_output[payload_start:end].lstrip()
    )
    absolute_value_start = payload_start + leading_space + value_start
    return absolute_value_start + 1, absolute_value_start + len(encoded_name) - 1


def _generated_name_tokens(
    tokenizer: Any, raw_output: str, parsed_name: str
) -> tuple[list[int], list[tuple[int, int]], list[int]]:
    encoded = tokenizer(
        raw_output,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    generated_ids = [int(token_id) for token_id in encoded["input_ids"]]
    offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]
    if tokenizer.decode(generated_ids, skip_special_tokens=False) != raw_output:
        raise ValueError("saved generation does not round-trip through its tokenizer")
    char_start, char_end = _top_level_name_span(raw_output, parsed_name)
    indices = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > char_start and token_start < char_end
    ]
    if not indices or indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError("generated tool name does not map to contiguous tokens")
    return generated_ids, offsets, indices


def _token_rows(
    *,
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    tokenizer: Any,
    layer: int,
    readout: str,
    example: dict[str, Any],
    generation: dict[str, Any],
    generated_indices: list[int],
    offsets: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    ranks = _ranks_of(logits.float(), target_ids).diagonal().tolist()
    top_ids = logits.argmax(dim=-1).tolist()
    rows = []
    for name_index, (generated_index, target_id, rank, top_id) in enumerate(
        zip(generated_indices, target_ids.tolist(), ranks, top_ids, strict=True)
    ):
        char_start, char_end = offsets[generated_index]
        rows.append(
            {
                "example_id": example["example_id"],
                "order": example["order"],
                "generation_outcome": generation["outcome"],
                "parsed_tool": generation["parsed_tool"],
                "tool_name_token_index": name_index,
                "generated_token_index": generated_index,
                "generated_char_start": char_start,
                "generated_char_end": char_end,
                "decoder_layer": layer,
                "readout": readout,
                "target_token_id": target_id,
                "target_token": tokenizer.decode(
                    [target_id], skip_special_tokens=False
                ),
                "target_rank_zero_based": int(rank),
                "top_token_id": int(top_id),
                "top_token": tokenizer.decode(
                    [top_id], skip_special_tokens=False
                ),
                "target_is_top1": int(rank) == 0,
            }
        )
    return rows


def read_tool_calls(
    config: dict[str, Any], lens_path: Path, result_name: str
) -> list[Path]:
    if model_family(config["model"]) != "qwen3":
        raise ValueError("the current family launcher supports Qwen3 checkpoints")
    output_dir = Path(config["run"]["output_dir"])
    examples = _read_jsonl(output_dir / "examples.jsonl")
    generations = _read_jsonl(output_dir / "generations.jsonl")
    if len(examples) != len(generations):
        raise ValueError("examples and generations have different row counts")
    for example, generation in zip(examples, generations, strict=True):
        if (example["example_id"], example["order"]) != (
            generation["example_id"],
            generation["order"],
        ):
            raise ValueError("examples and generations are not aligned")

    prompt_interface = load_tokenizer(config["model"])
    tokenizer = text_tokenizer(prompt_interface)
    model = load_model(config["model"])
    lens_model = from_hf(model, tokenizer, compile=False, force_bos=False)
    lens = JacobianLens.load(str(lens_path))
    if lens.d_model != lens_model.d_model:
        raise ValueError("lens and model residual widths differ")
    if lens.source_layers != list(range(lens_model.n_layers - 1)):
        raise ValueError("lens must cover every block below the final block")
    target_layer = lens_model.n_layers - 1

    result_rows: list[dict[str, Any]] = []
    valid_calls = 0
    for number, (example, generation) in enumerate(
        zip(examples, generations, strict=True), start=1
    ):
        parsed_name = generation["parsed_tool"]
        if parsed_name is None:
            continue
        valid_calls += 1
        prompt_ids = tokenizer(
            example["prompt"], add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0].tolist()
        generated_ids, offsets, name_indices = _generated_name_tokens(
            tokenizer, generation["raw_output"], str(parsed_name)
        )
        full_ids = torch.tensor(
            [prompt_ids + generated_ids],
            dtype=torch.long,
            device=lens_model.input_device,
        )
        readout_positions = [len(prompt_ids) + index - 1 for index in name_indices]
        if min(readout_positions) < 0:
            raise ValueError("tool-name token has no preceding readout position")
        target_ids = torch.tensor(
            [generated_ids[index] for index in name_indices],
            dtype=torch.long,
            device=lens_model.input_device,
        )

        with torch.inference_mode():
            with ActivationRecorder(
                lens_model.layers,
                at=[*lens.source_layers, target_layer],
            ) as recorder:
                lens_model.forward(full_ids)
            final_residual = recorder.activations[target_layer][
                0, readout_positions
            ].float()
            final_logits = lens_model.unembed(final_residual).float()
            final_rows = _token_rows(
                logits=final_logits,
                target_ids=target_ids,
                tokenizer=tokenizer,
                layer=target_layer,
                readout="final_model",
                example=example,
                generation=generation,
                generated_indices=name_indices,
                offsets=offsets,
            )
            result_rows.extend(final_rows)

            for layer in lens.source_layers:
                residual = recorder.activations[layer][0, readout_positions].float()
                logits = lens_model.unembed(lens.transport(residual, layer)).float()
                result_rows.extend(
                    _token_rows(
                        logits=logits,
                        target_ids=target_ids,
                        tokenizer=tokenizer,
                        layer=layer,
                        readout="jacobian_lens",
                        example=example,
                        generation=generation,
                        generated_indices=name_indices,
                        offsets=offsets,
                    )
                )
                del logits
        del recorder, full_ids, final_residual, final_logits
        if number % 25 == 0:
            print(
                f"processed_rows={number}/{len(examples)} valid_calls={valid_calls}",
                flush=True,
            )

    grouped: defaultdict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in result_rows:
        grouped[(int(row["decoder_layer"]), str(row["readout"]))].append(row)
    summary_rows = []
    for (layer, readout), rows in sorted(grouped.items()):
        top1 = sum(bool(row["target_is_top1"]) for row in rows)
        summary_rows.append(
            {
                "decoder_layer": layer,
                "readout": readout,
                "tool_name_tokens": len(rows),
                "target_top1_count": top1,
                "target_top1_fraction": top1 / len(rows),
                "lens_n_prompts": lens.n_prompts if readout == "jacobian_lens" else "",
            }
        )

    outputs = [
        output_dir / f"jacobian_lens_{result_name}_tool_name_tokens.csv",
        output_dir / f"jacobian_lens_{result_name}_tool_name_summary.csv",
    ]
    _write_csv(outputs[0], result_rows)
    _write_csv(outputs[1], summary_rows)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--lens", required=True)
    parser.add_argument("--result-name", required=True)
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for output in read_tool_calls(config, Path(args.lens), args.result_name):
        print(output)


if __name__ == "__main__":
    main()
