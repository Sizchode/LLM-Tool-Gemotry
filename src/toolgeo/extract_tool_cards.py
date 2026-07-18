"""Extract mean-pooled Qwen3 tool-card states from every decoder layer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from .data.bfcl import load_from_config
from .model import ToolCardStateRecorder, load_model, load_tokenizer, model_family
from .prompts import MENU_ORDERS, prompt_record, tool_card_token_spans


def canonical_card(function: dict[str, Any]) -> str:
    return json.dumps(
        function, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract(config: dict[str, Any]) -> Path:
    output_dir = Path(config["run"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_from_config(config)
    prompt_interface = load_tokenizer(config["model"])
    family = model_family(config["model"])
    if family != "qwen3":
        raise ValueError("RQ1 tool-card extraction is currently defined for Qwen3")
    enable_thinking = config["generation"]["enable_thinking"]
    if not isinstance(enable_thinking, bool):
        raise ValueError("generation.enable_thinking must be a boolean")

    model = load_model(config["model"])
    device = next(model.parameters()).device
    recorder = ToolCardStateRecorder(model, family)
    metadata: list[dict[str, Any]] = []
    state_batches: list[torch.Tensor] = []
    prompt_count = 0
    try:
        with torch.inference_mode():
            for example in examples:
                for order in MENU_ORDERS:
                    record = prompt_record(
                        prompt_interface,
                        example,
                        order,
                        family,
                        enable_thinking,
                    )
                    functions = record["functions"]
                    encoded, char_spans, token_spans = tool_card_token_spans(
                        prompt_interface,
                        record["prompt"],
                        functions,
                        family,
                    )
                    if not (
                        len(functions) == len(char_spans) == len(token_spans)
                    ):
                        raise ValueError("tool-card spans do not match the menu")
                    first_row = len(metadata)
                    for position, (function, char_span, token_span) in enumerate(
                        zip(functions, char_spans, token_spans)
                    ):
                        metadata.append(
                            {
                                "row_index": len(metadata),
                                "example_id": example.example_id,
                                "order": order,
                                "menu_position": position,
                                "tool_name": str(function["name"]),
                                "card_json": canonical_card(function),
                                "gold_tool": example.gold_tool,
                                "char_start": char_span[0],
                                "char_end": char_span[1],
                                "token_start": token_span[0],
                                "token_end": token_span[1],
                            }
                        )

                    encoded = {key: value.to(device) for key, value in encoded.items()}
                    recorder.reset(token_spans)
                    model(
                        **encoded,
                        output_hidden_states=False,
                        use_cache=False,
                        return_dict=True,
                        logits_to_keep=1,
                    )
                    states = recorder.states().cpu()
                    if states.shape[0] != len(functions):
                        raise ValueError("recorded state count does not match the menu")
                    if metadata[first_row]["row_index"] != sum(
                        batch.shape[0] for batch in state_batches
                    ):
                        raise ValueError("tool-card metadata and state rows diverged")
                    state_batches.append(states)
                    prompt_count += 1
                    print(
                        f"tool-card prompt={prompt_count} cards={len(functions)}",
                        flush=True,
                    )
    finally:
        recorder.close()

    all_states = torch.cat(state_batches, dim=0)
    if all_states.shape[0] != len(metadata):
        raise ValueError("tool-card metadata and state tensor have different lengths")
    metadata_path = output_dir / "tool_cards.jsonl"
    _write_jsonl(metadata_path, metadata)
    state_path = output_dir / "tool_card_states.pt"
    torch.save(
        {
            "row_indices": [row["row_index"] for row in metadata],
            "decoder_layers": list(range(all_states.shape[1])),
            "states": all_states,
        },
        state_path,
    )
    print(
        f"saved prompts={prompt_count} cards={len(metadata)} "
        f"layers={all_states.shape[1]}",
        flush=True,
    )
    return state_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    print(extract(config))


if __name__ == "__main__":
    main()
