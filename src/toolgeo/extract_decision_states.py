"""Extract every decoder-layer state at the final BFCL prompt token."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from .data.bfcl import load_from_config, write_precheck
from .model import DecisionStateRecorder, load_model, load_tokenizer, model_family
from .prompts import MENU_ORDERS, prompt_record, tokenize_prompts


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _batches(rows: list[dict[str, Any]], size: int):
    if size < 1:
        raise ValueError("extraction batch_size must be positive")
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def extract(config: dict[str, Any], config_path: Path) -> Path:
    output_dir = Path(config["run"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    examples = load_from_config(config)
    write_precheck(output_dir / "bfcl_precheck.csv", examples)
    tokenizer = load_tokenizer(config["model"])
    family = model_family(config["model"])
    enable_thinking = config["generation"]["enable_thinking"]
    if not isinstance(enable_thinking, bool):
        raise ValueError("generation.enable_thinking must be a boolean")
    records = [
        prompt_record(tokenizer, example, order, family, enable_thinking)
        for example in examples
        for order in MENU_ORDERS
    ]
    _write_jsonl(output_dir / "examples.jsonl", records)

    model = load_model(config["model"])
    device = next(model.parameters()).device
    state_batches: list[torch.Tensor] = []
    batch_size = int(config["extraction"]["batch_size"])
    recorder = DecisionStateRecorder(model, family)
    try:
        with torch.inference_mode():
            for number, batch in enumerate(_batches(records, batch_size), start=1):
                encoded = tokenize_prompts(tokenizer, [row["prompt"] for row in batch])
                encoded = {key: value.to(device) for key, value in encoded.items()}
                recorder.reset()
                model(
                    **encoded,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                    logits_to_keep=1,
                )
                states = recorder.states(encoded["attention_mask"])
                state_batches.append(states.cpu())
                print(f"extraction batch={number} examples={len(batch)}", flush=True)
    finally:
        recorder.close()

    all_states = torch.cat(state_batches, dim=0)
    replication_layer = int(config["analysis"]["replication_decoder_layer"])
    if replication_layer not in range(all_states.shape[1]):
        raise ValueError("replication_decoder_layer is outside the decoder")
    state_path = output_dir / "states.pt"
    torch.save(
        {
            "example_ids": [row["example_id"] for row in records],
            "orders": [row["order"] for row in records],
            "decoder_layers": list(range(all_states.shape[1])),
            "states": all_states,
        },
        state_path,
    )
    return state_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    print(extract(config, config_path))


if __name__ == "__main__":
    main()
