"""Exact name/description/schema ablations for contextual tool-card geometry."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .context_geometry import (
    _apply_context_template, _span_token_indices, render_context_cards,
)
from .data import load_normalized, validate
from .hf_model import config_int, load_generation_model
from .io import write_json


ABLATIONS = ("no_name", "no_description", "no_schema", "opaque_name")


def _representations(
    tokenizer: Any, model: Any, torch: Any, device: str,
    query: str, candidates: list[str], by_id: dict[str, Any],
    ablation: str, layers: list[int], limit: int,
    opaque_aliases: dict[str, str],
) -> list[Any]:
    content, character_spans = render_context_cards(
        query, candidates, by_id, ablation, opaque_aliases,
    )
    encoded, content_start = _apply_context_template(tokenizer, content)
    offsets = [tuple(map(int, pair)) for pair in encoded.pop("offset_mapping")[0].tolist()]
    spans = [
        _span_token_indices(offsets, (content_start + start, content_start + end))
        for start, end in character_spans
    ]
    if int(encoded["input_ids"].shape[1]) > limit:
        raise ValueError(f"Ablation prompt has {encoded['input_ids'].shape[1]} tokens (limit={limit})")
    encoded = encoded.to(device)
    with torch.inference_mode():
        states = model(**encoded, output_hidden_states=True, use_cache=False).hidden_states
    result = []
    for layer in layers:
        state = states[layer][0].float()
        pooled = torch.stack([state[indices].mean(0) for indices in spans])
        result.append(torch.nn.functional.normalize(pooled, dim=1))
    return result


def ablate(
    input_dir: str, model_id: str, cache_dir: str, output: str,
) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install toolgeo[hf] to run component ablations") from exc
    tools, decisions, traces = load_normalized(input_dir)
    errors = validate(tools, decisions, traces)
    if errors:
        raise ValueError("Invalid normalized data:\n" + "\n".join(errors))
    eligible = [item for item in decisions if item.gold_call_count == 1]
    by_id = {item.tool_id: item for item in tools}
    opaque_aliases = {tool.tool_id: f"tool_{index:05d}" for index, tool in enumerate(tools)}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Exact card spans require a fast tokenizer")
    model = load_generation_model(
        model_id, cache_dir, torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    selected_layers = list(range(config_int(model, "num_hidden_layers") + 1))
    limit = config_int(model, "max_position_embeddings")
    displacement = {name: np.zeros(len(selected_layers), dtype=np.float64) for name in ABLATIONS}
    neighbor_overlap = {name: np.zeros(len(selected_layers), dtype=np.float64) for name in ABLATIONS}
    counts = np.zeros(len(selected_layers), dtype=np.int64)

    for decision_number, decision in enumerate(eligible, start=1):
        candidates = list(decision.candidate_tool_ids)
        full = _representations(
            tokenizer, model, torch, device, decision.query, candidates, by_id,
            "full", selected_layers, limit, opaque_aliases,
        )
        variants = {
            name: _representations(
                tokenizer, model, torch, device, decision.query, candidates, by_id,
                name, selected_layers, limit, opaque_aliases,
            )
            for name in ABLATIONS
        }
        for layer_offset, full_values in enumerate(full):
            n_tools = len(candidates)
            k = min(5, n_tools - 1)
            full_cosine = full_values @ full_values.T
            full_cosine.fill_diagonal_(-float("inf"))
            full_neighbors = torch.topk(full_cosine, k=k, dim=1).indices.cpu().tolist()
            counts[layer_offset] += n_tools
            for name, layer_values in variants.items():
                value = layer_values[layer_offset]
                displacement[name][layer_offset] += float((1 - (full_values * value).sum(1)).sum().cpu())
                cosine = value @ value.T
                cosine.fill_diagonal_(-float("inf"))
                neighbors = torch.topk(cosine, k=k, dim=1).indices.cpu().tolist()
                neighbor_overlap[name][layer_offset] += sum(
                    len(set(left) & set(right)) / k
                    for left, right in zip(full_neighbors, neighbors)
                )
        if decision_number % 25 == 0 or decision_number == len(eligible):
            print(f"ablated {decision_number}/{len(eligible)} decisions", flush=True)

    result = {
        "analysis": "component_ablation_without_learned_probe",
        "model_id": model_id,
        "n_decisions": len(eligible),
        "layers": [
            {
                "layer": layer,
                "mean_cosine_displacement": {
                    name: float(displacement[name][offset] / counts[offset]) for name in ABLATIONS
                },
                "within_menu_neighbor_overlap_at_5": {
                    name: float(neighbor_overlap[name][offset] / counts[offset]) for name in ABLATIONS
                },
            }
            for offset, layer in enumerate(selected_layers)
        ],
    }
    write_json(Path(output), result)
    return result
