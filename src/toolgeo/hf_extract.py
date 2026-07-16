"""Protocolized standalone tool-card residual extraction."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .rollout_hf import DEFAULT_SELECTION_PREFIX
from .schema import Tool

POOLINGS = ("name", "description", "schema", "last", "mean")
CARD_TEMPLATES = (
    ("canonical", ("Tool: ", "\nDescription: ", "\nSchema: ", "\n")),
    ("compact", ("Function `", "` — ", "\nArguments JSON schema: ", "\n")),
    ("prose", ("Available function name: ", "\nWhat it does: ", "\nAccepted arguments: ", "\n")),
)


def render_card(tool: Tool, template_index: int) -> tuple[str, dict[str, tuple[int, int]], str]:
    """Return a controlled card plus exact character spans for its three views."""
    template_name, parts = CARD_TEMPLATES[template_index]
    schema = json.dumps(tool.schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    values = (tool.name, tool.description or "(no description)", schema)
    text = ""
    spans: dict[str, tuple[int, int]] = {}
    for label, prefix, value in zip(("name", "description", "schema"), parts[:3], values):
        text += prefix
        start = len(text)
        text += value
        spans[label] = (start, len(text))
    text += parts[3]
    return text, spans, template_name


def _span_token_indices(offsets: list[tuple[int, int]], span: tuple[int, int]) -> list[int]:
    start, end = span
    indices = [index for index, (left, right) in enumerate(offsets) if right > start and left < end and right > left]
    if not indices:
        raise ValueError(f"No tokenizer offsets overlap character span {span}")
    return indices


def _parse_layers(value: str, n_layers: int) -> list[int]:
    if value == "all":
        return list(range(n_layers + 1))
    try:
        result = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError("layers must be 'all' or comma-separated residual layer indices") from exc
    if not result or any(item < 0 or item > n_layers for item in result):
        raise ValueError(f"layers must lie in [0,{n_layers}]")
    return result


def _contextual_name_token(tokenizer: Any, name: str) -> int:
    text = DEFAULT_SELECTION_PREFIX + name
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    start = len(DEFAULT_SELECTION_PREFIX)
    for token_id, (left, right) in zip(encoded["input_ids"], encoded["offset_mapping"]):
        if right > start and left < len(text):
            return int(token_id)
    raise ValueError(f"Tool name {name!r} contributes no token in tool-call context")


def extract(tools: list[Tool], model_id: str, cache_dir: str, layers: str, output: Path, max_tokens: int = 4096) -> None:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install toolgeo[hf] to use extract-hf.") from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer with return_offsets_mapping is required for span pooling.")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache_dir,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    n_layers = int(getattr(model.config, "num_hidden_layers", 0))
    selected_layers = _parse_layers(layers, n_layers)
    limit = min(int(getattr(model.config, "max_position_embeddings", max_tokens)), max_tokens)
    template_names = [item[0] for item in CARD_TEMPLATES]
    hidden_size = int(getattr(model.config, "hidden_size"))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".residuals.tmp.npy")
    centroid_store = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float16,
        shape=(len(tools), len(selected_layers), len(POOLINGS), hidden_size),
    )
    template_cosine_by_tool = np.empty(
        (len(tools), len(CARD_TEMPLATES), len(selected_layers), len(POOLINGS)), dtype=np.float32,
    )
    for tool_index, tool in enumerate(tools, start=1):
        variants: list[np.ndarray] = []
        for template_index in range(len(CARD_TEMPLATES)):
            text, spans, _ = render_card(tool, template_index)
            encoded = tokenizer(
                text, return_tensors="pt", return_offsets_mapping=True,
                add_special_tokens=True, truncation=False,
            )
            if int(encoded["input_ids"].shape[1]) > limit:
                raise ValueError(
                    f"Tool {tool.tool_id}/{template_names[template_index]} card has "
                    f"{encoded['input_ids'].shape[1]} tokens (limit={limit}); refusing truncation."
                )
            offsets = [tuple(map(int, pair)) for pair in encoded.pop("offset_mapping")[0].tolist()]
            attention = encoded["attention_mask"][0].bool()
            encoded = encoded.to(device)
            with torch.inference_mode():
                states = model(**encoded, output_hidden_states=True, use_cache=False).hidden_states
            valid = [index for index, flag in enumerate(attention.tolist()) if flag and offsets[index][1] > offsets[index][0]]
            indices = {name: _span_token_indices(offsets, span) for name, span in spans.items()}
            pooled_layers: list[np.ndarray] = []
            for layer in selected_layers:
                state = states[layer][0].float()
                pooled = [state[indices[name]].mean(0) for name in ("name", "description", "schema")]
                pooled.extend((state[valid[-1]], state[valid].mean(0)))
                pooled_layers.append(torch.stack(pooled).cpu().numpy())
            variants.append(np.stack(pooled_layers))
        variants_array = np.stack(variants).astype(np.float32)
        normalized = variants_array / np.clip(np.linalg.norm(variants_array, axis=-1, keepdims=True), 1e-12, None)
        centroid = normalized.mean(axis=0)
        centroid /= np.clip(np.linalg.norm(centroid, axis=-1, keepdims=True), 1e-12, None)
        centroid_store[tool_index - 1] = centroid.astype(np.float16)
        template_cosine_by_tool[tool_index - 1] = np.sum(normalized * centroid[None, ...], axis=-1)
        if tool_index % 50 == 0 or tool_index == len(tools):
            print(f"extracted card residuals {tool_index}/{len(tools)}", flush=True)
    token_ids = [_contextual_name_token(tokenizer, tool.name) for tool in tools]
    output_weight = model.get_output_embeddings().weight.detach().float()
    name_unembedding = output_weight[token_ids].cpu().numpy().astype(np.float16)
    centroid_store.flush()
    try:
        np.savez_compressed(
            output,
            tool_ids=np.array([tool.tool_id for tool in tools]),
            centroids=centroid_store,
            # centroids: [tool, residual_layer, pooling, hidden]
            template_cosine_to_centroid=template_cosine_by_tool,
            template_names=np.array(template_names), layers=np.array(selected_layers, dtype=np.int32),
            pooling_names=np.array(POOLINGS), model_id=np.array(model_id),
            name_token_ids=np.array(token_ids, dtype=np.int64),
            name_unembedding=name_unembedding,
            name_unembedding_norm=np.linalg.norm(name_unembedding.astype(np.float32), axis=1),
        )
    finally:
        del centroid_store
        temporary.unlink(missing_ok=True)
