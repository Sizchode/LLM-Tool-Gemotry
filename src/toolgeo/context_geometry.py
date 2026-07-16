"""Contextual tool-card geometry and native tool choice for Paper 1."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_normalized, validate
from .hf_model import config_int, load_generation_model
from .io import write_jsonl
from .rollout_hf import _constrained_trie_scores, render_native_choice_sequences
from .schema import Decision, Tool, record


def _span_token_indices(offsets: list[tuple[int, int]], span: tuple[int, int]) -> list[int]:
    start, end = span
    indices = [
        index for index, (left, right) in enumerate(offsets)
        if right > start and left < end and right > left
    ]
    if not indices:
        raise ValueError(f"No tokenizer offsets overlap the declared character span {span}")
    return indices


def render_context_cards(
    query: str,
    candidates: list[str],
    by_id: dict[str, Tool],
    ablation: str = "full",
    opaque_aliases: dict[str, str] | None = None,
) -> tuple[str, list[tuple[int, int]]]:
    """Build the declared measurement prompt and record exact card spans.

    This renderer is deliberately explicit. It never searches for a tool name
    in model-specific template output and never infers a span by diffing two
    prompts. The returned character offsets are created at the same moment as
    the text they index.
    """
    allowed = {"full", "no_name", "no_description", "no_schema", "opaque_name"}
    if ablation not in allowed:
        raise ValueError(f"ablation must be one of {sorted(allowed)}")
    text = "Available tools:\n\n"
    spans: list[tuple[int, int]] = []
    for tool_id in candidates:
        tool = by_id[tool_id]
        name = tool.name
        description = tool.description or "(empty)"
        schema = json.dumps(tool.schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if ablation == "opaque_name":
            if opaque_aliases is None or tool_id not in opaque_aliases:
                raise ValueError("opaque_name requires the declared dataset-level alias map")
            name = opaque_aliases[tool_id]
        start = len(text)
        if ablation != "no_name":
            text += f"Name: {name}\n"
        if ablation != "no_description":
            text += f"Description: {description}\n"
        if ablation != "no_schema":
            text += f"Schema: {schema}\n"
        spans.append((start, len(text)))
        text += "\n"
    # No trailing whitespace: chat templates are allowed to trim message-end
    # whitespace, which would make the declared character coordinate system
    # differ from the rendered one.
    text += f"Query:\n{query}"
    return text, spans


def _apply_context_template(tokenizer: Any, content: str) -> tuple[dict[str, Any], int]:
    """Render one user message and map its exact content into template offsets."""
    conversation = [{"role": "user", "content": content}]
    try:
        try:
            rendered = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True,
            )
    except Exception as exc:
        raise ValueError("Tokenizer cannot render the declared contextual-card prompt") from exc
    if not isinstance(rendered, str):
        raise ValueError("tokenize=False chat-template rendering did not return text")
    start = rendered.find(content)
    if start < 0 or rendered.find(content, start + 1) >= 0:
        raise ValueError("The exact card content must occur exactly once in the rendered chat prompt")
    encoded = tokenizer(
        rendered, return_tensors="pt", return_offsets_mapping=True,
        add_special_tokens=False, truncation=False,
    )
    return encoded, start


def _context_variants(candidates: list[str]) -> list[tuple[str, list[str]]]:
    """The only position intervention: original order and its exact reverse."""
    reversed_order = list(reversed(candidates))
    if reversed_order == candidates:
        return [("original", list(candidates))]
    return [("original", list(candidates)), ("reverse", reversed_order)]


def measure(
    input_dir: str,
    model_id: str,
    cache_dir: str,
    output: str,
    max_branch_batch: int = 8,
) -> None:
    """Measure every eligible decision; no sampled contexts or learned probe."""
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install toolgeo[hf] to use measure-hf.") from exc

    tools, decisions, traces = load_normalized(input_dir)
    errors = validate(tools, decisions, traces)
    if errors:
        raise ValueError("Invalid normalized data:\n" + "\n".join(errors))
    eligible = [item for item in decisions if item.gold_call_count == 1]
    by_id = {item.tool_id: item for item in tools}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Exact card spans require a fast tokenizer with offset mappings")
    model = load_generation_model(
        model_id, cache_dir, torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    selected_layers = list(range(config_int(model, "num_hidden_layers") + 1))
    hidden_size = config_int(model, "hidden_size")
    limit = config_int(model, "max_position_embeddings")

    occurrence_count = sum(
        sum(len(order) for _, order in _context_variants(item.candidate_tool_ids))
        for item in eligible
    )
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    final_shards = [root / f"context_representations.layer_{layer:03d}.npy" for layer in selected_layers]
    temporary = [path.with_suffix(".tmp.npy") for path in final_shards]
    stores = [
        np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(occurrence_count, hidden_size))
        for path in temporary
    ]
    context_rows: list[dict[str, Any]] = []
    choice_rows: list[dict[str, Any]] = []
    measurement_rows: list[dict[str, Any]] = []
    occurrence = 0
    try:
        for decision_number, decision in enumerate(eligible, start=1):
            original_pooled: list[Any] | None = None
            for variant, ordered in _context_variants(decision.candidate_tool_ids):
                content, character_spans = render_context_cards(decision.query, ordered, by_id)
                encoded, content_start = _apply_context_template(tokenizer, content)
                offsets = [tuple(map(int, pair)) for pair in encoded.pop("offset_mapping")[0].tolist()]
                token_spans = [
                    _span_token_indices(offsets, (content_start + start, content_start + end))
                    for start, end in character_spans
                ]
                if int(encoded["input_ids"].shape[1]) > limit:
                    raise ValueError(
                        f"{decision.decision_id}/{variant}: prompt has {encoded['input_ids'].shape[1]} "
                        f"tokens (limit={limit}); refusing truncation"
                    )
                encoded = encoded.to(device)
                with torch.inference_mode():
                    states = model(**encoded, output_hidden_states=True, use_cache=False).hidden_states
                pooled_layers = []
                for layer_index, layer in enumerate(selected_layers):
                    state = states[layer][0].float()
                    pooled = torch.stack([state[indices].mean(0) for indices in token_spans])
                    pooled = torch.nn.functional.normalize(pooled, dim=1)
                    pooled_layers.append(pooled)
                    stores[layer_index][occurrence : occurrence + len(ordered)] = pooled.cpu().numpy().astype(np.float16)
                for position, tool_id in enumerate(ordered):
                    context_rows.append({
                        "decision_id": decision.decision_id,
                        "context_variant": variant,
                        "tool_id": tool_id,
                        "menu_position": position,
                        "menu_size": len(ordered),
                    })
                occurrence += len(ordered)
                if variant == "original":
                    original_pooled = pooled_layers

            ordered = list(decision.candidate_tool_ids)
            _, sequences, thinking_disabled = render_native_choice_sequences(
                tokenizer, decision.query, ordered, by_id,
            )
            if max(map(len, sequences)) > limit:
                raise ValueError(
                    f"{decision.decision_id}: native candidate call exceeds {limit} tokens"
                )
            common = 0
            while common < min(map(len, sequences)) and len({row[common] for row in sequences}) == 1:
                common += 1
            scores, _ = _constrained_trie_scores(
                model, torch, sequences, common, device, max_branch_batch=max_branch_batch,
            )
            chosen_index = max(range(len(scores)), key=scores.__getitem__)
            chosen = ordered[chosen_index]
            choice_rows.append(record(Decision(
                decision.decision_id, decision.query, ordered, decision.gold_tool_id, chosen,
                decision.source, ordered.index(decision.gold_tool_id), chosen_index,
                None, "original", decision.gold_call_count,
            )))
            gold_index = ordered.index(decision.gold_tool_id)
            if original_pooled is None:
                raise RuntimeError("Original contextual geometry was not measured")
            similarities = [
                (pooled @ pooled[gold_index]).cpu().numpy().astype(float).tolist()
                for pooled in original_pooled
            ]
            measurement_rows.append({
                "decision_id": decision.decision_id,
                "candidate_tool_ids": ordered,
                "gold_tool_id": decision.gold_tool_id,
                "chosen_tool_id": chosen,
                "layers": selected_layers,
                "gold_candidate_cosine_by_layer": similarities,
                "candidate_scores": scores,
                "thinking_disabled": thinking_disabled,
            })
            if decision_number % 25 == 0 or decision_number == len(eligible):
                print(f"measured {decision_number}/{len(eligible)} decisions", flush=True)

        if occurrence != occurrence_count:
            raise RuntimeError(f"Wrote {occurrence} occurrences, expected {occurrence_count}")
        for store in stores:
            store.flush()
        del stores
        for source, target in zip(temporary, final_shards):
            source.replace(target)
        np.savez_compressed(
            root / "geometry_index.npz",
            model_id=np.array(model_id),
            pooling=np.array("exact_tool_card_span_mean"),
            context_variants=np.array(["original", "reverse"]),
            layers=np.array(selected_layers, dtype=np.int32),
            representation_shards=np.array([path.name for path in final_shards]),
        )
        write_jsonl(root / "tools.jsonl", (record(item) for item in tools))
        write_jsonl(root / "decisions.jsonl", choice_rows)
        write_jsonl(root / "traces.jsonl", (record(item) for item in traces))
        write_jsonl(root / "context_index.jsonl", context_rows)
        write_jsonl(root / "geometry_measurements.jsonl", measurement_rows)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
