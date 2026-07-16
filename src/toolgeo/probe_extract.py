"""Decision-context residual extraction for auditable outcome probes."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import load_normalized
from .rollout_hf import _prompt


def extract_decision_contexts(input_dir: str, model_id: str, cache_dir: str, layers: str, output: str, max_prompt_tokens: int = 4096) -> None:
    """Save the final prompt residual for each benchmark decision.

    The prompt is deliberately identical to ``rollout-hf`` before the tool-name
    continuation is scored.  It contains the candidate menu, so this is a
    decision-context representation rather than a tool-definition embedding.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install toolgeo[hf] to use extract-decision-hf.") from exc
    tools, decisions, _ = load_normalized(input_dir)
    by_id = {tool.tool_id: tool for tool in tools}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache_dir, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device).eval()
    n_layers = int(getattr(model.config, "num_hidden_layers", 0))
    if layers == "all":
        selected_layers = list(range(n_layers + 1))
    else:
        try:
            selected_layers = [int(layers)]
        except ValueError as exc:
            raise ValueError("layers must be 'all' or one integer layer index") from exc
    if any(layer < 0 or layer > n_layers for layer in selected_layers):
        raise ValueError(f"layers must be in [0,{n_layers}]")
    limit = min(int(getattr(model.config, "max_position_embeddings", max_prompt_tokens)), max_prompt_tokens)
    residuals: list[np.ndarray] = []
    lengths: list[int] = []
    for number, decision in enumerate(decisions, 1):
        ids = tokenizer.encode(_prompt(decision.query, decision.candidate_tool_ids, by_id), add_special_tokens=False)
        ids = ids[-limit:]
        encoded = torch.tensor([ids], device=device)
        with torch.inference_mode():
            hidden_states = model(input_ids=encoded, output_hidden_states=True, use_cache=False).hidden_states
            # fp16 keeps Qwen3-8B × 12k × 37 × 4096 at ~3.6 GiB before
            # compression, while retaining the exact model activations needed
            # by the downstream float32 probe.
            residual = torch.stack([hidden_states[layer][0, -1] for layer in selected_layers])
        residuals.append(residual.cpu().numpy().astype(np.float16, copy=False))
        lengths.append(len(ids))
        if number % 100 == 0 or number == len(decisions):
            print(f"extracted decision contexts {number}/{len(decisions)}", flush=True)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        decision_ids=np.array([item.decision_id for item in decisions]),
        residuals=np.stack(residuals),
        prompt_lengths=np.array(lengths, dtype=np.int32),
        layers=np.array(selected_layers, dtype=np.int32),
        model_id=np.array(model_id),
    )
