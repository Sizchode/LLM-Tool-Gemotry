"""Optional Hugging Face extraction for Paper 1 internal/unembedding features."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .features import _normalise, _hash_vector
from .schema import Tool

def extract(tools: list[Tool], model_id: str, cache_dir: str, layer: int, output: Path) -> None:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install toolgeo[hf] to use extract-hf.") from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
    n_layers = int(getattr(model.config, "num_hidden_layers", 0))
    if layer < 0 or layer > n_layers:
        raise ValueError(f"layer must be in [0,{n_layers}]")
    # Tool-specific matched prompts.  Never concatenate an entire benchmark
    # tool inventory: it is both a confounded measurement and can exceed the
    # model context window.  Context truncation is an explicit safety guard.
    internal = []
    max_length = min(int(getattr(model.config, "max_position_embeddings", 4096)), 4096)
    for index, tool in enumerate(tools, start=1):
        text = f"Available tool: {tool.name}: {tool.description}; schema={tool.schema}\nUser request: select a tool.\nAssistant:"
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        with torch.inference_mode():
            state = model(**encoded, output_hidden_states=True, use_cache=False).hidden_states[layer][0, -1].float().cpu().numpy()
        internal.append(state)
        if index % 100 == 0 or index == len(tools): print(f"extracted {index}/{len(tools)} tools", flush=True)
    token_ids = [tokenizer.encode(tool.name, add_special_tokens=False)[0] for tool in tools]
    unembedding = model.get_output_embeddings().weight.detach().float()[token_ids].cpu().numpy()
    dimension = internal[0].shape[0]
    np.savez_compressed(output, tool_ids=np.array([tool.tool_id for tool in tools]), internal=_normalise(np.stack(internal)),
                        unembedding=_normalise(unembedding), description=_normalise(np.stack([_hash_vector(tool.description, dimension) for tool in tools])),
                        schema=_normalise(np.stack([_hash_vector(str(sorted(tool.schema.items())), dimension) for tool in tools])))
