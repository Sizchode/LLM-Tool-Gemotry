"""Deterministic model-side candidate selection for Paper 1 behaviour labels."""
from __future__ import annotations

from pathlib import Path

from .data import load_normalized
from .io import write_jsonl
from .schema import Decision

def _prompt(query, candidates, by_id):
    menu = "\n".join(f"- {by_id[x].name}: {by_id[x].description}; schema={by_id[x].schema}" for x in candidates)
    return f"You are a tool-calling agent. Choose the single best tool.\nTools:\n{menu}\n\nUser request: {query}\nTool name:"

def rollout(input_dir: str, model_id: str, cache_dir: str, output: str, max_prompt_tokens: int = 4096) -> None:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc: raise RuntimeError("Install toolgeo[hf] for rollout-hf.") from exc
    tools, decisions, traces = load_normalized(input_dir); by_id = {x.tool_id: x for x in tools}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
    limit = min(int(getattr(model.config, "max_position_embeddings", max_prompt_tokens)), max_prompt_tokens)
    selected = []
    for number, decision in enumerate(decisions, 1):
        prefix = tokenizer.encode(_prompt(decision.query, decision.candidate_tool_ids, by_id), add_special_tokens=False)
        prefix = prefix[-(limit - 32):]
        candidates = [(tool_id, tokenizer.encode(" " + by_id[tool_id].name, add_special_tokens=False)) for tool_id in decision.candidate_tool_ids]
        sequences = [prefix + continuation for _, continuation in candidates]
        width = max(map(len, sequences)); pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        ids = torch.tensor([row + [pad] * (width - len(row)) for row in sequences], device=device)
        mask = torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in sequences], device=device)
        with torch.inference_mode(): logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
        scores = []
        for batch, (_, continuation) in enumerate(candidates):
            positions = range(len(prefix) - 1, len(prefix) + len(continuation) - 1)
            scores.append(sum(float(logits[batch, pos].log_softmax(-1)[token].cpu()) for pos, token in zip(positions, continuation)))
        choice = candidates[max(range(len(scores)), key=scores.__getitem__)][0]
        selected.append(Decision(decision.decision_id, decision.query, decision.candidate_tool_ids, decision.gold_tool_id, choice, decision.source).__dict__)
        if number % 50 == 0 or number == len(decisions): print(f"rolled out {number}/{len(decisions)} decisions", flush=True)
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / "tools.jsonl", (x.__dict__ for x in tools)); write_jsonl(root / "decisions.jsonl", selected); write_jsonl(root / "traces.jsonl", (x.__dict__ for x in traces))
