"""Native tool-call candidate selection without summed-name likelihoods.

The benchmark menu is rendered through the tokenizer's own tool-aware chat
template. Candidate-name probability is normalized only at branches of the
risk-set token trie, so non-discriminative name length is never accumulated.
"""
from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

from .data import load_normalized
from .hf_model import config_int, load_generation_model
from .io import write_json, write_jsonl
from .schema import Decision, Tool, record

def tool_spec(tool: Tool) -> dict[str, Any]:
    parameters = tool.schema if tool.schema else {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {"name": tool.name, "description": tool.description, "parameters": parameters},
    }


def _tokens(rendered: Any) -> list[int]:
    if isinstance(rendered, Mapping):
        if "input_ids" not in rendered:
            raise ValueError("Tool-aware chat template returned a mapping without input_ids")
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if rendered and isinstance(rendered[0], list):
        rendered = rendered[0]
    return [int(value) for value in rendered]


def _apply_template(tokenizer: Any, conversation: list[dict[str, Any]], specs: list[dict[str, Any]], **kwargs: Any) -> tuple[Any, bool]:
    """Apply a native template, retrying only Qwen's optional thinking flag."""
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Tokenizer has no apply_chat_template; native tool-call rollout is required.")
    try:
        try:
            rendered = tokenizer.apply_chat_template(
                conversation, tools=specs, enable_thinking=False, **kwargs,
            )
            thinking_disabled = True
        except TypeError:
            rendered = tokenizer.apply_chat_template(conversation, tools=specs, **kwargs)
            thinking_disabled = False
    except Exception as exc:
        raise ValueError(
            "The tokenizer chat template could not render structured tools. "
            "Do not replace this with a hand-written menu; use a tool-capable instruct checkpoint."
        ) from exc
    return rendered, thinking_disabled


def render_native_prompt(tokenizer: Any, query: str, candidates: Sequence[str], by_id: dict[str, Tool]) -> list[int]:
    """Render the exact ordered risk set with the model's native chat template."""
    specs = [tool_spec(by_id[item]) for item in candidates]
    rendered, _ = _apply_template(
        tokenizer, [{"role": "user", "content": query}], specs,
        add_generation_prompt=True, tokenize=True,
    )
    return _tokens(rendered)


def render_native_choice_sequences(
    tokenizer: Any, query: str, candidates: Sequence[str], by_id: dict[str, Tool],
) -> tuple[list[int], list[list[int]], bool]:
    """Render each candidate as a native assistant tool call.

    This avoids a Qwen-specific JSON/XML prefix.  A checkpoint whose template
    ignores ``tools`` or cannot serialize assistant ``tool_calls`` is rejected
    because all candidate sequences would be identical or fail to extend the
    generation prompt.
    """
    specs = [tool_spec(by_id[item]) for item in candidates]
    user = {"role": "user", "content": query}
    prompt_rendered, thinking_disabled = _apply_template(
        tokenizer, [user], specs, add_generation_prompt=True, tokenize=True,
    )
    prompt = _tokens(prompt_rendered)
    sequences: list[list[int]] = []
    for tool_id in candidates:
        call = {
            "role": "assistant", "content": None,
            "tool_calls": [{
                "type": "function",
                "function": {"name": by_id[tool_id].name, "arguments": {}},
            }],
        }
        rendered, _ = _apply_template(
            tokenizer, [user, call], specs,
            add_generation_prompt=False, tokenize=True,
        )
        sequence = _tokens(rendered)
        if len(sequence) <= len(prompt) or sequence[:len(prompt)] != prompt:
            raise ValueError(
                "Native assistant tool-call rendering does not extend the generation prompt. "
                "This checkpoint/template cannot support controlled native tool choice."
            )
        sequences.append(sequence)
    if len({tuple(value) for value in sequences}) != len(sequences):
        raise ValueError(
            "Native chat template produced indistinguishable candidate tool calls; "
            "the checkpoint is not a valid behavior model for this experiment."
        )
    return prompt, sequences, thinking_disabled


def _first_unique_positions(sequences: Sequence[Sequence[int]], common_length: int) -> list[int]:
    """Return the first token position whose prefix uniquely identifies each leaf."""
    if len({tuple(value) for value in sequences}) != len(sequences):
        raise ValueError("Candidate tool names are not token-distinguishable in this risk set.")
    result: list[int] = []
    for index, sequence in enumerate(sequences):
        found = None
        for position in range(common_length, len(sequence)):
            prefix = tuple(sequence[: position + 1])
            if all(other == index or tuple(candidate[: position + 1]) != prefix for other, candidate in enumerate(sequences)):
                found = position
                break
        if found is None:
            raise ValueError("Could not find a discriminative token for a candidate tool name.")
        result.append(found)
    return result


def _constrained_trie_scores(model: Any, torch: Any, sequences: Sequence[Sequence[int]], common_length: int, device: str, max_branch_batch: int = 8) -> tuple[list[float], list[int]]:
    """Normalize probability only at branches that distinguish candidates.

    Tokens shared by every remaining candidate have probability one under the
    constrained choice space.  Consequently a long name is not penalized for
    extra non-discriminative tokens.  Branch forwards are chunked so wide
    menus with long prompts stay within single-GPU memory.
    """
    if max_branch_batch < 1:
        raise ValueError("max_branch_batch must be at least one")
    scores = [0.0] * len(sequences)
    branch_counts = [0] * len(sequences)
    branches: list[tuple[list[int], dict[int, list[int]]]] = []

    def collect(indices: list[int], position: int) -> None:
        if len(indices) <= 1:
            return
        groups: dict[int, list[int]] = {}
        for index in indices:
            if position >= len(sequences[index]):
                raise ValueError("Candidate sequence ended before becoming unique; append a name delimiter")
            groups.setdefault(int(sequences[index][position]), []).append(index)
        if len(groups) > 1:
            context = list(sequences[indices[0]][:position])
            branches.append((context, groups))
        for group in groups.values():
            collect(group, position + 1)

    collect(list(range(len(sequences))), common_length)
    if not branches:
        return scores, branch_counts
    for start in range(0, len(branches), max_branch_batch):
        chunk = branches[start:start + max_branch_batch]
        width = max(len(context) for context, _ in chunk)
        # Left padding preserves the last-token position for every branch context.
        # Position IDs are supplied explicitly because decoder-only models may not
        # infer them correctly from arbitrary left padding.
        input_ids = torch.zeros((len(chunk), width), dtype=torch.long, device=device)
        attention = torch.zeros_like(input_ids)
        for row, (context, _) in enumerate(chunk):
            input_ids[row, -len(context):] = torch.tensor(context, dtype=torch.long, device=device)
            attention[row, -len(context):] = 1
        position_ids = attention.cumsum(-1) - 1
        position_ids.masked_fill_(attention == 0, 0)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention, position_ids=position_ids, use_cache=False).logits[:, -1].float()
        for row, (_, groups) in enumerate(chunk):
            tokens = list(groups)
            branch_logprob = logits[row, tokens].log_softmax(0).cpu().tolist()
            for token, logprob in zip(tokens, branch_logprob):
                for index in groups[token]:
                    scores[index] += float(logprob)
                    branch_counts[index] += 1
    return scores, branch_counts


def _ordered_variants(decision: Decision, repeats: int, seed: int) -> list[tuple[list[str], int | None, str]]:
    if repeats < 1:
        raise ValueError("menu_repeats must be at least one")
    variants = [(list(decision.candidate_tool_ids), None, "original")]
    for repeat in range(1, repeats):
        digest = hashlib.sha256(f"{seed}|{decision.decision_id}|{repeat}".encode()).digest()
        order_seed = int.from_bytes(digest[:8], "big")
        ordered = list(decision.candidate_tool_ids)
        random.Random(order_seed).shuffle(ordered)
        variants.append((ordered, order_seed, f"shuffle-{repeat}"))
    return variants


def rollout(
    input_dir: str, model_id: str, cache_dir: str, output: str,
    max_prompt_tokens: int = 4096, menu_repeats: int = 3, seed: int = 17,
    opaque_names: bool = False,
    include_multi_call: bool = False,
    max_branch_batch: int = 8,
) -> None:
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install toolgeo[hf] for rollout-hf.") from exc
    tools, decisions, traces = load_normalized(input_dir)
    by_id = {item.tool_id: item for item in tools}
    if opaque_names:
        alias = {tool_id: f"tool_{index:05d}" for index, tool_id in enumerate(sorted(by_id))}
        rendered_by_id = {
            tool_id: Tool(tool.tool_id, alias[tool_id], tool.description, tool.schema, tool.source)
            for tool_id, tool in by_id.items()
        }
    else:
        alias = {tool_id: tool.name for tool_id, tool in by_id.items()}
        rendered_by_id = by_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, use_fast=True)
    model = load_generation_model(
        model_id, cache_dir, torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    model_limit = config_int(model, "max_position_embeddings", max_prompt_tokens)
    limit = min(model_limit, max_prompt_tokens)
    selected: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    eligible_decisions = [item for item in decisions if include_multi_call or item.gold_call_count == 1]
    total = len(eligible_decisions) * menu_repeats
    completed = 0
    for decision in eligible_decisions:
        for ordered, order_seed, variant_id in _ordered_variants(decision, menu_repeats, seed):
            prompt, sequences, thinking_disabled = render_native_choice_sequences(
                tokenizer, decision.query, ordered, rendered_by_id,
            )
            common_length = 0
            while common_length < min(map(len, sequences)) and len({row[common_length] for row in sequences}) == 1:
                common_length += 1
            unique_positions = _first_unique_positions(sequences, common_length)
            if max(unique_positions) + 1 > limit:
                raise ValueError(
                    f"{decision.decision_id}/{variant_id}: native tool prompt requires "
                    f"{max(unique_positions)+1} tokens (limit={limit}); refusing left truncation."
                )
            scores, branch_counts = _constrained_trie_scores(
                model, torch, sequences, common_length, device, max_branch_batch=max_branch_batch,
            )
            choice_index = max(range(len(scores)), key=scores.__getitem__)
            choice = ordered[choice_index]
            decision_id = decision.decision_id if variant_id == "original" else f"{decision.decision_id}::{variant_id}"
            selected.append(record(Decision(
                decision_id, decision.query, ordered, decision.gold_tool_id, choice, decision.source,
                ordered.index(decision.gold_tool_id) if decision.gold_tool_id in ordered else None,
                choice_index, order_seed, variant_id, decision.gold_call_count,
            )))
            diagnostics.append({
                "decision_id": decision_id, "scoring": "candidate_trie_branch_normalized_logprob",
                "native_chat_template": True,
                "thinking_disabled_for_controlled_choice": thinking_disabled,
                "max_branch_batch": max_branch_batch,
                "prompt_tokens": len(prompt),
                "opaque_name_control": opaque_names,
                "single_call_estimand": not include_multi_call,
                "rendered_candidate_names": [alias[item] for item in ordered],
                "candidate_tool_ids": ordered, "candidate_scores": scores,
                "discriminative_token_offsets": [position - common_length for position in unique_positions],
                "discriminative_branch_counts": branch_counts,
                "chosen_tool_id": choice,
            })
            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"rolled out {completed}/{total} menu decisions", flush=True)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    if opaque_names:
        # The alias map is the single source of truth for the rendered names.
        # Downstream analysis must read it instead of re-deriving the aliases.
        write_json(root / "opaque_alias.json", alias)
    write_jsonl(root / "tools.jsonl", (record(item) for item in tools))
    write_jsonl(root / "decisions.jsonl", selected)
    write_jsonl(root / "rollout_scores.jsonl", diagnostics)
    write_jsonl(root / "traces.jsonl", (record(item) for item in traces))
