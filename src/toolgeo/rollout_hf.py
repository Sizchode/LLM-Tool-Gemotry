"""Native tool-call rendering and exact risk-set scoring."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Sequence

from .schema import Tool


def tool_spec(tool: Tool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.schema or {"type": "object", "properties": {}},
        },
    }


def _tokens(rendered: Any) -> list[int]:
    if isinstance(rendered, Mapping):
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if rendered and isinstance(rendered[0], list):
        rendered = rendered[0]
    return [int(value) for value in rendered]


def _apply_template(
    tokenizer: Any, conversation: list[dict[str, Any]],
    specs: list[dict[str, Any]], **kwargs: Any,
) -> tuple[Any, bool]:
    try:
        try:
            return tokenizer.apply_chat_template(
                conversation, tools=specs, enable_thinking=False, **kwargs,
            ), True
        except TypeError:
            return tokenizer.apply_chat_template(conversation, tools=specs, **kwargs), False
    except Exception as exc:
        raise ValueError("Checkpoint cannot render native structured tools") from exc


def render_native_choice_sequences(
    tokenizer: Any, query: str, candidates: Sequence[str], by_id: dict[str, Tool],
) -> tuple[list[int], list[list[int]], bool]:
    specs = [tool_spec(by_id[item]) for item in candidates]
    user = {"role": "user", "content": query}
    prompt_rendered, thinking_disabled = _apply_template(
        tokenizer, [user], specs, add_generation_prompt=True, tokenize=True,
    )
    prompt = _tokens(prompt_rendered)
    sequences = []
    for tool_id in candidates:
        call = {
            "role": "assistant",
            "content": None,
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
        if len(sequence) <= len(prompt) or sequence[: len(prompt)] != prompt:
            raise ValueError("Native assistant tool call does not extend the native prompt")
        sequences.append(sequence)
    if len({tuple(value) for value in sequences}) != len(sequences):
        raise ValueError("Native template produced indistinguishable candidate calls")
    return prompt, sequences, thinking_disabled


def _constrained_trie_scores(
    model: Any, torch: Any, sequences: Sequence[Sequence[int]],
    common_length: int, device: str, max_branch_batch: int = 8,
) -> tuple[list[float], list[int]]:
    """Sum conditional log-probability only at candidate-divergence branches."""
    if max_branch_batch < 1:
        raise ValueError("max_branch_batch must be positive")
    scores = [0.0] * len(sequences)
    branch_counts = [0] * len(sequences)
    branches: list[tuple[list[int], dict[int, list[int]]]] = []

    def collect(indices: list[int], position: int) -> None:
        if len(indices) <= 1:
            return
        groups: dict[int, list[int]] = {}
        for index in indices:
            if position >= len(sequences[index]):
                raise ValueError("Candidate ended before the native call became distinguishable")
            groups.setdefault(int(sequences[index][position]), []).append(index)
        if len(groups) > 1:
            branches.append((list(sequences[indices[0]][:position]), groups))
        for group in groups.values():
            collect(group, position + 1)

    collect(list(range(len(sequences))), common_length)
    for start in range(0, len(branches), max_branch_batch):
        chunk = branches[start : start + max_branch_batch]
        width = max(len(context) for context, _ in chunk)
        input_ids = torch.zeros((len(chunk), width), dtype=torch.long, device=device)
        attention = torch.zeros_like(input_ids)
        for row, (context, _) in enumerate(chunk):
            input_ids[row, -len(context) :] = torch.tensor(context, dtype=torch.long, device=device)
            attention[row, -len(context) :] = 1
        position_ids = attention.cumsum(-1) - 1
        position_ids.masked_fill_(attention == 0, 0)
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids, attention_mask=attention,
                position_ids=position_ids, use_cache=False,
            ).logits[:, -1].float()
        for row, (_, groups) in enumerate(chunk):
            tokens = list(groups)
            logprob = logits[row, tokens].log_softmax(0).cpu().tolist()
            for token, value in zip(tokens, logprob):
                for index in groups[token]:
                    scores[index] += float(value)
                    branch_counts[index] += 1
    return scores, branch_counts
