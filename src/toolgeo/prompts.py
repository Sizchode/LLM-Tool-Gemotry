"""Render BFCL menus with each checkpoint's official chat template."""
from __future__ import annotations

import json
from typing import Any

from .data.bfcl import BFCLExample
from .model import text_tokenizer


MENU_ORDERS = ("original", "reverse")
THINKING_CONTROL_FAMILIES = {"qwen3", "gemma4"}


def ordered_functions(
    functions: tuple[dict[str, Any], ...], order: str
) -> tuple[dict[str, Any], ...]:
    if order == "original":
        return functions
    if order == "reverse":
        return tuple(reversed(functions))
    raise ValueError(f"unsupported menu order: {order}")


def tool_schemas(functions: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": dict(function)} for function in functions
    ]


def render_prompt(
    prompt_interface: Any,
    example: BFCLExample,
    order: str,
    family: str,
    enable_thinking: bool,
) -> str:
    if family == "gemma3":
        raise ValueError(
            "Gemma 3 has no official Hugging Face tool template; the exact Wu et al. "
            "Gemma 3 prompt protocol is required before this replication can run"
        )
    functions = ordered_functions(example.functions, order)
    kwargs = {
        "tools": tool_schemas(functions),
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if family in THINKING_CONTROL_FAMILIES:
        kwargs["enable_thinking"] = enable_thinking
    elif enable_thinking:
        raise ValueError(f"thinking mode is not defined for model family {family}")
    rendered = prompt_interface.apply_chat_template(list(example.messages), **kwargs)
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("official chat template returned an empty prompt")
    missing = [str(function["name"]) for function in functions if str(function["name"]) not in rendered]
    if missing:
        raise ValueError(
            "official chat template did not render candidate tools: " + ", ".join(missing)
        )
    return rendered


def prompt_record(
    prompt_interface: Any,
    example: BFCLExample,
    order: str,
    family: str,
    enable_thinking: bool,
) -> dict[str, Any]:
    functions = ordered_functions(example.functions, order)
    return {
        "example_id": example.example_id,
        "order": order,
        "messages": list(example.messages),
        "functions": list(functions),
        "candidate_tools": [str(function["name"]) for function in functions],
        "gold_tool": example.gold_tool,
        "exact_query_key": example.exact_query_key,
        "prompt": render_prompt(
            prompt_interface, example, order, family, enable_thinking
        ),
        "thinking_enabled": enable_thinking,
        "model_family": family,
    }


def tokenize_prompts(prompt_interface: Any, prompts: list[str]) -> dict[str, Any]:
    tokenizer = text_tokenizer(prompt_interface)
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    try:
        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
    finally:
        tokenizer.padding_side = previous_padding_side
    return encoded


def qwen3_tool_card_char_spans(
    prompt: str, functions: list[dict[str, Any]]
) -> list[tuple[int, int]]:
    """Locate the exact JSON card lines emitted by the official Qwen3 template."""
    opening = "<tools>\n"
    closing = "\n</tools>"
    if prompt.count(opening) != 1 or prompt.count(closing) != 1:
        raise ValueError("Qwen3 prompt does not contain exactly one tools block")
    body_start = prompt.index(opening) + len(opening)
    body_end = prompt.index(closing, body_start)
    body = prompt[body_start:body_end]
    lines = body.split("\n")
    expected = tool_schemas(tuple(functions))
    if len(lines) != len(expected):
        raise ValueError("Qwen3 tools block line count does not match candidate menu")

    spans: list[tuple[int, int]] = []
    cursor = body_start
    for line, schema in zip(lines, expected):
        try:
            rendered_schema = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("Qwen3 tool card line is not valid JSON") from exc
        if rendered_schema != schema:
            raise ValueError("Qwen3 rendered tool card differs from the BFCL schema")
        spans.append((cursor, cursor + len(line)))
        cursor += len(line) + 1
    if cursor - 1 != body_end:
        raise ValueError("Qwen3 tool card character spans do not cover the tools block")
    return spans


def tool_card_token_spans(
    prompt_interface: Any,
    prompt: str,
    functions: list[dict[str, Any]],
    family: str,
) -> tuple[dict[str, Any], list[tuple[int, int]], list[tuple[int, int]]]:
    """Map exact rendered card character spans to overlapping prompt tokens."""
    if family != "qwen3":
        raise ValueError("tool-card span extraction is currently defined for Qwen3")
    char_spans = qwen3_tool_card_char_spans(prompt, functions)
    tokenizer = text_tokenizer(prompt_interface)
    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = [tuple(map(int, pair)) for pair in encoded.pop("offset_mapping")[0]]
    token_spans: list[tuple[int, int]] = []
    for char_start, char_end in char_spans:
        indices = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > char_start and token_start < char_end
        ]
        if not indices:
            raise ValueError("rendered tool card contains no tokenizer tokens")
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError("tool-card tokens are not contiguous")
        token_start = indices[0]
        token_end = indices[-1] + 1
        if offsets[token_start][1] <= char_start or offsets[token_end - 1][0] >= char_end:
            raise ValueError("token span does not overlap both card boundaries")
        token_spans.append((token_start, token_end))
    return encoded, char_spans, token_spans
