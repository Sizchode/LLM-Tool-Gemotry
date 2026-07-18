"""Render BFCL menus with each checkpoint's official chat template."""
from __future__ import annotations

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
