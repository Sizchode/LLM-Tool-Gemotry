from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer

from toolgeo.data.bfcl import BFCLExample
from toolgeo.prompts import (
    prompt_record,
    qwen3_tool_card_char_spans,
    render_prompt,
    tool_card_token_spans,
)
import pytest


MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
CACHE = "/oscar/scratch/zliu328/llm_tool_ckpt/hf"


def _example() -> BFCLExample:
    return BFCLExample(
        example_id="fixture",
        messages=({"role": "user", "content": "Use beta."},),
        functions=(
            {
                "name": "alpha",
                "description": "Alpha tool",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "beta",
                "description": "Beta tool",
                "parameters": {"type": "object", "properties": {}},
            },
        ),
        gold_tool="beta",
    )


def _tokenizer():
    return AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=CACHE
    )


def test_official_prompt_contains_full_menu_in_requested_order() -> None:
    tokenizer = _tokenizer()
    original = render_prompt(tokenizer, _example(), "original", "qwen3", True)
    reverse = render_prompt(tokenizer, _example(), "reverse", "qwen3", True)
    assert original.index('"name": "alpha"') < original.index('"name": "beta"')
    assert reverse.index('"name": "beta"') < reverse.index('"name": "alpha"')
    assert "Use beta." in original
    assert "Use beta." in reverse


def test_extraction_and_generation_share_saved_thinking_prompt() -> None:
    tokenizer = _tokenizer()
    record = prompt_record(tokenizer, _example(), "original", "qwen3", True)
    assert record["thinking_enabled"] is True
    assert record["prompt"] == render_prompt(
        tokenizer, _example(), "original", "qwen3", True
    )
    assert record["candidate_tools"] == ["alpha", "beta"]


def test_nonthinking_mode_is_recorded_and_rendered() -> None:
    tokenizer = _tokenizer()
    record = prompt_record(tokenizer, _example(), "original", "qwen3", False)
    assert record["thinking_enabled"] is False
    assert record["prompt"].endswith("<think>\n\n</think>\n\n")


def test_qwen3_card_spans_are_exact_and_follow_menu_order() -> None:
    tokenizer = _tokenizer()
    record = prompt_record(tokenizer, _example(), "reverse", "qwen3", False)
    char_spans = qwen3_tool_card_char_spans(
        record["prompt"], record["functions"]
    )
    rendered_cards = [record["prompt"][start:end] for start, end in char_spans]
    assert [__import__("json").loads(card)["function"]["name"] for card in rendered_cards] == [
        "beta",
        "alpha",
    ]


def test_qwen3_card_token_spans_cover_each_exact_json_card() -> None:
    tokenizer = _tokenizer()
    record = prompt_record(tokenizer, _example(), "original", "qwen3", False)
    encoded, char_spans, token_spans = tool_card_token_spans(
        tokenizer, record["prompt"], record["functions"], "qwen3"
    )
    assert encoded["input_ids"].shape[0] == 1
    assert len(char_spans) == len(token_spans) == 2
    assert token_spans[0][1] <= token_spans[1][0]


def test_gemma3_stops_without_wu_prompt_protocol() -> None:
    with pytest.raises(ValueError, match="exact Wu et al"):
        render_prompt(object(), _example(), "original", "gemma3", False)
