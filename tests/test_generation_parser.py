from __future__ import annotations

from toolgeo.generate_tool_calls import (
    classify_tool_call,
    parse_gemma4_tool_call,
    parse_llama_tool_call,
    parse_qwen_tool_call,
    strip_thinking_tokens,
)


def test_parser_strips_thinking_and_reads_first_structured_call() -> None:
    output = (
        "<think>reasoning</think>\n"
        '<tool_call>{"name":"beta","arguments":{"x":1}}</tool_call>'
    )
    assert parse_qwen_tool_call(output) == ("beta", {"x": 1})


def test_parser_distinguishes_all_behavioral_outcomes() -> None:
    candidates = ["alpha", "beta"]
    assert classify_tool_call(("beta", {}), candidates, "beta")[1] == "correct"
    assert classify_tool_call(("alpha", {}), candidates, "beta")[1] == "wrong_in_menu"
    assert classify_tool_call(("gamma", {}), candidates, "beta")[1] == "out_of_menu"
    assert classify_tool_call(None, candidates, "beta")[1] == "invalid"
    assert parse_qwen_tool_call("natural language only") is None


def test_thinking_tokens_are_removed_before_parsing() -> None:
    class TokenizerFixture:
        def encode(self, text, add_special_tokens=False):
            assert text == "</think>"
            assert add_special_tokens is False
            return [9, 10]

    assert strip_thinking_tokens([1, 2, 9, 10, 3, 4], TokenizerFixture()) == [3, 4]


def test_llama_parser_requires_official_json_shape() -> None:
    assert parse_llama_tool_call(
        '{"name":"beta","parameters":{"x":1}}'
    ) == ("beta", {"x": 1})
    assert parse_llama_tool_call(
        '{"name":"beta","arguments":{"x":1}}'
    ) is None


def test_gemma4_parser_uses_published_control_token_grammar() -> None:
    output = (
        '<|channel>analysis reasoning<channel|>'
        '<|tool_call>call:beta{x:1,label:<|"|>a,b<|"|>}<tool_call|>'
    )
    assert parse_gemma4_tool_call(output) == ("beta", {"x": 1, "label": "a,b"})
