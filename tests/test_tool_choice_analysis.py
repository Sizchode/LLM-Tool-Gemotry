from toolgeo.analyze_tool_choice import _pair_category


def _generation(outcome: str, tool: str | None):
    return {"outcome": outcome, "parsed_tool": tool}


def test_pair_category_preserves_correctness_direction() -> None:
    assert (
        _pair_category(
            _generation("correct", "gold"),
            _generation("wrong_in_menu", "wrong"),
        )
        == "original_correct_reverse_wrong"
    )
    assert (
        _pair_category(
            _generation("wrong_in_menu", "wrong"),
            _generation("correct", "gold"),
        )
        == "original_wrong_reverse_correct"
    )


def test_pair_category_distinguishes_same_and_different_wrong_tools() -> None:
    assert (
        _pair_category(
            _generation("wrong_in_menu", "a"),
            _generation("wrong_in_menu", "a"),
        )
        == "both_wrong_same_tool"
    )
    assert (
        _pair_category(
            _generation("wrong_in_menu", "a"),
            _generation("wrong_in_menu", "b"),
        )
        == "both_wrong_different_tools"
    )
