from __future__ import annotations

import torch

from toolgeo.analyze_bfcl import (
    _global_predictions,
    _menu_predictions,
    cosine_argmax,
    leave_out_prototypes,
)


def test_test_example_does_not_enter_its_prototype() -> None:
    states = torch.tensor([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 4.0]])
    labels = ["a", "a", "b", "b"]
    prototypes = leave_out_prototypes(states, labels, {0})
    assert torch.equal(prototypes["a"], torch.tensor([3.0, 0.0]))
    assert torch.equal(prototypes["b"], torch.tensor([0.0, 3.0]))


def test_cosine_argmax_is_restricted_to_requested_menu() -> None:
    state = torch.tensor([1.0, 0.0])
    prototypes = {
        "a": torch.tensor([0.8, 0.2]),
        "b": torch.tensor([0.0, 1.0]),
        "outside": torch.tensor([1.0, 0.0]),
    }
    assert cosine_argmax(state, prototypes, ["a", "b"]) == "a"


def test_replication_and_menu_extension_match_hand_computation() -> None:
    states = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
    )
    labels = ["a", "a", "b", "b"]
    rows = [
        {
            "gold_tool": label,
            "candidate_tools": ["a", "b"],
            "exact_query_key": f"query-{index}",
        }
        for index, label in enumerate(labels)
    ]
    global_predictions, global_eligible = _global_predictions(states, labels)
    menu_predictions, menu_eligible = _menu_predictions(states, rows)
    assert global_eligible == [True, True, True, True]
    assert menu_eligible == [True, True, True, True]
    assert global_predictions == labels
    assert menu_predictions == labels
