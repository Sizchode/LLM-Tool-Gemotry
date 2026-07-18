from __future__ import annotations

import torch

from toolgeo.analyze_tool_cards import (
    _leave_context_out_scores,
    _pairwise_cosine_components,
)


def test_pairwise_cosine_components_are_exact() -> None:
    states = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    labels = torch.tensor([0, 0, 1, 1])
    within_count, within, between_count, between = _pairwise_cosine_components(
        states, labels, 2
    )
    assert within_count == 2
    assert within == 1.0
    assert between_count == 4
    assert between == 0.0


def test_retrieval_excludes_every_prototype_occurrence_from_test_menu() -> None:
    train = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    )
    labels = torch.tensor([0, 1, 0, 1])
    examples = ["a", "a", "b", "b"]
    scores = _leave_context_out_scores(train, train, labels, examples, 2)
    assert scores.argmax(dim=1).tolist() == labels.tolist()
