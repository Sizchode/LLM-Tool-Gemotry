from __future__ import annotations

import torch

from toolgeo.model import decoder_decision_states


def test_state_uses_final_attended_prompt_token_at_every_decoder_layer() -> None:
    layer_zero = torch.tensor(
        [[[10.0], [11.0], [12.0], [13.0]], [[20.0], [21.0], [22.0], [23.0]]]
    ) + 100
    layer_one = layer_zero + 100
    attention_mask = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]])
    result = decoder_decision_states((layer_zero, layer_one), attention_mask)
    assert result.shape == (2, 2, 1)
    assert torch.equal(result[:, :, 0], torch.tensor([[113.0, 213.0], [123.0, 223.0]]))


def test_returned_state_count_must_equal_decoder_layer_count() -> None:
    decoder = torch.tensor([[[3.0], [4.0]]])
    result = decoder_decision_states((decoder,), torch.tensor([[1, 1]]), 1)
    assert torch.equal(result, torch.tensor([[[4.0]]]))
