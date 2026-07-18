from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from toolgeo.model import ToolCardStateRecorder, decoder_layers


class Config:
    def __init__(self, model_type: str, count: int):
        self.model_type = model_type
        self.num_hidden_layers = count

    def get_text_config(self):
        return self


def test_causal_lm_decoder_path_is_explicit() -> None:
    layers = [object(), object()]
    model = SimpleNamespace(
        config=Config("llama", 2), model=SimpleNamespace(layers=layers)
    )
    assert decoder_layers(model, "llama3_1") is layers


def test_multimodal_decoder_path_is_explicit() -> None:
    layers = [object(), object(), object()]
    model = SimpleNamespace(
        config=Config("gemma4", 3),
        model=SimpleNamespace(language_model=SimpleNamespace(layers=layers)),
    )
    assert decoder_layers(model, "gemma4") is layers


def test_unknown_architecture_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported model family"):
        decoder_layers(SimpleNamespace(), "unknown")


def test_tool_card_recorder_mean_pools_exact_spans_at_every_layer() -> None:
    layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
    model = SimpleNamespace(
        config=Config("qwen3", 2), model=SimpleNamespace(layers=layers)
    )
    recorder = ToolCardStateRecorder(model, "qwen3")
    values = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    recorder.reset([(1, 3), (4, 6)])
    try:
        output = values
        for layer in layers:
            output = layer(output)
        states = recorder.states()
    finally:
        recorder.close()
    expected = torch.stack(
        [values[0, 1:3].mean(dim=0), values[0, 4:6].mean(dim=0)]
    )
    assert states.shape == (2, 2, 4)
    assert torch.equal(states[:, 0], expected)
    assert torch.equal(states[:, 1], expected)
