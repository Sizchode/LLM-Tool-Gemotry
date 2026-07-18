"""Load configured checkpoints and record raw decoder-block decision states."""
from __future__ import annotations

from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMultimodalLM,
    AutoProcessor,
    AutoTokenizer,
)


CAUSAL_LM_FAMILIES = {"qwen3", "llama3_1"}
MULTIMODAL_LM_FAMILIES = {"gemma3", "gemma4"}
EXPECTED_MODEL_TYPES = {
    "qwen3": "qwen3",
    "llama3_1": "llama",
    "gemma3": "gemma3",
    "gemma4": "gemma4",
}


def torch_dtype(name: str) -> torch.dtype:
    try:
        value = getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"unsupported torch dtype: {name}") from exc
    if not isinstance(value, torch.dtype):
        raise ValueError(f"unsupported torch dtype: {name}")
    return value


def model_family(model_config: dict[str, Any]) -> str:
    family = str(model_config["family"])
    if family not in EXPECTED_MODEL_TYPES:
        raise ValueError(f"unsupported model family: {family}")
    return family


def load_tokenizer(model_config: dict[str, Any]) -> Any:
    """Load the official prompt interface for the configured checkpoint."""
    family = model_family(model_config)
    loader = AutoProcessor if family in MULTIMODAL_LM_FAMILIES else AutoTokenizer
    interface = loader.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        cache_dir=model_config["cache_dir"],
    )
    if getattr(interface, "chat_template", None) is None:
        tokenizer = getattr(interface, "tokenizer", None)
        if tokenizer is None or tokenizer.chat_template is None:
            raise ValueError("checkpoint has no official chat template")
    return interface


def text_tokenizer(prompt_interface: Any) -> Any:
    """Return the text tokenizer underlying a tokenizer or multimodal processor."""
    return getattr(prompt_interface, "tokenizer", prompt_interface)


def load_model(model_config: dict[str, Any]) -> Any:
    family = model_family(model_config)
    dtype = torch_dtype(str(model_config["dtype"]))
    loader = (
        AutoModelForCausalLM
        if family in CAUSAL_LM_FAMILIES
        else AutoModelForMultimodalLM
    )
    model = loader.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        cache_dir=model_config["cache_dir"],
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    actual_type = str(model.config.model_type)
    expected_type = EXPECTED_MODEL_TYPES[family]
    if actual_type != expected_type:
        raise ValueError(
            f"configured family {family} requires model_type={expected_type}, "
            f"checkpoint reports {actual_type}"
        )
    model.to(str(model_config["device"]))
    model.eval()
    if model.dtype != dtype:
        raise ValueError(f"loaded dtype {model.dtype} does not match requested {dtype}")
    return model


def decoder_layers(model: Any, family: str) -> Any:
    """Resolve the documented text-decoder path for each supported architecture."""
    if family in CAUSAL_LM_FAMILIES:
        layers = model.model.layers
    elif family in MULTIMODAL_LM_FAMILIES:
        layers = model.model.language_model.layers
    else:
        raise ValueError(f"unsupported model family: {family}")
    if len(layers) != int(model.config.get_text_config().num_hidden_layers):
        raise ValueError("decoder path does not contain every configured text layer")
    return layers


def decoder_decision_states(
    layer_outputs: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    expected_layers: int | None = None,
) -> torch.Tensor:
    """Return [batch, decoder_layer, hidden] at the final attended prompt token."""
    decoder_layer_count = len(layer_outputs)
    if decoder_layer_count < 1:
        raise ValueError("model returned no decoder block states")
    if expected_layers is not None and decoder_layer_count != expected_layers:
        raise ValueError(
            f"model returned {decoder_layer_count} hidden-state tensors for "
            f"{expected_layers} decoder layers"
        )

    final_positions = []
    for row in attention_mask:
        attended = row.nonzero(as_tuple=False).flatten()
        if attended.numel() == 0:
            raise ValueError("prompt has no attended token")
        final_positions.append(int(attended[-1]))

    per_layer = []
    for state in layer_outputs:
        selected = torch.stack(
            [state[index, position] for index, position in enumerate(final_positions)]
        )
        per_layer.append(selected)
    return torch.stack(per_layer, dim=1)


class DecisionStateRecorder:
    """Record raw outputs of every text decoder block for one forward pass."""

    def __init__(self, model: Any, family: str) -> None:
        expected_type = EXPECTED_MODEL_TYPES[family]
        if str(model.config.model_type) != expected_type:
            raise ValueError(
                f"recorder family {family} requires model_type={expected_type}"
            )
        layers = decoder_layers(model, family)
        self._outputs: list[torch.Tensor | None] = [None] * len(layers)
        self._handles = [
            layer.register_forward_hook(self._hook(index))
            for index, layer in enumerate(layers)
        ]

    def _hook(self, index: int):
        def record(module: Any, inputs: Any, output: torch.Tensor) -> None:
            if not isinstance(output, torch.Tensor):
                raise ValueError("decoder block did not return a tensor")
            self._outputs[index] = output

        return record

    def reset(self) -> None:
        self._outputs = [None] * len(self._outputs)

    def states(self, attention_mask: torch.Tensor) -> torch.Tensor:
        if any(output is None for output in self._outputs):
            raise ValueError("not every decoder hook recorded an output")
        outputs = tuple(output for output in self._outputs if output is not None)
        return decoder_decision_states(outputs, attention_mask, len(self._outputs))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


class ToolCardStateRecorder:
    """Mean-pool configured token spans at every raw text-decoder block."""

    def __init__(self, model: Any, family: str) -> None:
        layers = decoder_layers(model, family)
        self._spans: list[tuple[int, int]] = []
        self._outputs: list[torch.Tensor | None] = [None] * len(layers)
        self._handles = [
            layer.register_forward_hook(self._hook(index))
            for index, layer in enumerate(layers)
        ]

    def _hook(self, index: int):
        def record(module: Any, inputs: Any, output: torch.Tensor) -> None:
            if not isinstance(output, torch.Tensor) or output.shape[0] != 1:
                raise ValueError("tool-card extraction requires one tensor prompt")
            self._outputs[index] = torch.stack(
                [output[0, start:end].mean(dim=0) for start, end in self._spans]
            ).detach()

        return record

    def reset(self, spans: list[tuple[int, int]]) -> None:
        if not spans or any(start < 0 or end <= start for start, end in spans):
            raise ValueError("tool-card token spans must be non-empty intervals")
        self._spans = list(spans)
        self._outputs = [None] * len(self._outputs)

    def states(self) -> torch.Tensor:
        if any(output is None for output in self._outputs):
            raise ValueError("not every decoder hook recorded tool-card states")
        outputs = tuple(output for output in self._outputs if output is not None)
        return torch.stack(outputs, dim=1)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


# Kept for compatibility with already-running Qwen3 jobs.
class Qwen3DecisionStateRecorder(DecisionStateRecorder):
    def __init__(self, model: Any) -> None:
        super().__init__(model, "qwen3")
