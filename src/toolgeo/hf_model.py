"""Shared Hugging Face loading helpers for text and multimodal decoder models."""
from __future__ import annotations

from typing import Any


def load_generation_model(model_id: str, cache_dir: str, dtype: Any) -> Any:
    """Load a decoder through the architecture-specific AutoModel family.

    Qwen3 is registered as a causal LM, while Qwen3.5 and Gemma 3 checkpoints
    are multimodal conditional-generation models even for text-only inputs.
    Falling back only on architecture/configuration errors keeps OOM and
    checkpoint corruption visible.
    """
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForMultimodalLM

    errors: list[str] = []
    classes = (AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForMultimodalLM)
    for model_class in classes:
        try:
            return model_class.from_pretrained(model_id, cache_dir=cache_dir, dtype=dtype)
        except (ValueError, AttributeError) as exc:
            errors.append(f"{model_class.__name__}: {exc}")
    raise ValueError(f"No supported generation AutoModel can load {model_id}: " + " | ".join(errors))


def text_config(model: Any) -> Any:
    config = model.config
    return getattr(config, "text_config", getattr(config, "language_config", config))


def config_int(model: Any, name: str, default: int = 0) -> int:
    nested = text_config(model)
    value = getattr(nested, name, getattr(model.config, name, default))
    return int(value)

