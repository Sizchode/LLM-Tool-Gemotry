"""Shared Hugging Face loading helpers for text and multimodal decoder models."""
from __future__ import annotations

from typing import Any

_CLASS_NAMES = (
    "AutoModelForCausalLM",
    "AutoModelForImageTextToText",
    "AutoModelForMultimodalLM",
)


def load_generation_model(model_id: str, cache_dir: str, dtype: Any) -> Any:
    """Load a decoder through the architecture-specific AutoModel family.

    Qwen3 is registered as a causal LM, while Qwen3.5 and Gemma 3 checkpoints
    are multimodal conditional-generation models even for text-only inputs.
    Class names are resolved defensively because not every supported
    Transformers installation exposes every AutoModel family. Falling back
    only on architecture/configuration errors keeps OOM and checkpoint
    corruption visible.
    """
    import transformers

    errors: list[str] = []
    available = [
        (name, model_class)
        for name in _CLASS_NAMES
        if (model_class := getattr(transformers, name, None)) is not None
    ]
    if not available:
        raise ValueError(
            f"transformers {transformers.__version__} exposes none of {_CLASS_NAMES}"
        )

    for name, model_class in available:
        try:
            model = model_class.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                dtype=dtype,
            )
        except (ValueError, AttributeError) as exc:
            errors.append(f"{name}: {exc}")
            continue

        if str(model.dtype) != str(dtype):
            raise ValueError(
                f"{name} loaded {model_id} in {model.dtype} instead of requested "
                f"{dtype}; refusing a silent precision/memory change."
            )
        return model

    raise ValueError(
        f"No supported generation AutoModel in transformers "
        f"{transformers.__version__} can load {model_id}: " + " | ".join(errors)
    )


def text_config(model: Any) -> Any:
    config = model.config
    return getattr(config, "text_config", getattr(config, "language_config", config))


def config_int(model: Any, name: str, default: int = 0) -> int:
    nested = text_config(model)
    value = getattr(nested, name, getattr(model.config, name, default))
    return int(value)
