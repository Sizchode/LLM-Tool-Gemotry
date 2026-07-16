"""Real text, schema, and name baselines for geometry comparisons."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .schema import Tool


def schema_text(tool: Tool) -> str:
    return json.dumps(tool.schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def extract_semantic_embeddings(
    tools: list[Tool], model_id: str, cache_dir: str, output: Path, batch_size: int = 16,
) -> None:
    """Extract a modern, frozen embedding baseline with SentenceTransformers."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install toolgeo[baselines] to extract semantic baselines.") from exc
    model = SentenceTransformer(model_id, cache_folder=cache_dir)
    cards = [
        f"Name: {tool.name}\nDescription: {tool.description or '(empty)'}\nSchema: {schema_text(tool)}"
        for tool in tools
    ]
    card_embedding = model.encode(
        cards, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, tool_ids=np.array([tool.tool_id for tool in tools]),
        model_id=np.array(model_id), card_embedding=card_embedding,
    )
