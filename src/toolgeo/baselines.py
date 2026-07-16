"""Real text, schema, and name baselines for geometry comparisons."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

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
    descriptions = [tool.description or "(no description)" for tool in tools]
    schemas = [schema_text(tool) for tool in tools]
    combined = [f"Tool: {tool.name}\nDescription: {description}\nSchema: {schema}" for tool, description, schema in zip(tools, descriptions, schemas)]
    arrays = {}
    for name, values in (("description_embedding", descriptions), ("schema_embedding", schemas), ("card_embedding", combined)):
        arrays[name] = model.encode(
            values, batch_size=batch_size, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, tool_ids=np.array([tool.tool_id for tool in tools]), model_id=np.array(model_id), **arrays)


def lexical_similarity(values: Iterable[str]) -> np.ndarray:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for TF-IDF baselines") from exc
    corpus = list(values)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=True, norm="l2", sublinear_tf=True)
    matrix = vectorizer.fit_transform(corpus)
    return (matrix @ matrix.T).toarray().astype(np.float32)


def name_form_similarities(tools: list[Tool]) -> tuple[np.ndarray, np.ndarray]:
    try:
        from rapidfuzz import process
        from rapidfuzz.distance import Levenshtein, Prefix
    except ImportError as exc:
        raise RuntimeError("rapidfuzz is required for scalable name-form baselines") from exc
    names = [tool.name.lower() for tool in tools]
    edit = process.cdist(names, names, scorer=Levenshtein.normalized_similarity, dtype=np.float32, workers=-1)
    prefix = process.cdist(names, names, scorer=Prefix.normalized_similarity, dtype=np.float32, workers=-1)
    return np.asarray(edit), np.asarray(prefix)


def _schema_signature(value: object, prefix: str = "") -> set[str]:
    if not isinstance(value, dict):
        return {f"{prefix}:literal:{type(value).__name__}"}
    result: set[str] = set()
    kind = value.get("type", "unknown")
    result.add(f"{prefix}:type:{kind}")
    properties = value.get("properties", {})
    if isinstance(properties, dict):
        required = set(value.get("required", []))
        for name, child in properties.items():
            path = f"{prefix}.{name}" if prefix else name
            result.add(f"{path}:required:{name in required}")
            result.update(_schema_signature(child, path))
    items = value.get("items")
    if items is not None:
        result.update(_schema_signature(items, prefix + "[]"))
    return result


def schema_structure_similarity(tools: list[Tool]) -> np.ndarray:
    signatures = [_schema_signature(tool.schema) for tool in tools]
    try:
        from sklearn.preprocessing import MultiLabelBinarizer
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for structural schema baselines") from exc
    encoded = MultiLabelBinarizer(sparse_output=True).fit_transform(signatures).astype(np.float32)
    intersection = (encoded @ encoded.T).toarray()
    sizes = np.asarray(encoded.sum(axis=1)).ravel()
    union = sizes[:, None] + sizes[None, :] - intersection
    return np.divide(intersection, union, out=np.ones_like(intersection), where=union > 0).astype(np.float32)


def deterministic_baseline_matrices(tools: list[Tool]) -> dict[str, np.ndarray]:
    name_edit, name_prefix = name_form_similarities(tools)
    return {
        "description_char_tfidf": lexical_similarity(tool.description for tool in tools),
        "schema_char_tfidf": lexical_similarity(schema_text(tool) for tool in tools),
        "schema_structure": schema_structure_similarity(tools),
        "name_char_tfidf": lexical_similarity(tool.name for tool in tools),
        "name_edit_similarity": name_edit,
        "name_prefix_overlap": name_prefix,
    }
