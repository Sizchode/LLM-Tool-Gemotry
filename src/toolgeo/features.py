from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np

from .schema import Tool

def _hash_vector(text: str, dimension: int) -> np.ndarray:
    values = np.empty(dimension, dtype=np.float32)
    for index in range(dimension):
        digest = hashlib.blake2b(f"{text}|{index}".encode(), digest_size=8).digest()
        values[index] = int.from_bytes(digest, "little") / 2**64 * 2 - 1
    return values

def _normalise(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)

def mock_features(tools: list[Tool], latent: np.ndarray, dimension: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    projection = rng.normal(size=(latent.shape[1], dimension))
    internal = _normalise(latent @ projection + rng.normal(scale=.12, size=(len(tools), dimension)))
    description = _normalise(np.stack([_hash_vector(tool.description, dimension) for tool in tools]))
    schema = _normalise(np.stack([_hash_vector(str(sorted(tool.schema.items())), dimension) for tool in tools]))
    unembedding = _normalise(np.stack([_hash_vector(tool.name, dimension) for tool in tools]))
    return {"internal": internal, "description": description, "schema": schema, "unembedding": unembedding}

def cosine(matrix: np.ndarray) -> np.ndarray:
    matrix = _normalise(matrix)
    return matrix @ matrix.T
