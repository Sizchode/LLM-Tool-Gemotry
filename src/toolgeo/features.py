from __future__ import annotations

import numpy as np

def _normalise(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)

def cosine(matrix: np.ndarray) -> np.ndarray:
    matrix = _normalise(matrix)
    return matrix @ matrix.T


def geometry_views(archive: np.lib.npyio.NpzFile) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Average card paraphrases and expose every layer × pooling geometry.

    Stability is the mean cosine from each template-specific representation to
    the per-tool template centroid.  It is an Act-I measurement, not an
    optional ablation.
    """
    layers = [int(value) for value in archive["layers"].tolist()]
    poolings = [str(value) for value in archive["pooling_names"].tolist()]
    views: dict[str, np.ndarray] = {}
    stability: dict[str, float] = {}
    if "centroids" in archive:
        centroids = archive["centroids"]
        template_cosine = archive["template_cosine_to_centroid"]
        if centroids.ndim != 4 or template_cosine.shape[0] != centroids.shape[0] or template_cosine.shape[2:] != centroids.shape[1:3]:
            raise ValueError("centroids and template cosine arrays have incompatible shapes")
        for layer_index, layer in enumerate(layers):
            for pooling_index, pooling in enumerate(poolings):
                key = f"internal/layer={layer}/pooling={pooling}"
                # Slices share the one float16 backing array. Analysis
                # materializes and releases one pairwise cosine at a time.
                views[key] = centroids[:, layer_index, pooling_index, :]
                stability[key] = float(np.mean(template_cosine[:, :, layer_index, pooling_index]))
        return views, stability
    residuals = archive["residuals"].astype(np.float32)
    if residuals.ndim != 5:
        raise ValueError("Legacy geometry residuals must be [tool, template, layer, pooling, hidden]")
    for layer_index, layer in enumerate(layers):
        for pooling_index, pooling in enumerate(poolings):
            variants = residuals[:, :, layer_index, pooling_index, :]
            normalized = variants / np.clip(np.linalg.norm(variants, axis=2, keepdims=True), 1e-12, None)
            centroid = _normalise(normalized.mean(1))
            key = f"internal/layer={layer}/pooling={pooling}"
            views[key] = centroid
            stability[key] = float(np.mean(np.sum(normalized * centroid[:, None, :], axis=2)))
    return views, stability
