from __future__ import annotations

from pathlib import Path

import numpy as np

def _normalise(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)

def cosine(matrix: np.ndarray) -> np.ndarray:
    matrix = _normalise(matrix)
    return matrix @ matrix.T


def stability_tool_quantiles(archive: np.lib.npyio.NpzFile) -> dict[str, dict[str, float]]:
    """Per-view distribution of per-tool template stability.

    The scalar gate in :func:`geometry_views` averages over tools; this exposes
    the spread so a few highly unstable tools cannot hide behind the mean.
    """
    if "template_cosine_to_centroid" not in archive:
        return {}
    template_cosine = archive["template_cosine_to_centroid"]
    layers = [int(value) for value in archive["layers"].tolist()]
    poolings = [str(value) for value in archive["pooling_names"].tolist()]
    result: dict[str, dict[str, float]] = {}
    for layer_index, layer in enumerate(layers):
        for pooling_index, pooling in enumerate(poolings):
            per_tool = template_cosine[:, :, layer_index, pooling_index].mean(axis=1)
            quantiles = np.quantile(per_tool, [0.05, 0.10, 0.50])
            result[f"internal/layer={layer}/pooling={pooling}"] = {
                "p05": float(quantiles[0]), "p10": float(quantiles[1]), "p50": float(quantiles[2]),
                "mean": float(per_tool.mean()), "min": float(per_tool.min()),
                "n_tools_below_0.8": int((per_tool < 0.8).sum()),
            }
    return result


def stability_for_view(archive: np.lib.npyio.NpzFile, view: str) -> np.ndarray:
    """Return per-tool template stability for one ``layer/pooling`` view."""
    prefix = "internal/layer="
    if not view.startswith(prefix) or "/pooling=" not in view:
        raise ValueError(f"Unrecognized geometry view: {view}")
    layer_text, pooling = view[len(prefix):].split("/pooling=", 1)
    layers = [int(value) for value in archive["layers"].tolist()]
    poolings = [str(value) for value in archive["pooling_names"].tolist()]
    layer, pooling_index = int(layer_text), poolings.index(pooling)
    layer_index = layers.index(layer)
    return archive["template_cosine_to_centroid"][:, :, layer_index, pooling_index].mean(axis=1)


def validate_geometry_artifact(path: str | Path, expected_tool_ids: list[str] | None = None) -> list[str]:
    """Validate metadata and every external layer shard without loading them."""
    artifact = Path(path)
    errors: list[str] = []
    if not artifact.is_file():
        return [f"missing geometry metadata: {artifact}"]
    try:
        with np.load(artifact) as archive:
            required = {"tool_ids", "layers", "pooling_names", "template_cosine_to_centroid"}
            missing = sorted(required - set(archive.files))
            if missing:
                return [f"geometry metadata missing arrays: {missing}"]
            tool_ids = [str(value) for value in archive["tool_ids"].tolist()]
            if expected_tool_ids is not None and tool_ids != expected_tool_ids:
                errors.append("geometry tool_ids do not exactly match normalized data")
            n_layers = len(archive["layers"])
            n_poolings = len(archive["pooling_names"])
            cosine_shape = archive["template_cosine_to_centroid"].shape
            if len(cosine_shape) != 4 or cosine_shape[0] != len(tool_ids) or cosine_shape[2:] != (n_layers, n_poolings):
                errors.append("template cosine array has incompatible dimensions")
            if "centroid_shards" in archive:
                names = [str(value) for value in archive["centroid_shards"].tolist()]
                if len(names) != n_layers:
                    errors.append("centroid shard count does not match layers")
                for name in names:
                    shard_path = artifact.parent / name
                    if not shard_path.is_file():
                        errors.append(f"missing centroid shard: {shard_path}")
                        continue
                    shard = np.load(shard_path, mmap_mode="r")
                    if shard.ndim != 3 or shard.shape[:2] != (len(tool_ids), n_poolings):
                        errors.append(f"invalid centroid shard shape: {shard_path} {shard.shape}")
            elif "centroids" not in archive and "residuals" not in archive:
                errors.append("geometry contains neither shards nor a supported monolithic array")
    except Exception as exc:
        errors.append(f"cannot read geometry artifact {artifact}: {exc}")
    return errors


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
    if "centroid_shards" in archive:
        template_cosine = archive["template_cosine_to_centroid"]
        shard_names = [str(value) for value in archive["centroid_shards"].tolist()]
        if len(shard_names) != len(layers):
            raise ValueError("centroid shard count does not match residual layers")
        archive_path = Path(str(archive.zip.filename))
        for layer_index, (layer, shard_name) in enumerate(zip(layers, shard_names)):
            shard = np.load(archive_path.parent / shard_name, mmap_mode="r")
            if shard.ndim != 3 or shard.shape[1] != len(poolings):
                raise ValueError(f"centroid shard {shard_name} must be [tool,pooling,hidden]")
            for pooling_index, pooling in enumerate(poolings):
                key = f"internal/layer={layer}/pooling={pooling}"
                views[key] = shard[:, pooling_index, :]
                stability[key] = float(np.mean(template_cosine[:, :, layer_index, pooling_index]))
        return views, stability
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
