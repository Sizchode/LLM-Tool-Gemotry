"""Tool-identity-disjoint linear probes over decision-context residuals."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_normalized
from .io import write_json


def _assignment(value: str, heldout_fraction: float) -> bool:
    # Stable across processes/machines; Python's hash() is intentionally not.
    bucket = int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big") / 2**64
    return bucket < heldout_fraction


def _auc(y: np.ndarray, score: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # Average tied ranks, matching the conventional AUC definition.
    for value in np.unique(score):
        tied = np.flatnonzero(score == value)
        if len(tied) > 1:
            ranks[tied] = ranks[tied].mean()
    positive = y.astype(bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _probe_one_layer(x: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray, seed: int) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, log_loss
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    classifier = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=seed, class_weight="balanced"))
    classifier.fit(x[train], y[train])
    probability = classifier.predict_proba(x[test])[:, 1]
    prediction = (probability >= 0.5).astype(np.int8)
    return {
        "test_auroc": _auc(y[test], probability),
        "test_balanced_accuracy": float(balanced_accuracy_score(y[test], prediction)),
        "test_log_loss": float(log_loss(y[test], probability, labels=[0, 1])),
    }


def outcome_probe(behavior_dir: str, contexts_path: str, output: str, heldout_fraction: float = 0.25, seed: int = 17) -> dict[str, Any]:
    """Fit a linear outcome probe without sharing gold-tool identities.

    Target: whether the rollout selected the benchmark gold tool.  Split unit:
    ``gold_tool_id``.  This prevents examples of a target tool appearing on both
    sides of the split, while retaining realistic candidate menus in context.
    """
    try:
        import sklearn  # noqa: F401 -- verify optional dependency once, before long extraction jobs.
    except ImportError as exc:
        raise RuntimeError("Install toolgeo[probe] to use probe-outcome.") from exc
    _, decisions, _ = load_normalized(behavior_dir)
    archive = np.load(contexts_path)
    lookup = {str(key): index for index, key in enumerate(archive["decision_ids"].tolist())}
    observed = [item for item in decisions if item.gold_tool_id and item.chosen_tool_id and item.decision_id in lookup]
    if len(observed) != len(decisions):
        raise ValueError("Decision-context IDs do not exactly cover behavior decisions with choices.")
    x = np.stack([archive["residuals"][lookup[item.decision_id]] for item in observed]).astype(np.float32)
    if x.ndim == 2:  # Backward-compatible single-layer artifacts.
        x = x[:, None, :]
        layers = [int(archive["layer"].item())]
    elif x.ndim == 3:
        layers = [int(value) for value in archive["layers"].tolist()]
    else:
        raise ValueError("contexts residuals must have shape [decision, dimension] or [decision, layer, dimension]")
    y = np.array([item.chosen_tool_id == item.gold_tool_id for item in observed], dtype=np.int8)
    test = np.array([_assignment(str(item.gold_tool_id), heldout_fraction) for item in observed])
    train = ~test
    common = {
        "status": "ok", "probe": "linear_logistic_outcome_from_decision_context_residual",
        "target": "rollout_selected_gold_tool", "split_unit": "gold_tool_id",
        "heldout_fraction_requested": heldout_fraction, "seed": seed,
        "model_id": str(archive["model_id"].item()), "layers": layers,
        "representation_dimension": int(x.shape[2]), "n_decisions": int(len(observed)),
        "n_train": int(train.sum()), "n_test": int(test.sum()),
        "n_train_gold_tools": len({item.gold_tool_id for item, flag in zip(observed, train) if flag}),
        "n_test_gold_tools": len({item.gold_tool_id for item, flag in zip(observed, test) if flag}),
        "train_prevalence": float(y[train].mean()) if train.any() else None,
        "test_prevalence": float(y[test].mean()) if test.any() else None,
    }
    if not train.any() or not test.any() or len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
        common.update(status="unavailable_insufficient_class_variation")
        write_json(Path(output), common)
        return common
    common.update(
        majority_baseline_accuracy=float(max(y[test].mean(), 1 - y[test].mean())),
        prompt_length_mean=float(archive["prompt_lengths"].mean()),
        prompt_length_max=int(archive["prompt_lengths"].max()),
        # All layers are reported on the same held-out split.  The report does
        # not declare a post-hoc "winner"; a selected layer needs a separate
        # validation split or multiple-comparison correction.
        layerwise={str(layer): _probe_one_layer(x[:, index, :], y, train, test, seed) for index, layer in enumerate(layers)},
        layer_selection_note="All-layer descriptive curve; no layer is selected as a confirmatory winner.",
    )
    write_json(Path(output), common)
    return common
