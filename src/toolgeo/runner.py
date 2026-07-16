from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import evaluate
from .behavior import matrices
from .data import load_normalized, validate
from .features import cosine, mock_features
from .io import write_json, write_jsonl
from .mock import generate
from .schema import record

def run(config: dict[str, Any]) -> dict[str, Any]:
    run_config, data_config = config["run"], config["data"]
    out = Path(run_config["output_dir"]); seed = int(run_config["seed"])
    if data_config["source"] == "mock":
        tools, decisions, traces, latent = generate(seed, int(data_config["n_tools"]), int(data_config["n_decisions"]), int(data_config["n_traces"]))
        features = mock_features(tools, latent, int(config["features"]["dimension"]), seed)
    else:
        tools, decisions, traces = load_normalized(data_config["path"])
        errors = validate(tools, decisions, traces)
        if errors: raise ValueError("Invalid normalized data:\n" + "\n".join(errors))
        feature_path = config["features"].get("path")
        if not feature_path:
            raise ValueError("Real runs require features.path. First run `toolgeo extract-hf`, then set its NPZ path here.")
        loaded = np.load(feature_path)
        expected, actual = [item.tool_id for item in tools], list(loaded["tool_ids"])
        if expected != actual:
            raise ValueError("Feature tool_ids do not exactly match normalized tools.jsonl order.")
        features = {name: loaded[name] for name in ("internal", "description", "schema", "unembedding")}
        behaviour_path = data_config.get("behavior_path")
        if behaviour_path:
            behaviour_tools, decisions, traces = load_normalized(behaviour_path)
            if [item.tool_id for item in behaviour_tools] != [item.tool_id for item in tools]:
                raise ValueError("Behaviour rollout tools do not match representation tools.")
    write_jsonl(out / "normalized" / "tools.jsonl", (record(value) for value in tools))
    write_jsonl(out / "normalized" / "decisions.jsonl", (record(value) for value in decisions))
    write_jsonl(out / "normalized" / "traces.jsonl", (record(value) for value in traces))
    np.savez_compressed(out / "features.npz", tool_ids=np.array([item.tool_id for item in tools]), **features)
    behavior = matrices(tools, decisions, traces)
    np.savez_compressed(out / "behavior.npz", tool_ids=np.array([item.tool_id for item in tools]), **behavior)
    report = evaluate({key: cosine(value) for key, value in features.items()}, behavior, float(config["analysis"]["heldout_fraction"]), seed)
    result = {"run_id": run_config["id"], "n_tools": len(tools), "n_decisions": len(decisions), "n_traces": len(traces), "heldout_pairwise_spearman": report}
    write_json(out / "report.json", result); write_json(out / "config.resolved.json", config)
    return result
