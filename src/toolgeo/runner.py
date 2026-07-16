from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .analysis import evaluate_choice_geometry, evaluate_cross_dataset_transfer
from .baselines import deterministic_baseline_matrices
from .behavior import matrices
from .data import load_normalized, validate
from .features import cosine, geometry_views
from .io import write_json, write_jsonl
from .schema import record


def _load_exact(path: str, tools, label: str) -> np.lib.npyio.NpzFile:
    archive = np.load(path)
    expected = [item.tool_id for item in tools]
    actual = [str(item) for item in archive["tool_ids"].tolist()]
    if expected != actual:
        raise ValueError(f"{label} tool_ids do not exactly match tools.jsonl order")
    return archive


def _semantic_baselines(tools, path: str | None) -> dict[str, np.ndarray]:
    result = deterministic_baseline_matrices(tools)
    if not path:
        raise ValueError(
            "Real runs require baselines.path from `toolgeo extract-baselines-hf`; "
            "the strong semantic baseline is a confirmatory requirement."
        )
    archive = _load_exact(path, tools, "Semantic baseline")
    for name in ("description_embedding", "schema_embedding", "card_embedding"):
        result[name] = cosine(archive[name].astype(np.float32))
    return result


def _analysis_options(config: dict[str, Any], seed: int) -> dict[str, Any]:
    value = config.get("analysis", {})
    return {
        "validation_fraction": float(value.get("validation_fraction", 0.2)),
        "test_fraction": float(value.get("test_fraction", value.get("heldout_fraction", 0.2))),
        "stability_threshold": float(value.get("stability_threshold", 0.8)),
        "bootstrap_samples": int(value.get("bootstrap_samples", 500)),
        "permutation_samples": int(value.get("permutation_samples", 200)),
        "l2": float(value.get("l2", 1e-4)), "seed": seed,
    }


def _real_bundle(config: dict[str, Any]) -> tuple[Any, ...]:
    data_config = config["data"]
    tools, _, _ = load_normalized(data_config["path"])
    behavior_tools, decisions, traces = load_normalized(data_config["behavior_path"])
    if [item.tool_id for item in behavior_tools] != [item.tool_id for item in tools]:
        raise ValueError("Behavior rollout tools do not match representation tools")
    errors = validate(behavior_tools, decisions, traces)
    if errors:
        raise ValueError("Invalid behavior data:\n" + "\n".join(errors))
    archive = _load_exact(config["features"]["path"], tools, "Geometry")
    geometry, stability = geometry_views(archive)
    baselines = _semantic_baselines(tools, config.get("baselines", {}).get("path"))
    directional = {
        "name_unembedding_norm_difference": archive["name_unembedding_norm"].astype(float),
        "name_length_difference": np.array([len(item.name) for item in tools], dtype=float),
    }
    return tools, decisions, traces, geometry, stability, baselines, directional


def run_transfer(source_config: dict[str, Any], target_config: dict[str, Any], output: str) -> dict[str, Any]:
    if source_config["data"].get("source") != "jsonl" or target_config["data"].get("source") != "jsonl":
        raise ValueError("Cross-dataset transfer requires two real JSONL datasets")
    source = _real_bundle(source_config)
    target = _real_bundle(target_config)
    analysis = source_config.get("analysis", {})
    result = evaluate_cross_dataset_transfer(
        source[0], source[1], source[3], source[5], source[4], source[6],
        target[0], target[1], target[3], target[5], target[6],
        validation_fraction=float(analysis.get("validation_fraction", 0.2)),
        stability_threshold=float(analysis.get("stability_threshold", 0.8)),
        bootstrap_samples=int(analysis.get("bootstrap_samples", 500)),
        l2=float(analysis.get("l2", 1e-4)), seed=int(source_config["run"]["seed"]),
    )
    report = {
        "source_run_id": source_config["run"]["id"],
        "target_run_id": target_config["run"]["id"], "transfer": result,
    }
    write_json(Path(output), report)
    return report


def run(config: dict[str, Any]) -> dict[str, Any]:
    run_config, data_config = config["run"], config["data"]
    out = Path(run_config["output_dir"])
    seed = int(run_config["seed"])
    control_specs: list[tuple[str, str]] = []
    opaque_decisions = None
    if data_config.get("source") != "jsonl":
        raise ValueError("Production runs require data.source: jsonl")
    tools, decisions, traces = load_normalized(data_config["path"])
    errors = validate(tools, decisions, traces)
    if errors:
        raise ValueError("Invalid normalized data:\n" + "\n".join(errors))
    behavior_path = data_config.get("behavior_path")
    if not behavior_path:
        raise ValueError("Real runs require data.behavior_path containing model choices")
    behavior_tools, decisions, traces = load_normalized(behavior_path)
    if [item.tool_id for item in behavior_tools] != [item.tool_id for item in tools]:
        raise ValueError("Behavior rollout tools do not match representation tools")
    errors = validate(behavior_tools, decisions, traces)
    if errors:
        raise ValueError("Invalid behavior data:\n" + "\n".join(errors))
    opaque_path = data_config.get("opaque_behavior_path")
    if opaque_path:
        opaque_tools, opaque_decisions, _ = load_normalized(opaque_path)
        if [item.tool_id for item in opaque_tools] != [item.tool_id for item in tools]:
            raise ValueError("Opaque-control rollout tools do not match representation tools")
        errors = validate(opaque_tools, opaque_decisions, [])
        if errors:
            raise ValueError("Invalid opaque-control behavior data:\n" + "\n".join(errors))
    feature_path = config["features"].get("path")
    if not feature_path:
        raise ValueError("Real runs require features.path from `toolgeo extract-hf --layers all`")
    archive = _load_exact(feature_path, tools, "Geometry")
    geometry, stability = geometry_views(archive)
    baselines = _semantic_baselines(tools, config.get("baselines", {}).get("path"))
    directional = {
        "name_unembedding_norm_difference": archive["name_unembedding_norm"].astype(float),
        "name_length_difference": np.array([len(item.name) for item in tools], dtype=float),
    }
    for control in config["features"].get("control_paths", []):
        control_specs.append((str(control["name"]), str(control["path"])))
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "normalized" / "tools.jsonl", (record(value) for value in tools))
    write_jsonl(out / "normalized" / "decisions.jsonl", (record(value) for value in decisions))
    write_jsonl(out / "normalized" / "traces.jsonl", (record(value) for value in traces))
    behavior = matrices(tools, decisions, traces)
    np.savez_compressed(out / "behavior.npz", tool_ids=np.array([item.tool_id for item in tools]), **behavior)
    options = _analysis_options(config, seed)
    report = evaluate_choice_geometry(tools, decisions, geometry, baselines, stability, directional, **options)
    control_reports = {}
    for name, path in control_specs:
        control_archive = _load_exact(path, tools, f"Control geometry {name}")
        control_geometry, control_stability = geometry_views(control_archive)
        control_reports[name] = evaluate_choice_geometry(
            tools, decisions, control_geometry, baselines, control_stability, directional, **options,
        )
        control_archive.close()
    own_delta = report["query_bootstrap"]["delta_nll"]
    specificity = {
        name: {
            "target_model_geometry_delta_nll": own_delta,
            "control_model_geometry_delta_nll": value["query_bootstrap"]["delta_nll"],
            "target_minus_control_delta_nll": own_delta - value["query_bootstrap"]["delta_nll"],
        }
        for name, value in control_reports.items()
    }
    opaque_report = None
    if opaque_decisions is not None:
        opaque_tools = [
            type(tool)(tool.tool_id, f"tool_{index:05d}", tool.description, tool.schema, tool.source)
            for index, tool in enumerate(sorted(tools, key=lambda item: item.tool_id))
        ]
        opaque_by_id = {item.tool_id: item for item in opaque_tools}
        opaque_ordered = [opaque_by_id[item.tool_id] for item in tools]
        opaque_baselines = deterministic_baseline_matrices(opaque_ordered)
        # Semantic description/schema/card embeddings remain valid except the
        # combined card, which contains the original name.
        for name in ("description_embedding", "schema_embedding"):
            if name in baselines:
                opaque_baselines[name] = baselines[name]
        opaque_directional = {"opaque_name_length_difference": np.array([len(item.name) for item in opaque_ordered], dtype=float)}
        opaque_report = evaluate_choice_geometry(
            tools, opaque_decisions, geometry, opaque_baselines, stability, opaque_directional, **options,
        )
    result = {
        "run_id": run_config["id"], "n_tools": len(tools), "n_decisions": len(decisions),
        "n_traces": len(traces), "research_question": "Does stable standalone-card residual geometry predict in-context tool choices beyond surface and schema similarity?",
        "choice_model": report,
        "cross_model_specificity_controls": control_reports,
        "cross_model_specificity_summary": specificity,
        "opaque_name_control": opaque_report,
        "substitutability": {"status": "unavailable_without_paired_counterfactual_menu_replacements"},
    }
    write_json(out / "report.json", result)
    write_json(out / "config.resolved.json", config)
    return result
