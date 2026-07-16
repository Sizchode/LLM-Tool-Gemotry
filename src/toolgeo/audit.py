"""Create a compact, machine-auditable manifest for a completed experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .data import load_normalized
from .io import write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entry(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def manifest(config_path: str, output: str) -> dict[str, Any]:
    """Hash the exact artifacts required to reproduce/audit one configured run."""
    import yaml

    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    out = Path(config["run"]["output_dir"])
    raw = Path(config["data"]["path"])
    behavior = Path(config["data"].get("behavior_path", ""))
    entries: dict[str, dict[str, Any]] = {"config": _entry(config_file)}
    required = {
        "source_tools": raw / "tools.jsonl",
        "source_decisions": raw / "decisions.jsonl",
        "source_traces": raw / "traces.jsonl",
        "extracted_features": Path(config["features"]["path"]),
        "model_behavior_decisions": behavior / "decisions.jsonl",
        "main_report": out / "report.json",
        "resolved_config": out / "config.resolved.json",
    }
    if "baselines" in config:
        required["semantic_baselines"] = Path(config["baselines"]["path"])
    if behavior:
        required["model_behavior_scores"] = behavior / "rollout_scores.jsonl"
    opaque_behavior = config["data"].get("opaque_behavior_path")
    if opaque_behavior:
        required["opaque_behavior_decisions"] = Path(opaque_behavior) / "decisions.jsonl"
        required["opaque_behavior_scores"] = Path(opaque_behavior) / "rollout_scores.jsonl"
    probe = config.get("probe", {})
    if probe:
        required["decision_contexts"] = Path(probe["contexts_path"])
        required["probe_report"] = Path(probe["report_path"])
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Cannot create final manifest; missing: " + ", ".join(missing))
    tools, source_decisions, source_traces = load_normalized(raw)
    entries.update({name: _entry(path) for name, path in required.items()})
    _, behavior_decisions, _ = load_normalized(behavior)
    result = {
        "schema_version": 2,
        "run_id": config["run"]["id"],
        "reproducibility_note": "All hashes are SHA-256 of exact input/output bytes.",
        "hash_note": "SHA-256 is an integrity checksum for artifact identity, not encryption.",
        "input_counts": {"tools": len(tools), "decisions": len(source_decisions), "traces": len(source_traces)},
        "behavior_counts": {
            "decisions": len(behavior_decisions),
            "with_model_choice": sum(item.chosen_tool_id is not None for item in behavior_decisions),
            "with_gold": sum(item.gold_tool_id is not None for item in behavior_decisions),
        },
        "artifacts": entries,
        "reports": {
            "geometry": json.loads((out / "report.json").read_text(encoding="utf-8")),
            "probe": json.loads(Path(probe["report_path"]).read_text(encoding="utf-8")) if probe else None,
        },
    }
    write_json(Path(output), result)
    return result
