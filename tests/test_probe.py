import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from toolgeo.audit import manifest
from toolgeo.probe import outcome_probe


def test_outcome_probe_gold_tool_disjoint(tmp_path: Path):
    root = tmp_path / "behaviour"; root.mkdir()
    (root / "tools.jsonl").write_text('{"tool_id":"a","name":"a","description":"","schema":{},"source":"x"}\n')
    rows = []
    for index in range(20):
        gold = f"tool-{index % 8}"
        rows.append({"decision_id": str(index), "query": "q", "candidate_tool_ids": [gold], "gold_tool_id": gold,
                     "chosen_tool_id": gold if index % 3 else "other", "source": "x"})
    (root / "decisions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (root / "traces.jsonl").write_text("")
    contexts = tmp_path / "contexts.npz"
    np.savez_compressed(contexts, decision_ids=np.array([str(i) for i in range(20)]), residuals=np.random.default_rng(1).normal(size=(20, 6)),
                        prompt_lengths=np.ones(20), layer=np.array(1), model_id=np.array("unit-test"))
    report = outcome_probe(str(root), str(contexts), str(tmp_path / "report.json"), heldout_fraction=0.4)
    assert report["split_unit"] == "gold_tool_id"
    assert report["n_train_gold_tools"] + report["n_test_gold_tools"] == 8


def test_manifest_rejects_missing_complete_artifacts(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text("run: {id: x, output_dir: 'out'}\ndata: {path: raw}\nfeatures: {path: feat.npz}\n")
    try:
        manifest(str(config), str(tmp_path / "audit.json"))
    except FileNotFoundError as exc:
        assert "source_tools" in str(exc)
    else:
        raise AssertionError("Incomplete run must not receive an audit manifest")


def test_cli_exposes_single_layer_features_and_all_layer_probe():
    result = subprocess.run([sys.executable, "-m", "toolgeo", "extract-decision-hf", "--help"], capture_output=True, text=True, check=True)
    assert "--layers" in result.stdout
    result = subprocess.run([sys.executable, "-m", "toolgeo", "extract-hf", "--help"], capture_output=True, text=True, check=True)
    assert "--layer" in result.stdout
