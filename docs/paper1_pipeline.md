# Paper 1 pipeline

The unit of analysis is an ordered tool pair. The pipeline deliberately keeps
predictors and behavioral targets separate:

1. `tools.jsonl` supplies four predictors: internal residual geometry,
   unembedding geometry, description similarity, and schema similarity.
2. `decisions.jsonl` yields empirical confusion and candidate-set
   substitutability; `traces.jsonl` yields co-occurrence and directed order.
3. Evaluation samples held-out ordered tool pairs once per run seed and reports
   Spearman correlations independently for every target and predictor.

For a real model/dataset combination:

```bash
# one-time: create normalized tables described in docs/data_contract.md
PYTHONPATH=src $PY -m toolgeo validate-data --input data/raw/bfcl_v4_live_multiple

# install the optional model backend, then extract features
PYTHONPATH=src $PY -m toolgeo extract-hf \
  --input data/raw/bfcl_v4_live_multiple --model-id Qwen/Qwen3-8B \
  --layer 18 --output outputs/paper1_bfcl_v4_qwen3_8b/extracted_features.npz

# set features.path to that file in the config, then score behavior prediction
PYTHONPATH=src $PY -m toolgeo run --config configs/paper1_real_template.yaml
```

Run the same config template separately for BFCL, Seal-Tools, and AppWorld.
The experiment never pools pairs across source datasets: each source is an
independent replication, preserving the paper's generalization claim.

## Decision-context outcome probe

The geometry correlations above are not a probing result.  The optional probe
is a separate measurement: it extracts the final residual at the exact
candidate-selection prompt, then fits a regularized linear logistic readout for
whether the model rollout selected the benchmark gold tool.  Its train/test
partition is deterministic and disjoint in `gold_tool_id`; examples targeting a
test tool never train that probe.  Candidate menus remain in the prompt, which
is explicitly recorded as a limitation rather than silently called a fully
tool-disjoint context split.

```bash
PYTHONPATH=src $PY -m toolgeo extract-decision-hf \
  --input data/raw/seal_tools_train --model-id Qwen/Qwen3-8B --layers all \
  --output outputs/paper1_sealtools_qwen3_8b_train/decision_contexts_all_layers.npz

PYTHONPATH=src $PY -m toolgeo probe-outcome \
  --behavior outputs/paper1_sealtools_qwen3_8b_train/model_behavior \
  --contexts outputs/paper1_sealtools_qwen3_8b_train/decision_contexts_all_layers.npz \
  --output outputs/paper1_sealtools_qwen3_8b_train/probe_outcome.json
```

When the `probe` block is present in a config, `paper1_geometry.sbatch` performs
both commands after its behavior and geometry report steps.  The JSON report
states the representation layer, model, class prevalences, split counts, and
held-out metrics for audit.
