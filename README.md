# LLM Tool Geometry

Paper 1 asks whether a model has a stable internal *tool space* whose geometry
predicts its own tool-selection confusions beyond name, description, schema,
menu position, and a modern semantic-embedding baseline.

The identification design deliberately separates the two measurements:

- geometry comes from standalone tool cards under three controlled wordings;
- behavior comes from queries with an in-context candidate menu rendered by
  the model's native tool-aware chat template.

This prevents prompt-local information from being used to predict itself. The
main analysis is a conditional logit over each query's actual risk set, not a
pairwise probe or correlation.

## Install

```bash
cd /users/zliu328/llm_tool
python -m pip install -e '.[hf,analysis,baselines,probe,datasets]'
```

Checkpoints and caches are kept outside the repository at
`/oscar/scratch/zliu328/llm_tool_ckpt`.

## Paper 1 launch

Import and validate the three Paper-1 datasets once, then submit any complete
pipeline. BFCL and ToolHop downloads are pinned to official revisions.

```bash
PYTHONPATH=src python -m toolgeo import-bfcl \
  --output data/raw/bfcl_v4_live_multiple
PYTHONPATH=src python -m toolgeo import-seal-tools \
  --split train --output data/raw/seal_tools_train
PYTHONPATH=src python -m toolgeo import-toolhop \
  --output data/raw/toolhop

PYTHONPATH=src python -m toolgeo validate-data --input data/raw/bfcl_v4_live_multiple
PYTHONPATH=src python -m toolgeo validate-data --input data/raw/seal_tools_train
PYTHONPATH=src python -m toolgeo validate-data --input data/raw/toolhop

scripts/launch_paper1.sh --slurm configs/paper1_sealtools_train_qwen35_9b.yaml
```

ToolHop is expanded using its 3,912 official sub-tasks: each sub-task is one
single-choice decision over that query's provided tool set, while the 995 full
chains remain in `traces.jsonl` and `trajectories.jsonl`. Official Python
sources are retained in `executables.jsonl` as untrusted data and are never
executed by the importer. BFCL and Seal call payloads remain in
`gold_calls.jsonl`.

The job performs layer-sharded all-layer/multi-view card extraction, Qwen3-Embedding
baselines, native tool-call rollouts with deterministic menu shuffles,
conditional-logit analysis, optional outcome probes, and an integrity
manifest. `SHA-256` in the manifest is only a reproducibility checksum; it is
not encryption and does not alter model/data artifacts.

See [docs/paper1_pipeline.md](docs/paper1_pipeline.md) for the estimand,
controls, split protocol, and artifact shapes.

## Experiment coverage

- Confirmatory matrix: Qwen3.5-9B, Qwen3.5-9B-Base, and Gemma-3-4B-IT across
  BFCL v4, Seal-Tools train, and ToolHop. `scripts/launch_matrix.sh --print`
  shows all nine jobs; `scripts/launch_matrix.sh --submit` submits them as a
  dependency chain so shared cross-model geometry files cannot race.
- `scripts/launch_paper1.sh --slurm CONFIG` submits one full pipeline;
  `scripts/launch_paper1.sh CONFIG` runs that full chain in the current shell;
  `--analysis-only` is reserved for already-complete artifacts. The chain is
  validate -> card-geometry extraction -> semantic baselines -> missing
  cross-model control geometry -> native rollout -> opaque-name control
  rollout -> choice-model analysis -> outcome probe -> audit manifest.
- Cross-model specificity: every config declares `features.control_paths`
  (name, model_id, path); the launcher extracts any missing control geometry
  on the identical tool inventory.
- Cross-dataset transfer matrix: `scripts/run_transfers.sh --print` previews
  and `scripts/run_transfers.sh` runs the
  six ordered Qwen3.5-9B transfers. Passing any three same-model configs runs
  the corresponding Base or Gemma matrix.
- Opaque-name contract: `rollout-hf --opaque-names` writes `opaque_alias.json`
  next to its decisions; the analysis refuses to re-derive aliases and reads
  that file as the single source of truth.

Gemma weights are gated. Accept Google's Hugging Face license and authenticate
on the compute node before submitting the matrix. The pipeline does not
replace a missing/unsupported native Gemma tool template with a Qwen prompt;
it fails explicitly, protecting the behavioral estimand.

Before Gemma access is ready, a real-data non-confirmatory Seal pilot can run
without cross-model controls:

```bash
scripts/launch_paper1.sh --slurm configs/pilot_sealtools_train_qwen35_9b.yaml
```

The older Qwen3-8B configs remain lightweight pilot/compatibility runs; they
are not the preregistered 3x3 matrix.
