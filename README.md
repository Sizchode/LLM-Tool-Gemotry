# LLM Tool Geometry

This repository studies two questions. First, does the same tool retain a
recognizable representation when its query, neighboring tools, and menu
position change? Second, when a model chooses the wrong tool, is that tool
locally close to the gold tool in the model's own representation space?

The main experiments use three tool-capable instruction-tuned models:

- `Qwen/Qwen3.5-9B`
- `Qwen/Qwen3.5-4B`
- `google/gemma-3-4b-it`

They run on BFCL v4 `live_multiple`, Seal-Tools train, and ToolHop. There are
no base-model runs in Paper 1.

## What one run does

For every single-tool decision, the model reads the query and candidate tool
cards. Each card is rendered as its name, description, and JSON schema. The
code records the exact character span while constructing the card, maps that
span to tokens with tokenizer offsets, and mean-pools it at every residual
layer. The same context is also rendered with the menu order reversed. No
layer or pooling is selected.

The model's choice is scored with its native tool-call format. Analysis then
reports the layerwise same-tool retrieval accuracy, within-minus-between
cosine gap, nearest-neighbor graph and schema enrichment, and whether the
selected wrong tool is closer to the gold tool than the unselected
distractors. A frozen full-card Qwen embedding and schema Jaccard are the two
external comparisons. No conditional logit, outcome probe, transfer model,
SAE, tuned lens, Jacobian lens, or audit stage is part of this paper pipeline.

## Install

```bash
cd /users/zliu328/llm_tool
/users/zliu328/.venv/bin/python -m pip install -e '.[hf,analysis,baselines,datasets]'
```

Model weights and residual arrays are written under
`/oscar/scratch/zliu328/llm_tool_ckpt`.

## Run

One experiment:

```bash
cd /users/zliu328/llm_tool
scripts/launch_paper1.sh --slurm configs/paper1_bfcl_qwen35_4b.yaml
```

Print or submit the three-model by three-dataset matrix:

```bash
scripts/launch_matrix.sh --print
scripts/launch_matrix.sh --submit
```

The Slurm entry point is `scripts/paper1_geometry.sbatch`. The component
ablation is a separate Seal-Tools × Qwen3.5-4B job and is not repeated across
the full matrix:

```bash
sbatch scripts/paper1_component_ablation.sbatch
```

See `docs/paper1_pipeline.md` for the measurement definitions.
