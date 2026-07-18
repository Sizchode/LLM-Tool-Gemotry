# Tool-Specific Decision States on BFCL

This code runs the BFCL-first experiment in
`/users/zliu328/tool_geometry_master_plan.md`. It uses the official BFCL v3
`live_multiple` release: despite the category name, each of its 1,053 records
has one gold function call and a benchmark-provided candidate menu.

The Wu et al. replication and our extensions are kept separate:

- replication: original menu, Wu et al.'s decoder block for each replicated
  architecture, leave-one-out tool means, and global cosine argmax;
- extensions: every decoder layer, prediction restricted to the benchmark
  menu, error-conditioned agreement with native generation, and matched
  original-versus-reverse menus.

Configured model families are:

- `Qwen/Qwen3-4B` and `Qwen/Qwen3-8B`;
- `meta-llama/Llama-3.1-8B-Instruct`, using its official tool template and
  JSON call format;
- `google/gemma-4-E4B-it`, using its official tool template, thinking mode,
  and Google's published function-call parser;
- `google/gemma-3-4b-it`, with model loading and raw decoder extraction
  implemented but BFCL execution deliberately blocked until the exact Wu et
  al. Gemma 3 prompt is recovered. Gemma 3's official Hugging Face template
  does not render a `tools` argument, so substituting a new prompt would not be
  a replication.

Raw decoder blocks are addressed explicitly: Qwen/Llama use
`model.model.layers`; Gemma 3/4 use `model.model.language_model.layers`.

The code does not implement bootstrap, Monte Carlo analysis, random menu
permutations, constrained candidate scoring, or the under-specified τ-bench
classifier.

## Install

```bash
cd /users/zliu328/llm_tool
/users/zliu328/.local/bin/uv pip install --python /users/zliu328/.venv/bin/python -e '.[test]'
```

## Run locally

The three experiment stages use the same rendered prompts saved in
`examples.jsonl`:

```bash
export PYTHONPATH=/users/zliu328/llm_tool/src
CONFIG=/users/zliu328/llm_tool/configs/rq1_bfcl_qwen3_4b.yaml
/users/zliu328/.venv/bin/python -m toolgeo.data.bfcl --config "$CONFIG"
/users/zliu328/.venv/bin/python -m toolgeo.extract_decision_states --config "$CONFIG"
/users/zliu328/.venv/bin/python -m toolgeo.generate_tool_calls --config "$CONFIG"
/users/zliu328/.venv/bin/python -m toolgeo.analyze_bfcl --config "$CONFIG"
/users/zliu328/.venv/bin/python -m toolgeo.analyze_order --config "$CONFIG"
```

Qwen3 and Gemma 4 thinking remain enabled. Generation is greedy with
`max_new_tokens=200`. Each family is parsed according to its checkpoint
protocol; the same saved rendered prompt is used for extraction and
generation.

## Run with Slurm

Print the commands for all runnable configured models:

```bash
scripts/launch_bfcl.sh
```

Submit one model or the runnable matrix:

```bash
scripts/launch_bfcl.sh --submit configs/rq1_bfcl_qwen3_4b.yaml
scripts/launch_bfcl.sh --submit
```

The Llama 3.1 and Gemma checkpoints require Hugging Face access. Verify access
with `hf auth whoami` before submitting them. The default launcher excludes the
blocked Gemma 3 configuration.

Model checkpoints and experiment outputs live under
`/oscar/scratch/zliu328/llm_tool_ckpt`. The primary result files are
`bfcl_replication_results.csv`, `bfcl_results.csv`,
`bfcl_error_conditioning.csv`, `order_results.csv`, and
`order_representation_results.csv`.
