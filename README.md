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

## Jacobian Lens readout of generated tool names

The Qwen3-8B experiment applies the Jacobian Lens readout from
[Verbalizable Representations Form a Global Workspace in Language
Models](https://transformer-circuits.pub/2026/workspace/) using the authors'
[reference implementation](https://github.com/anthropics/jacobian-lens).
It does not compare tool-card vectors or construct a multi-token tool score.

For every BFCL generation with a parsed tool call, the analysis:

1. reconstructs the saved generation token-for-token and requires an exact
   tokenizer round trip;
2. locates the top-level JSON `name` value and its overlapping tokenizer
   tokens;
3. teacher-forces the same saved prompt and generated prefix;
4. at the position before each generated tool-name token, applies
   `unembed(J_l h_l)` for every fitted source layer; and
5. records the zero-based full-vocabulary rank of the token that the model
   actually generated. The same rank from the model's final layer under the
   teacher-forced forward pass is recorded as the direct reference readout.

The saved generation used KV caching, while the reference implementation's
teacher-forced forward pass does not. Their final-layer ranks are therefore
reported rather than required to be identical under bfloat16 arithmetic.

No layer is selected and tool-name token ranks are not combined into a new
sequence score.

The Jacobian Lens fitting corpus is an explicit experimental condition:

- `published_wikitext` uses Neuronpedia's public Qwen3 lenses at revision
  `a4114d7752d11eb546e6cf372213d7e75526d3a1`. Their configs use the
  `Salesforce/wikitext` train split, concatenate non-heading rows into
  2,000-character chunks, truncate to 128 tokens, and exclude the first 16
  positions. Their convergence rule stopped the released 1.7B, 4B, 8B, and
  14B fits after 466, 479, 461, and 615 prompts, respectively.
- `wikitext_100` uses the same published prompt construction and the first
  100 resulting chunks. Anthropic describes approximately 100 prompts as a
  usable fit; this is a project sensitivity analysis, not the paper's
  1,000-prompt protocol.
- `bfcl_original_all` uses every original-order rendered BFCL prompt saved by
  the completed run. It does not sample BFCL examples. This condition tests
  whether a domain-conditioned average Jacobian changes the readout.

Both custom conditions call Anthropic's unmodified `jlens.fit`; `dim_batch`
only batches output dimensions and does not change the estimated Jacobian.
The fit settings are declared in
`configs/jacobian_lens_fit_wikitext_100.yaml` and
`configs/jacobian_lens_fit_bfcl.yaml`.

Submit the complete Qwen3 family comparison after the BFCL runs exist:

```bash
bash scripts/launch_qwen3_jacobian_family.sh
```

## Matched menu-order Jacobian trajectories

`read_menu_order_jacobian_trajectories.py` compares each BFCL decision under
the original and reversed candidate-menu orders. It follows the generated
output from the first `<tool_call>` token through the end of a shared tool
name, or through the first token where two different selected tool names
diverge. Only positions with a common generated prefix are compared, so the
two contexts differ by menu order rather than by earlier sampled output.

At every included generation position and decoder layer, the output records
the rank and log-probability of (a) both branches' actual next tokens and (b)
the fixed future tool token that identifies each branch. For changed calls,
the fixed tokens are the first differing tool-name tokens; for unchanged
calls, both are the shared first tool-name token. This reveals whether the
eventual tool token is readable before it becomes the immediate next token.
The same quantities from the model's final layer are the direct reference. No
probability threshold or selected onset layer is used; the result is the
complete layer-by-position matched trajectory.

Run the Qwen3 size family with:

```bash
bash scripts/launch_qwen3_menu_order_jacobian.sh
```

Each model writes:

- `menu_order_jacobian_<condition>_pairs.csv`, containing every matched
  decision and its exact output-alignment status; and
- `menu_order_jacobian_<condition>_trajectories.csv.gz`, containing the two
  context × two target rank/log-probability readouts at every layer and
  aligned generation position.

- `jacobian_lens_<condition>_tool_name_tokens.csv`: one row per generated
  name token and decoder layer;
- `jacobian_lens_<condition>_tool_name_summary.csv`: exact top-1 counts and
  fractions for every decoder layer.

Install the published reference implementation with the optional dependency:

```bash
/users/zliu328/.local/bin/uv pip install \
  --python /users/zliu328/.venv/bin/python -e '.[jacobian]'
```

Run one completed BFCL generation through a matching lens with:

```bash
CONFIG=/users/zliu328/llm_tool/configs/rq1_bfcl_qwen3_8b_nothink.yaml
LENS=/path/to/qwen3_8b_lens.pt
RESULT_NAME=wikitext_100
sbatch --export=ALL,CONFIG="$CONFIG",LENS="$LENS",RESULT_NAME="$RESULT_NAME" \
  scripts/read_tool_calls_with_jacobian_lens.sbatch
```
