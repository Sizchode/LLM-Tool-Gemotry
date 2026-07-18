# BFCL Jacobian-Lens tool-call readout

This repository contains the current BFCL experiment: read the token-level
tool-call decision with the model's own hidden states and a Jacobian Lens. The
same rendered BFCL prompt is used for decision-state extraction and greedy
generation. The companion matched-pair experiment compares the original and
reversed candidate-menu order.

The study has two separate parts:

- **Readout:** at every decoder layer, measure the full-vocabulary rank of the
  token the model actually generated while emitting the JSON tool name.
- **Menu-order trajectories:** for original/reverse prompt pairs, record the
  two branch readouts at every aligned generation position and layer, through
  the first observed tool-name divergence.

The code does not add bootstrap, Monte Carlo sampling, random menu orders,
candidate-score heuristics, multi-token sequence scores, or the unspecified
τ-bench classifier.

## Models and data

The configured Qwen3 checkpoints are:

`Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`,
`Qwen/Qwen3-8B`, and `Qwen/Qwen3-14B`.

The data source is BFCL v3 `live_multiple`. It contains 1,053 records, each
with one gold function call and a benchmark-provided candidate menu. Menu
orders are only `original` and `reverse`.

Generation is greedy with `max_new_tokens=200`. Qwen3 thinking behavior is
controlled by the model configuration; generated thinking text is removed
before parsing the tool call.

## Layout

```text
configs/                         experiment and lens-fit YAML files
src/toolgeo/data/bfcl.py         BFCL loader and prompt records
src/toolgeo/model.py             Hugging Face model/tokenizer loader
src/toolgeo/extract_decision_states.py
                                  all-layer decision-state extraction
src/toolgeo/generate_tool_calls.py
                                  greedy generation and parsing
src/toolgeo/fit_jacobian_lens.py WikiText/BFCL lens fitting
src/toolgeo/read_tool_calls_with_jacobian_lens.py
                                  token-level Jacobian readout
src/toolgeo/read_menu_order_jacobian_trajectories.py
                                  original/reverse aligned trajectories
scripts/launch_qwen3_jacobian_family.sh
scripts/launch_qwen3_menu_order_jacobian.sh
scripts/plot_wikitext_jacobian_results.py
```

## Installation

```bash
cd /users/zliu328/llm_tool
/users/zliu328/.local/bin/uv pip install \
  --python /users/zliu328/.venv/bin/python -e '.[jacobian]'
```

The model and experiment artifacts are stored outside the repository under:

```text
/oscar/scratch/zliu328/llm_tool_ckpt
```

## Generate BFCL inputs and calls

For a single configuration, the data loader, state extractor, and generator
can be launched locally:

```bash
export PYTHONPATH=/users/zliu328/llm_tool/src
CONFIG=/users/zliu328/llm_tool/configs/rq1_bfcl_qwen3_4b_nothink.yaml

/users/zliu328/.venv/bin/python -m toolgeo.data.bfcl \
  --config "$CONFIG"
/users/zliu328/.venv/bin/python -m toolgeo.extract_decision_states \
  --config "$CONFIG"
/users/zliu328/.venv/bin/python -m toolgeo.generate_tool_calls \
  --config "$CONFIG"
```

The generated `examples.jsonl` is the shared rendered-prompt source for all
subsequent readout jobs.

## Jacobian Lens fitting

The fitting code calls the reference `jlens.fit` implementation. WikiText
fitting estimates a general layer-to-logit Jacobian, without using BFCL tool
labels. The local `wikitext_100` condition is configured in
`configs/jacobian_lens_fit_wikitext_100.yaml`; the optional BFCL-conditioned
fit uses every saved original-order BFCL prompt and is configured in
`configs/jacobian_lens_fit_bfcl.yaml`.

For Qwen3-1.7B, 4B, 8B, and 14B, the launcher also reads the corresponding
published WikiText lenses. Qwen3-0.6B uses the local WikiText fit.

Submit the complete Qwen3 fitting/readout family with:

```bash
bash scripts/launch_qwen3_jacobian_family.sh
```

This submits WikiText readouts and BFCL-conditioned fits/readouts with Slurm
dependencies. It does not alter the saved model checkpoints.

## Token-level Jacobian readout

For every parsed BFCL generation, the readout script:

1. verifies an exact tokenizer round trip for the saved generation;
2. locates the tokenizer tokens covering the JSON `name` value;
3. teacher-forces the same prompt and generated prefix;
4. applies `unembed(J_l h_l)` at every fitted source layer; and
5. records the full-vocabulary rank of the actual generated token.

The final-model rank from the same teacher-forced pass is stored as a direct
reference. Layers are all reported; no layer is selected and token ranks are
not combined into a sequence-level score.

Run the readout explicitly when a lens is available:

```bash
CONFIG=/users/zliu328/llm_tool/configs/rq1_bfcl_qwen3_8b_nothink.yaml
LENS=/oscar/scratch/zliu328/llm_tool_ckpt/lenses/qwen3_8b/wikitext_100/jacobian_lens.pt
RESULT_NAME=wikitext_100

sbatch --export=ALL,CONFIG="$CONFIG",LENS="$LENS",RESULT_NAME="$RESULT_NAME" \
  scripts/read_tool_calls_with_jacobian_lens.sbatch
```

The main output files are:

- `jacobian_lens_<condition>_tool_name_tokens.csv` — one row per generated
  name token and source layer;
- `jacobian_lens_<condition>_tool_name_summary.csv` — exact layerwise counts
  and fractions;
- `bfcl_generation_results.csv` and `generations.jsonl` — saved BFCL calls.

## Matched original/reverse trajectories

Submit the menu-order trajectory jobs with:

```bash
bash scripts/launch_qwen3_menu_order_jacobian.sh
```

For each pair, the trajectory contains both contexts, both fixed branch
tokens, every aligned generation position, and every source layer. The pair
file records whether the generated calls stayed identical, diverged at a tool
name, or could not be aligned. No onset threshold is imposed; the complete
layer-by-position record is retained.

Outputs:

- `menu_order_jacobian_<condition>_pairs.csv`;
- `menu_order_jacobian_<condition>_trajectories.csv` or `.csv.gz`.

## Plot completed WikiText results

The plotting script reads the existing scratch artifacts and writes SVG files
to a repository-local directory:

```bash
python scripts/plot_wikitext_jacobian_results.py \
  --output /users/zliu328/llm_tool/figures/wikitext_jacobian
```

It produces the layerwise readout curve, the exact original/reverse change
rate, and the observed first-divergence-position distributions.
