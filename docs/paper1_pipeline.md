# Paper 1: geometry predicts tool choice

## Research question and claim boundary

**RQ.** Does a language model form a stable, model-specific internal geometry
over tools that predicts which distractor it selects, beyond similarities
available from the tools' surface text and schemas?

The confirmatory claim is narrower than “we can probe tool identity”:

> Across tool-use datasets, intermediate residual representations induced by
> standalone tool cards predict in-context tool-selection choices beyond
> modern semantic embeddings, lexical/name similarity, schema structure, and
> menu-position/directional priors.

Geometry and behavior come from different prompt distributions. This is the
central identification principle, not an ablation: standalone-card geometry
must be stable across wording before it is allowed into the choice model.

## Measurement protocol

`extract-hf` renders every tool under three controlled card templates and
stores their normalized centroid with shape `[tool, layer, pooling, hidden]`
plus each template's cosine-to-centroid with shape
`[tool, template, layer, pooling]`. Every residual layer is retained.
The five preregistered pooling views are `name`, `description`, `schema`,
`last`, and `mean`; exact spans use fast-tokenizer offset mappings. Cards are
never concatenated with the benchmark inventory and are never truncated.

The template-specific representations are normalized and averaged into a tool
centroid. Mean template-to-centroid cosine is the stability score. Views below
`analysis.stability_threshold` are ineligible. Among eligible views, layer and
pooling are selected on a tool-disjoint validation split; the test split is
read once for the selected view.

`rollout-hf` renders `apply_chat_template(..., tools=...)`. It does not use the
old handwritten `Tool name:` prompt. Candidate names form a token trie; model
probability is normalized only where remaining candidates diverge, while
shared/non-discriminative tokens contribute probability one. Thus a name is
not penalized merely for containing more tokens. The original menu plus
deterministic shuffled orders are retained with explicit positions and raw
scores in `rollout_scores.jsonl`. Prompts exceeding the context limit fail
rather than silently left-truncating tools.

For Qwen3, `enable_thinking=False` makes the controlled choice start at the
documented `<tool_call>` JSON name field instead of conditioning on an absent,
unobserved reasoning trace. The prefix matches the checkpoint's own chat
template rather than a benchmark-specific imitation.

Paper 1 is a single categorical-choice estimand. Multi-call benchmark rows are
recorded by the adapter but skipped by rollout by default; passing
`--include-multi-call` is an explicit non-Paper-1 diagnostic. All-layer card
residuals are written through a disk-backed temporary array before final NPZ
compression, keeping host-memory use bounded for thousands of tools.

BFCL v4 `live_multiple` directly supplies one gold function per ordered
multi-function menu. ToolHop is multi-hop, so its adapter uses each official
`sub_task` as the decision unit and retains the original chain separately; it
does not relabel the top-level query with its first tool. Seal-Tools retains
multi-call rows for provenance but confirmatory choice analysis excludes them.

The opaque-name control replaces rendered function names with uniform aliases
while keeping tool IDs, descriptions, schemas, queries, menus, and geometry
fixed. Its behavior is fit separately and reported under `opaque_name_control`.

## Baselines

Every confirmatory real run requires all of the following:

- Qwen3-Embedding description, schema, and combined-card embeddings;
- character 3–5 gram TF-IDF cosine for description, schema, and name;
- structural schema similarity over parameter paths, types, and requiredness;
- normalized tool-name edit similarity and common-prefix overlap;
- menu position, name length, and output-embedding norm differences for the
  asymmetric/prior component.

Random/hash text vectors do not exist in the pipeline. Production runs use the
documented lexical, structural, and frozen semantic-embedding baselines.

## Statistical estimand

For query/menu `q`, conditional logit models the observed candidate `j`:

`P(j | menu_q, gold_i) ∝ exp(beta · x(i,j,q))`.

The menu is the risk set, so co-occurrence is an exposure denominator rather
than a competing predictor. `is_gold` absorbs the model's correctness base
rate. All similarity values for the gold/self alternative are set to zero;
otherwise cosine(self,self)=1 would trivially reveal the answer. Geometry is
therefore tested on its ability to rank distractors.

Directed confusion rates are reported as `count(gold=i, chosen=j) / exposure(i,j)`
and decomposed into symmetric and antisymmetric components. Symmetric geometry
targets the former. Menu position and candidate-minus-gold priors model the
latter.

Tools—not pairs—are deterministically partitioned into train, validation, and
test. No validation/test tool appears in an earlier split's training menu.
Layer/pooling selection uses validation only; the selected model is refit
without test tools and evaluated once on held-out-gold test queries.
Uncertainty uses query-cluster bootstrap, preserving all shuffled-menu repeats
of one query. The incremental geometry test permutes geometry values only
within a menu while holding its exposure, choice, gold indicator, surface
features, and order fixed.

## Commands and artifacts

```bash
PYTHONPATH=src python -m toolgeo extract-hf \
  --input data/raw/seal_tools_train --model-id Qwen/Qwen3-8B --layers all \
  --output /oscar/scratch/zliu328/llm_tool_ckpt/artifacts/paper1_sealtools_qwen3_8b_train/tool_geometry.npz

PYTHONPATH=src python -m toolgeo extract-baselines-hf \
  --input data/raw/seal_tools_train --model-id Qwen/Qwen3-Embedding-0.6B \
  --output /oscar/scratch/zliu328/llm_tool_ckpt/artifacts/paper1_sealtools_qwen3_8b_train/semantic_baselines.npz

PYTHONPATH=src python -m toolgeo rollout-hf \
  --input data/raw/seal_tools_train --model-id Qwen/Qwen3-8B \
  --menu-repeats 3 \
  --output outputs/paper1_sealtools_qwen3_8b_train/model_behavior

PYTHONPATH=src python -m toolgeo run \
  --config configs/paper1_sealtools_train_qwen3_8b.yaml
```

`report.json` records the validation scan, single selected test result,
incremental negative-log-likelihood, clustered confidence interval,
within-menu permutation p-value, coefficients, tool/decision split counts,
and any cross-model control specified by `features.control_paths`.

Cross-model specificity is run by adding another model's standalone-card NPZ
as a control. The target model's behavior and all surface baselines remain
fixed; the report then asks whether its own geometry predicts its choices
better than the other model's geometry. Cross-dataset transfer fits all choice
coefficients on one configured dataset after source-only view selection and
then evaluates untouched queries from another configured dataset:

```bash
PYTHONPATH=src python -m toolgeo transfer \
  --source-config configs/paper1_sealtools_train_qwen3_8b.yaml \
  --target-config configs/paper1_bfcl_v4_qwen3_8b.yaml \
  --output outputs/transfers/seal_to_bfcl.json
```

Running every ordered config pair produces the proposed fit/test transfer
matrix. This is never approximated by random pair holdout.
