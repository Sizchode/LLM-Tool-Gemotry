# Paper 1 experiment

Paper 1 combines a small Act I and Act II.

Act I asks whether tools form a recognizable, context-sensitive geometry and
what card components shape it. Act II asks whether errors are local in that
geometry and whether internal geometry follows model behavior more closely
than a frozen text embedding.

## Representation

For a decision with query `q` and candidate menu `T`, each tool is rendered as:

```text
Name: ...
Description: ...
Schema: ...
```

The renderer records each card's character boundaries as it builds the
prompt. A fast tokenizer maps those declared boundaries to tokens through
offset mappings. At every residual layer, the representation is the mean over
that one card span. This is the only pooling rule.

Every eligible benchmark decision is measured. There is no sampled context
set. Menu position is tested with one declared intervention: the exact reverse
of the original menu. Variation in query and menu composition comes from the
benchmark decisions in which a tool occurs.

## Act I

For each layer, RQ1 reports:

- retrieval accuracy when a contextual occurrence is assigned to the nearest
  leave-one-occurrence-out tool centroid;
- mean same-tool cosine, mean different-tool cosine, and their difference;
- the nearest-tool graph and schema enrichment at k=1, 5, and 10;
- mean schema Jaccard among nearest neighbors and its difference from all tool
  pairs;
- neighbor overlap between adjacent layers.

RQ2 uses exactly five cards: full, no name, no description, no schema, and an
opaque positional name. It reports cosine displacement from the full card and
within-menu neighbor overlap. This ablation is run on Seal-Tools with
Qwen3.5-4B, not on all nine model-dataset combinations.

## Act II

The choice is determined in the checkpoint's native tool-call format. At each
branch where candidate calls diverge, logits are normalized over the remaining
candidates. Shared tokens do not contribute to the score.

For every wrong choice and every layer, RQ4 reports:

- cosine between the gold and selected wrong tool;
- selected-wrong cosine minus the mean cosine of unselected distractors;
- the selected tool's neighbor percentile among distractors;
- Hit@1, Hit@5, and Hit@10;
- Spearman correlation between directed confusion rate and contextual cosine
  over exposed gold-candidate pairs.

The same error-locality measurements are computed from one frozen full-card
embedding model and from schema Jaccard. There is no fitted behavior model and
no train/validation/test selection step.

## Scope

The core matrix is:

```text
{Qwen3.5-9B, Qwen3.5-4B, Gemma-3-4B-IT}
    x
{BFCL v4 live_multiple, Seal-Tools train, ToolHop}
```

Paper 1 does not include base models, cross-dataset transfer, outcome probes,
SAEs, tuned/Jacobian lenses, attention analysis, fusion tools, tool injection,
registration failure, commitment failure, or plan inertia. Those are separate
questions rather than robustness checks for this paper.
