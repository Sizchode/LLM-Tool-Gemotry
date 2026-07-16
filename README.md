# Tool Geometry

Reusable experiment code for three papers. The currently implemented experiment
is **Paper 1: static tool geometry → behavior prediction**. Paper 2 and 3 will
reuse the normalized data contract and artifact format, not Paper 1's results.

## Terminal acceptance path

No model download or GPU is required for this deterministic end-to-end mock:

```bash
cd /users/zliu328/llm_tool
PYTHONPATH=src python -m toolgeo run --config configs/paper1_mock.yaml
```

It creates normalized tools/decisions/traces, feature matrices for all four
predictor families, behavioral targets (confusion, co-occurrence, order, and
substitutability), and a held-out pairwise prediction report in
`outputs/paper1_mock/`.

## Real-data path

Export BFCL v4, Seal-Tools, and AppWorld into the documented JSONL contract,
then set `data.source: jsonl` and `data.path` in a config. Use the optional
`hf` dependency for residual/unembedding extraction. Checkpoints and Hugging
Face cache belong in `/oscar/scratch/zliu328/llm_tool_ckpt`.

```bash
PYTHONPATH=src python -m toolgeo validate-data --input data/raw/bfcl.jsonl
PYTHONPATH=src python -m toolgeo run --config configs/paper1_real_template.yaml
```

The launch script requires a config explicitly, so a real Slurm submission
cannot accidentally execute the mock experiment:

```bash
scripts/launch_paper1.sh --slurm configs/paper1_real_template.yaml
```
