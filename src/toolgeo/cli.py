from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .data import load_normalized, validate
from .audit import manifest
from .hf_extract import extract
from .features import validate_geometry_artifact
from .baselines import extract_semantic_embeddings
from .probe import outcome_probe
from .probe_extract import extract_decision_contexts
from .runner import run, run_transfer
from .seal_tools import export as export_seal
from .bfcl import export as export_bfcl
from .toolhop import export as export_toolhop
from .rollout_hf import render_native_choice_sequences, rollout

def config(path: str) -> dict:
    with open(path, encoding="utf-8") as handle: return yaml.safe_load(handle)

def main() -> None:
    parser = argparse.ArgumentParser(prog="toolgeo")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("run", help="run the Paper 1 geometry→behavior experiment")
    command.add_argument("--config", required=True)
    command = commands.add_parser("validate-data", help="validate normalized JSONL data")
    command.add_argument("--input", required=True)
    command = commands.add_parser("validate-features", help="validate geometry metadata and layer shards")
    command.add_argument("--input", required=True)
    command.add_argument("--data", required=True, help="normalized data directory for exact tool-ID checking")
    command = commands.add_parser("transfer", help="fit on one dataset and evaluate a different dataset")
    command.add_argument("--source-config", required=True)
    command.add_argument("--target-config", required=True)
    command.add_argument("--output", required=True)
    command = commands.add_parser("extract-hf", help="extract real residual/unembedding features")
    command.add_argument("--input", required=True, help="normalized data directory")
    command.add_argument("--model-id", required=True)
    command.add_argument("--cache-dir", default="/oscar/scratch/zliu328/llm_tool_ckpt/hf")
    command.add_argument("--layers", default="all", help="'all' (default) or comma-separated residual indices")
    command.add_argument("--output", required=True)
    command = commands.add_parser("extract-baselines-hf", help="extract strong frozen semantic baselines")
    command.add_argument("--input", required=True)
    command.add_argument("--model-id", default="Qwen/Qwen3-Embedding-0.6B")
    command.add_argument("--cache-dir", default="/oscar/scratch/zliu328/llm_tool_ckpt/hf")
    command.add_argument("--batch-size", type=int, default=16)
    command.add_argument("--output", required=True)
    command = commands.add_parser("extract-decision-hf", help="extract decision-context residuals for probing")
    command.add_argument("--input", required=True)
    command.add_argument("--model-id", required=True)
    command.add_argument("--cache-dir", default="/oscar/scratch/zliu328/llm_tool_ckpt/hf")
    command.add_argument("--layers", default="all", help="'all' (default) or one residual layer index")
    command.add_argument("--output", required=True)
    command = commands.add_parser("probe-outcome", help="run gold-tool-disjoint linear outcome probe")
    command.add_argument("--behavior", required=True)
    command.add_argument("--contexts", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--heldout-fraction", type=float, default=0.25)
    command.add_argument("--seed", type=int, default=17)
    command = commands.add_parser("audit-manifest", help="hash and summarize a completed configured run")
    command.add_argument("--config", required=True)
    command.add_argument("--output", required=True)
    command = commands.add_parser("import-seal-tools", help="export official Seal-Tools into normalized tables")
    command.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    command.add_argument("--output", required=True)
    command.add_argument("--limit", type=int)
    command = commands.add_parser("import-bfcl", help="export official BFCL v4 live_multiple into normalized tables")
    command.add_argument("--output", required=True)
    command.add_argument("--input-file", type=Path, help="optional local official BFCL input JSONL")
    command.add_argument("--answer-file", type=Path, help="optional local official BFCL possible-answer JSONL")
    command = commands.add_parser("import-toolhop", help="export official ToolHop into step-level normalized tables")
    command.add_argument("--output", required=True)
    command.add_argument("--input-file", type=Path, help="optional local official ToolHop JSON")
    command = commands.add_parser("rollout-hf", help="produce model-selected tools for Paper 1 behaviour")
    command.add_argument("--input", required=True)
    command.add_argument("--model-id", required=True)
    command.add_argument("--cache-dir", default="/oscar/scratch/zliu328/llm_tool_ckpt/hf")
    command.add_argument("--output", required=True)
    command.add_argument("--menu-repeats", type=int, default=3, help="original order plus deterministic shuffles")
    command.add_argument("--seed", type=int, default=17)
    command.add_argument("--max-branch-batch", type=int, default=8, help="maximum trie branch contexts per model forward")
    command.add_argument("--opaque-names", action="store_true", help="replace names by uniform opaque aliases in the rendered menu")
    command.add_argument("--include-multi-call", action="store_true", help="roll out multi-call labels (excluded by Paper 1 analysis)")
    command = commands.add_parser("validate-model-template", help="verify native tool-call rendering before loading model weights")
    command.add_argument("--input", required=True)
    command.add_argument("--model-id", required=True)
    command.add_argument("--cache-dir", default="/oscar/scratch/zliu328/llm_tool_ckpt/hf")
    args = parser.parse_args()
    if args.command == "run":
        result = run(config(args.config)); print(f"completed {result['run_id']}: {result['n_tools']} tools; report={config(args.config)['run']['output_dir']}/report.json")
    elif args.command == "validate-data":
        loaded = load_normalized(args.input)
        errors = validate(*loaded)
        if errors: raise SystemExit("\n".join(errors))
        print(f"valid normalized dataset: {len(loaded[0])} tools, {len(loaded[1])} decisions, {len(loaded[2])} traces")
    elif args.command == "validate-features":
        tools, _, _ = load_normalized(args.data)
        errors = validate_geometry_artifact(args.input, [tool.tool_id for tool in tools])
        if errors: raise SystemExit("\n".join(errors))
        print(f"valid geometry artifact: {args.input}")
    elif args.command == "transfer":
        report = run_transfer(config(args.source_config), config(args.target_config), args.output)
        print(f"wrote {args.output}: {report['source_run_id']} -> {report['target_run_id']}")
    elif args.command == "extract-hf":
        tools, decisions, traces = load_normalized(args.input)
        errors = validate(tools, decisions, traces)
        if errors: raise SystemExit("\n".join(errors))
        output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
        extract(tools, args.model_id, args.cache_dir, args.layers, output)
        print(f"wrote {output}")
    elif args.command == "extract-baselines-hf":
        tools, decisions, traces = load_normalized(args.input)
        errors = validate(tools, decisions, traces)
        if errors: raise SystemExit("\n".join(errors))
        extract_semantic_embeddings(tools, args.model_id, args.cache_dir, Path(args.output), args.batch_size)
        print(f"wrote {args.output}")
    elif args.command == "extract-decision-hf":
        extract_decision_contexts(args.input, args.model_id, args.cache_dir, args.layers, args.output)
        print(f"wrote {args.output}")
    elif args.command == "probe-outcome":
        report = outcome_probe(args.behavior, args.contexts, args.output, args.heldout_fraction, args.seed)
        print(f"wrote {args.output}: {report['status']}")
    elif args.command == "audit-manifest":
        report = manifest(args.config, args.output)
        print(f"wrote {args.output}: {report['run_id']}")
    elif args.command == "import-seal-tools":
        tools, decisions = export_seal(args.split, Path(args.output), args.limit)
        print(f"exported Seal-Tools: {tools} tools, {decisions} decisions")
    elif args.command == "import-bfcl":
        tools, decisions = export_bfcl(Path(args.output), args.input_file, args.answer_file)
        print(f"exported BFCL v4 live_multiple: {tools} tools, {decisions} decisions")
    elif args.command == "import-toolhop":
        tools, decisions, traces = export_toolhop(Path(args.output), args.input_file)
        print(f"exported ToolHop: {tools} tools, {decisions} decisions, {traces} traces")
    elif args.command == "validate-model-template":
        from transformers import AutoTokenizer
        tools, decisions, _ = load_normalized(args.input)
        by_id = {tool.tool_id: tool for tool in tools}
        decision = next((item for item in decisions if len(item.candidate_tool_ids) >= 2), None)
        if decision is None:
            raise SystemExit("dataset has no decision with at least two candidates")
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir, use_fast=True)
        candidates = decision.candidate_tool_ids[:2]
        prompt, sequences, thinking_disabled = render_native_choice_sequences(
            tokenizer, decision.query, candidates, by_id,
        )
        print(
            f"valid native tool template: model={args.model_id} prompt_tokens={len(prompt)} "
            f"candidate_lengths={[len(value) for value in sequences]} "
            f"thinking_disabled={thinking_disabled}"
        )
    else:
        rollout(
            args.input, args.model_id, args.cache_dir, args.output,
            menu_repeats=args.menu_repeats, seed=args.seed,
            opaque_names=args.opaque_names, include_multi_call=args.include_multi_call,
            max_branch_batch=args.max_branch_batch,
        )
        print(f"wrote model behaviour to {args.output}")

if __name__ == "__main__": main()
