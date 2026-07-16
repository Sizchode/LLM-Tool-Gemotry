from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .data import load_normalized, validate
from .audit import manifest
from .hf_extract import extract
from .probe import outcome_probe
from .probe_extract import extract_decision_contexts
from .runner import run
from .seal_tools import export as export_seal
from .rollout_hf import rollout

def config(path: str) -> dict:
    with open(path, encoding="utf-8") as handle: return yaml.safe_load(handle)

def main() -> None:
    parser = argparse.ArgumentParser(prog="toolgeo")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("run", help="run the Paper 1 geometry→behavior experiment")
    command.add_argument("--config", required=True)
    command = commands.add_parser("validate-data", help="validate normalized JSONL data")
    command.add_argument("--input", required=True)
    command = commands.add_parser("extract-hf", help="extract real residual/unembedding features")
    command.add_argument("--input", required=True, help="normalized data directory")
    command.add_argument("--model-id", required=True)
    command.add_argument("--cache-dir", default="/oscar/scratch/zliu328/llm_tool_ckpt/hf")
    command.add_argument("--layer", type=int, default=18)
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
    command = commands.add_parser("rollout-hf", help="produce model-selected tools for Paper 1 behaviour")
    command.add_argument("--input", required=True)
    command.add_argument("--model-id", required=True)
    command.add_argument("--cache-dir", default="/oscar/scratch/zliu328/llm_tool_ckpt/hf")
    command.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "run":
        result = run(config(args.config)); print(f"completed {result['run_id']}: {result['n_tools']} tools; report={config(args.config)['run']['output_dir']}/report.json")
    elif args.command == "validate-data":
        errors = validate(*load_normalized(args.input))
        if errors: raise SystemExit("\n".join(errors))
        print("valid normalized dataset")
    elif args.command == "extract-hf":
        tools, _, _ = load_normalized(args.input)
        errors = validate(tools, [], [])
        if errors: raise SystemExit("\n".join(errors))
        output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
        extract(tools, args.model_id, args.cache_dir, args.layer, output)
        print(f"wrote {output}")
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
    else:
        rollout(args.input, args.model_id, args.cache_dir, args.output)
        print(f"wrote model behaviour to {args.output}")

if __name__ == "__main__": main()
