from __future__ import annotations

import argparse
from pathlib import Path

from .baselines import extract_semantic_embeddings
from .bfcl import export as export_bfcl
from .context_geometry import measure
from .component_ablation import ablate
from .data import load_normalized, validate
from .seal_tools import export as export_seal
from .simple_analysis import analyze
from .toolhop import export as export_toolhop


def main() -> None:
    parser = argparse.ArgumentParser(prog="toolgeo")
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("validate-data", help="validate normalized JSONL data")
    command.add_argument("--input", required=True)

    command = commands.add_parser("measure-hf", help="measure contextual tool geometry and native choices")
    command.add_argument("--input", required=True)
    command.add_argument("--model-id", required=True)
    command.add_argument("--cache-dir", default="/oscar/scratch/zliu328/llm_tool_ckpt/hf")
    command.add_argument("--max-branch-batch", type=int, default=8)
    command.add_argument("--output", required=True)

    command = commands.add_parser("extract-baselines-hf", help="extract frozen full-card semantic embeddings")
    command.add_argument("--input", required=True)
    command.add_argument("--model-id", default="Qwen/Qwen3-Embedding-0.6B")
    command.add_argument("--cache-dir", default="/oscar/scratch/zliu328/llm_tool_ckpt/hf")
    command.add_argument("--batch-size", type=int, default=16)
    command.add_argument("--output", required=True)

    command = commands.add_parser("analyze", help="compute direct stability, neighbor, and error-locality results")
    command.add_argument("--measurements", required=True)
    command.add_argument("--semantic-baseline", required=True)
    command.add_argument("--output", required=True)

    command = commands.add_parser("ablate-hf", help="measure exact name/description/schema card ablations")
    command.add_argument("--input", required=True)
    command.add_argument("--model-id", required=True)
    command.add_argument("--cache-dir", default="/oscar/scratch/zliu328/llm_tool_ckpt/hf")
    command.add_argument("--output", required=True)

    command = commands.add_parser("import-seal-tools")
    command.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    command.add_argument("--output", required=True)

    command = commands.add_parser("import-bfcl")
    command.add_argument("--output", required=True)
    command.add_argument("--input-file", type=Path)
    command.add_argument("--answer-file", type=Path)

    command = commands.add_parser("import-toolhop")
    command.add_argument("--output", required=True)
    command.add_argument("--input-file", type=Path)

    args = parser.parse_args()
    if args.command == "validate-data":
        loaded = load_normalized(args.input)
        errors = validate(*loaded)
        if errors:
            raise SystemExit("\n".join(errors))
        print(f"valid normalized dataset: {len(loaded[0])} tools, {len(loaded[1])} decisions, {len(loaded[2])} traces")
    elif args.command == "measure-hf":
        measure(
            args.input, args.model_id, args.cache_dir, args.output,
            max_branch_batch=args.max_branch_batch,
        )
        print(f"wrote contextual geometry and choices to {args.output}")
    elif args.command == "extract-baselines-hf":
        tools, decisions, traces = load_normalized(args.input)
        errors = validate(tools, decisions, traces)
        if errors:
            raise SystemExit("\n".join(errors))
        extract_semantic_embeddings(tools, args.model_id, args.cache_dir, Path(args.output), args.batch_size)
        print(f"wrote {args.output}")
    elif args.command == "analyze":
        result = analyze(args.measurements, args.semantic_baseline, args.output)
        print(f"wrote {args.output}: {result['n_decisions']} decisions")
    elif args.command == "ablate-hf":
        result = ablate(
            args.input, args.model_id, args.cache_dir, args.output,
        )
        print(f"wrote {args.output}: {result['n_decisions']} decisions")
    elif args.command == "import-seal-tools":
        tools, decisions = export_seal(args.split, Path(args.output))
        print(f"exported Seal-Tools: {tools} tools, {decisions} decisions")
    elif args.command == "import-bfcl":
        tools, decisions = export_bfcl(Path(args.output), args.input_file, args.answer_file)
        print(f"exported BFCL v4 live_multiple: {tools} tools, {decisions} decisions")
    else:
        tools, decisions, traces = export_toolhop(Path(args.output), args.input_file)
        print(f"exported ToolHop: {tools} tools, {decisions} decisions, {traces} traces")


if __name__ == "__main__":
    main()
