"""Run native greedy generation on the saved model-specific BFCL prompts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from .model import load_model, load_tokenizer, model_family, text_tokenizer
from .prompts import tokenize_prompts


def strip_thinking(text: str) -> str:
    marker = "</think>"
    if marker not in text:
        return text.strip()
    return text.rsplit(marker, 1)[1].strip()


def strip_thinking_tokens(token_ids: list[int], tokenizer: Any) -> list[int]:
    closing = tokenizer.encode("</think>", add_special_tokens=False)
    if not closing:
        raise ValueError("tokenizer cannot encode the official thinking terminator")
    last_end: int | None = None
    width = len(closing)
    for start in range(len(token_ids) - width + 1):
        if token_ids[start : start + width] == closing:
            last_end = start + width
    return token_ids[last_end:] if last_end is not None else token_ids


def parse_qwen_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    content = strip_thinking(text)
    start_tag = "<tool_call>"
    end_tag = "</tool_call>"
    start = content.find(start_tag)
    if start < 0:
        return None
    end = content.find(end_tag, start + len(start_tag))
    if end < 0:
        return None
    payload = content[start + len(start_tag) : end].strip()
    try:
        call = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(call, dict):
        return None
    if not isinstance(call.get("name"), str):
        return None
    if not isinstance(call.get("arguments"), dict):
        return None
    return str(call["name"]), dict(call["arguments"])


def parse_llama_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    """Parse the JSON object requested by the official Llama 3.1 tool template."""
    try:
        call = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(call, dict) or not isinstance(call.get("name"), str):
        return None
    parameters = call.get("parameters")
    if not isinstance(parameters, dict):
        return None
    return str(call["name"]), dict(parameters)


def _gemma4_cast(value: str) -> Any:
    """Cast a Gemma 4 argument exactly as Google's official parser does."""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return {"true": True, "false": False}.get(
                value.lower(), value.strip("'\"")
            )


def parse_gemma4_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    """Parse the first call using Google's published Gemma 4 grammar."""
    import re

    calls = re.findall(
        r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>", text, re.DOTALL
    )
    if not calls:
        return None
    name, encoded_arguments = calls[0]
    arguments = {
        key: _gemma4_cast((quoted or plain).strip())
        for key, quoted, plain in re.findall(
            r'(\w+):(?:<\|"\|>(.*?)<\|"\|>|([^,}]*))', encoded_arguments
        )
    }
    return name, arguments


def parse_tool_call(text: str, family: str) -> tuple[str, dict[str, Any]] | None:
    if family == "qwen3":
        return parse_qwen_tool_call(text)
    if family == "llama3_1":
        return parse_llama_tool_call(text)
    if family == "gemma4":
        return parse_gemma4_tool_call(text)
    if family == "gemma3":
        raise ValueError("Gemma 3 parsing requires the exact Wu et al. output protocol")
    raise ValueError(f"unsupported model family: {family}")


def classify_tool_call(
    parsed: tuple[str, dict[str, Any]] | None,
    candidates: list[str],
    gold_tool: str,
) -> tuple[str, str, str | None, dict[str, Any] | None]:
    if parsed is None:
        return "invalid", "invalid", None, None
    name, arguments = parsed
    if name not in candidates:
        return "out_of_menu", "out_of_menu", name, arguments
    if name == gold_tool:
        return name, "correct", name, arguments
    return name, "wrong_in_menu", name, arguments


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _batches(rows: list[dict[str, Any]], size: int):
    if size < 1:
        raise ValueError("generation batch_size must be positive")
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def generate(config: dict[str, Any]) -> Path:
    output_dir = Path(config["run"]["output_dir"])
    records = _read_jsonl(output_dir / "examples.jsonl")
    prompt_interface = load_tokenizer(config["model"])
    tokenizer = text_tokenizer(prompt_interface)
    family = model_family(config["model"])
    model = load_model(config["model"])
    device = next(model.parameters()).device
    max_new_tokens = int(config["generation"]["max_new_tokens"])
    if max_new_tokens != 200:
        raise ValueError("BFCL replication requires max_new_tokens=200")

    generations: list[dict[str, Any]] = []
    batch_size = int(config["generation"]["batch_size"])
    for number, batch in enumerate(_batches(records, batch_size), start=1):
        encoded = tokenize_prompts(prompt_interface, [row["prompt"] for row in batch])
        encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_width = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        for index, row in enumerate(batch):
            generated_ids = output_ids[index, prompt_width:].tolist()
            raw_output = tokenizer.decode(generated_ids, skip_special_tokens=False)
            content_ids = (
                strip_thinking_tokens(generated_ids, tokenizer)
                if family == "qwen3"
                else generated_ids
            )
            content = tokenizer.decode(content_ids, skip_special_tokens=True).strip()
            parsed = parse_tool_call(raw_output if family == "gemma4" else content, family)
            action, outcome, parsed_name, arguments = classify_tool_call(
                parsed,
                list(row["candidate_tools"]),
                str(row["gold_tool"]),
            )
            generations.append(
                {
                    "example_id": row["example_id"],
                    "order": row["order"],
                    "gold_tool": row["gold_tool"],
                    "action": action,
                    "outcome": outcome,
                    "parsed_tool": parsed_name,
                    "arguments": arguments,
                    "raw_output": raw_output,
                    "content_after_thinking": content,
                }
            )
        print(f"generation batch={number} examples={len(batch)}", flush=True)

    path = output_dir / "generations.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in generations:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    print(generate(config))


if __name__ == "__main__":
    main()
