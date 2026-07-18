"""Fit Anthropic's Jacobian Lens on a declared WikiText or BFCL corpus."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset
from jlens import configure_logging, fit
from jlens.hf import from_hf

from .model import load_model, load_tokenizer, model_family, text_tokenizer


def wikitext_prompts(corpus: dict[str, Any]) -> list[str]:
    """Use Neuronpedia's published contiguous-chunk WikiText construction."""
    stream = load_dataset(
        corpus["dataset"],
        corpus["dataset_config"],
        split=corpus["split"],
        streaming=True,
    )
    prompts: list[str] = []
    buffer = ""
    n_prompts = int(corpus["n_prompts"])
    max_chars = int(corpus["max_chars"])
    min_chars = int(corpus["min_chars"])
    for record in stream:
        text = str(record.get(corpus["text_field"], "")).strip()
        if not text or text.startswith("="):
            continue
        buffer += " " + text
        while len(buffer) > max_chars:
            prompts.append(buffer[:max_chars].strip())
            buffer = buffer[max_chars:]
            if len(prompts) == n_prompts:
                return prompts
    if buffer.strip() and len(buffer.strip()) >= min_chars and len(prompts) < n_prompts:
        prompts.append(buffer.strip())
    if len(prompts) != n_prompts:
        raise ValueError(f"requested {n_prompts} WikiText prompts, obtained {len(prompts)}")
    return prompts


def bfcl_prompts(corpus: dict[str, Any], examples_path: Path) -> list[str]:
    """Use every saved rendered prompt from the declared BFCL menu order."""
    prompts = []
    with examples_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["order"] == corpus["order"]:
                prompts.append(str(row["prompt"]))
    if not prompts:
        raise ValueError(f"no {corpus['order']!r} prompts in {examples_path}")
    if corpus["n_prompts"] != "all":
        raise ValueError("BFCL fit must use n_prompts: all; subsampling is not implemented")
    return prompts


def load_prompts(
    fit_config: dict[str, Any], experiment_config: dict[str, Any]
) -> list[str]:
    corpus = fit_config["corpus"]
    if corpus["kind"] == "wikitext":
        return wikitext_prompts(corpus)
    if corpus["kind"] == "bfcl_rendered":
        output_dir = Path(experiment_config["run"]["output_dir"])
        return bfcl_prompts(corpus, output_dir / "examples.jsonl")
    raise ValueError(f"unsupported corpus kind: {corpus['kind']}")


def fit_lens(
    experiment_config: dict[str, Any],
    fit_config: dict[str, Any],
    output_dir: Path,
) -> Path:
    if model_family(experiment_config["model"]) != "qwen3":
        raise ValueError("this experiment currently fits the Qwen3 family")
    configure_logging()
    prompts = load_prompts(fit_config, experiment_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / "fit_prompts.jsonl"
    with prompt_path.open("w", encoding="utf-8") as handle:
        for index, prompt in enumerate(prompts):
            handle.write(json.dumps({"index": index, "text": prompt}, ensure_ascii=False) + "\n")

    interface = load_tokenizer(experiment_config["model"])
    tokenizer = text_tokenizer(interface)
    model = load_model(experiment_config["model"])
    settings = fit_config["fit"]
    lens_model = from_hf(
        model,
        tokenizer,
        compile=bool(settings["compile"]),
        force_bos=True,
    )
    checkpoint_path = output_dir / "fit_checkpoint.pt"
    lens = fit(
        lens_model,
        prompts,
        dim_batch=int(settings["dim_batch"]),
        max_seq_len=int(settings["max_seq_len"]),
        skip_first=int(settings["skip_first"]),
        checkpoint_path=str(checkpoint_path),
    )
    lens_path = output_dir / "jacobian_lens.pt"
    lens.save(str(lens_path))
    return lens_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--fit-config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    with Path(args.experiment_config).open(encoding="utf-8") as handle:
        experiment_config = yaml.safe_load(handle)
    with Path(args.fit_config).open(encoding="utf-8") as handle:
        fit_config = yaml.safe_load(handle)
    print(fit_lens(experiment_config, fit_config, Path(args.output_dir)))


if __name__ == "__main__":
    main()
