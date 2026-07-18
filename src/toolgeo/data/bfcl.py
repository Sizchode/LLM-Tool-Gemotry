"""Load the official BFCL v3 live_multiple single-gold decisions."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import yaml


_REPOSITORY = "https://raw.githubusercontent.com/ShishirPatil/gorilla"
_DATA_ROOT = "berkeley-function-call-leaderboard/bfcl_eval/data"


@dataclass(frozen=True)
class BFCLExample:
    example_id: str
    messages: tuple[dict[str, Any], ...]
    functions: tuple[dict[str, Any], ...]
    gold_tool: str

    @property
    def candidate_tools(self) -> tuple[str, ...]:
        return tuple(str(function["name"]) for function in self.functions)

    @property
    def exact_query_key(self) -> str:
        return json.dumps(self.messages, ensure_ascii=False, sort_keys=True)


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, path.open("wb") as output:
        output.write(response.read())


def official_paths(directory: Path, revision: str, category: str) -> tuple[Path, Path]:
    data_path = directory / f"BFCL_v3_{category}.json"
    answer_path = directory / "possible_answer" / f"BFCL_v3_{category}.json"
    data_url = f"{_REPOSITORY}/{revision}/{_DATA_ROOT}/{data_path.name}"
    answer_url = (
        f"{_REPOSITORY}/{revision}/{_DATA_ROOT}/possible_answer/{answer_path.name}"
    )
    if not data_path.exists():
        _download(data_url, data_path)
    if not answer_path.exists():
        _download(answer_url, answer_path)
    return data_path, answer_path


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_bfcl(data_path: Path, answer_path: Path) -> list[BFCLExample]:
    questions = _read_json_lines(data_path)
    answers = _read_json_lines(answer_path)
    answer_by_id = {str(row["id"]): row for row in answers}
    if len(answer_by_id) != len(answers):
        raise ValueError("BFCL answer IDs are not unique")

    examples: list[BFCLExample] = []
    seen: set[str] = set()
    for row in questions:
        example_id = str(row["id"])
        if example_id in seen:
            raise ValueError(f"duplicate BFCL example ID: {example_id}")
        seen.add(example_id)
        if example_id not in answer_by_id:
            raise ValueError(f"{example_id}: no official possible answer")

        conversations = row.get("question")
        if not isinstance(conversations, list) or len(conversations) != 1:
            raise ValueError(f"{example_id}: expected one BFCL conversation")
        messages = conversations[0]
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{example_id}: conversation is empty")

        functions = row.get("function")
        if not isinstance(functions, list) or not functions:
            raise ValueError(f"{example_id}: candidate menu is empty")
        names = [str(function["name"]) for function in functions]
        if len(names) != len(set(names)):
            raise ValueError(f"{example_id}: candidate names are not unique")

        ground_truth = answer_by_id[example_id].get("ground_truth")
        if (
            not isinstance(ground_truth, list)
            or len(ground_truth) != 1
            or not isinstance(ground_truth[0], dict)
            or len(ground_truth[0]) != 1
        ):
            raise ValueError(f"{example_id}: expected exactly one gold function call")
        gold_tool = str(next(iter(ground_truth[0])))
        if gold_tool not in names:
            raise ValueError(f"{example_id}: gold tool is absent from candidate menu")

        examples.append(
            BFCLExample(
                example_id=example_id,
                messages=tuple(messages),
                functions=tuple(functions),
                gold_tool=gold_tool,
            )
        )

    if seen != set(answer_by_id):
        raise ValueError("BFCL question and answer ID sets differ")
    return examples


def prototype_coverage(examples: list[BFCLExample]) -> list[dict[str, Any]]:
    counts = Counter(example.gold_tool for example in examples)
    by_query: defaultdict[str, list[BFCLExample]] = defaultdict(list)
    for example in examples:
        by_query[example.exact_query_key].append(example)
    rows: list[dict[str, Any]] = []
    for example in examples:
        excluded_counts = Counter(
            item.gold_tool for item in by_query[example.exact_query_key]
        )
        replication_eligible = counts[example.gold_tool] > 1
        menu_eligible = all(
            counts[name] - excluded_counts[name] > 0
            for name in example.candidate_tools
        )
        rows.append(
            {
                "example_id": example.example_id,
                "gold_tool": example.gold_tool,
                "gold_tool_count": counts[example.gold_tool],
                "menu_size": len(example.candidate_tools),
                "replication_eligible": replication_eligible,
                "menu_extension_eligible": menu_eligible,
            }
        )
    return rows


def write_precheck(path: Path, examples: list[BFCLExample]) -> None:
    rows = prototype_coverage(examples)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_from_config(config: dict[str, Any]) -> list[BFCLExample]:
    data = config["data"]
    directory = Path(data["directory"])
    data_path, answer_path = official_paths(
        directory=directory,
        revision=str(data["revision"]),
        category=str(data["category"]),
    )
    return load_bfcl(data_path, answer_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and validate BFCL v3 data")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    examples = load_from_config(config)
    output = Path(config["run"]["output_dir"]) / "bfcl_precheck.csv"
    write_precheck(output, examples)
    rows = prototype_coverage(examples)
    replication = sum(bool(row["replication_eligible"]) for row in rows)
    extension = sum(bool(row["menu_extension_eligible"]) for row in rows)
    print(
        f"BFCL rows={len(rows)} tools={len({x.gold_tool for x in examples})} "
        f"replication_eligible={replication} menu_extension_eligible={extension}"
    )


if __name__ == "__main__":
    main()
