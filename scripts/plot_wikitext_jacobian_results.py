#!/usr/bin/env python3
"""Plot the completed WikiText Jacobian-Lens and menu-order results."""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
from collections import Counter
from pathlib import Path
from html import escape


MODELS = ("0.6B", "1.7B", "4B", "8B", "14B")
DIRS = {
    "0.6B": "rq1_bfcl_qwen3_0_6b_nothink",
    "1.7B": "rq1_bfcl_qwen3_1_7b_nothink",
    "4B": "rq1_bfcl_qwen3_4b_nothink",
    "8B": "rq1_bfcl_qwen3_8b_nothink",
    "14B": "rq1_bfcl_qwen3_14b_nothink",
}


def rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="", errors="replace") as handle:
        yield from csv.DictReader(handle)


def first_file(directory: Path, pattern: str, *, prefer: str | None = None) -> Path:
    candidates = sorted(directory.glob(pattern))
    if prefer:
        preferred = [path for path in candidates if prefer in path.name]
        if preferred:
            return preferred[0]
    if not candidates:
        raise FileNotFoundError(f"no {pattern} in {directory}")
    return candidates[0]


def svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family: sans-serif;font-size:12px} .grid{stroke:#ddd} .axis{stroke:#333} </style>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-size="16">{escape(title)}</text>',
    ]


def write_svg(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines + ["</svg>"]), encoding="utf-8")


def plot_readout(artifact_root: Path, output: Path) -> None:
    width, height, left, right, top, bottom = 820, 500, 70, 25, 45, 55
    lines = svg_start(width, height, "WikiText-fitted Jacobian Lens readout")
    x0, x1, y0, y1 = left, width - right, height - bottom, top
    for tick in range(0, 11, 2):
        y = y0 - (y0 - y1) * tick / 10
        lines.append(f'<line class="grid" x1="{x0}" x2="{x1}" y1="{y}" y2="{y}"/>')
        lines.append(f'<text x="{x0-8}" y="{y+4}" text-anchor="end">{tick/10:g}</text>')
    lines.append(f'<line class="axis" x1="{x0}" x2="{x0}" y1="{y1}" y2="{y0}"/>')
    lines.append(f'<line class="axis" x1="{x0}" x2="{x1}" y1="{y0}" y2="{y0}"/>')
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2"]
    for model in MODELS:
        directory = artifact_root / DIRS[model]
        summary = first_file(
            directory,
            "jacobian_lens_*wikitext*_tool_name_summary.csv",
            prefer="published" if model != "0.6B" else "wikitext_100",
        )
        data = list(rows(summary))
        lens = [row for row in data if row["readout"] == "jacobian_lens"]
        final = [row for row in data if row["readout"] == "final_model"]
        xs = [int(row["decoder_layer"]) for row in lens]
        ys = [float(row["target_top1_fraction"]) for row in lens]
        max_layer = max(xs + [int(row["decoder_layer"]) for row in final])
        color = colors[MODELS.index(model)]
        points = []
        for x, y in zip(xs, ys):
            px = x0 + (x1 - x0) * x / max_layer
            py = y0 - (y0 - y1) * y
            points.append(f"{px:.2f},{py:.2f}")
            lines.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2.5" fill="{color}"/>')
        lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
        if final:
            row = final[-1]
            fx = x0 + (x1 - x0) * int(row["decoder_layer"]) / max_layer
            fy = y0 - (y0 - y1) * float(row["target_top1_fraction"])
            lines.append(f'<path d="M{fx-4},{fy-4} L{fx+4},{fy+4} M{fx+4},{fy-4} L{fx-4},{fy+4}" stroke="{color}"/>')
        ly = y1 + 20 + MODELS.index(model) * 18
        lines.append(f'<line x1="{x1-120}" x2="{x1-100}" y1="{ly}" y2="{ly}" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<text x="{x1-95}" y="{ly+4}">{model}</text>')
    lines += [f'<text x="{(x0+x1)/2}" y="{height-10}" text-anchor="middle">Decoder layer</text>',
              f'<text transform="translate(15 {(y0+y1)/2}) rotate(-90)" text-anchor="middle">Top-1 actual tool-name token</text>']
    write_svg(output / "wikitext_jacobian_layerwise_readout.svg", lines)


def pair_rows(directory: Path):
    path = first_file(directory, "menu_order_jacobian*_pairs.csv", prefer="published")
    return list(rows(path))


def plot_order_change(artifact_root: Path, output: Path) -> None:
    rates = []
    labels = []
    for model in MODELS:
        data = pair_rows(artifact_root / DIRS[model])
        counts = Counter(row["status"] for row in data)
        parsed = counts["same_tool_common_name_trajectory"] + counts[
            "different_tools_diverge_at_name"
        ]
        changed = counts["different_tools_diverge_at_name"]
        rates.append(changed / parsed if parsed else float("nan"))
        labels.append(f"{model}\n{changed}/{parsed}")
    width, height, left, right, top, bottom = 820, 420, 70, 25, 45, 65
    lines = svg_start(width, height, "Menu-order matched-pair changes")
    x0, x1, y0, y1 = left, width-right, height-bottom, top
    ymax = max(rates) * 1.25 if rates else 1
    for tick in range(0, 6):
        value = ymax * tick / 5
        y = y0 - (y0-y1) * value / ymax
        lines.append(f'<line class="grid" x1="{x0}" x2="{x1}" y1="{y}" y2="{y}"/>')
        lines.append(f'<text x="{x0-8}" y="{y+4}" text-anchor="end">{value:.1%}</text>')
    bw = (x1-x0) / len(labels) * .55
    for i,(label,rate) in enumerate(zip(labels,rates)):
        cx=x0+(i+.5)*(x1-x0)/len(labels); bh=(y0-y1)*rate/ymax
        lines.append(f'<rect x="{cx-bw/2}" y="{y0-bh}" width="{bw}" height="{bh}" fill="#4c78a8"/>')
        lines.append(f'<text x="{cx}" y="{y0+18}" text-anchor="middle">{escape(label).replace(chr(10), "&#10;")}</text>')
        lines.append(f'<text x="{cx}" y="{y0-bh-5}" text-anchor="middle">{rate:.2%}</text>')
    lines += [f'<line class="axis" x1="{x0}" x2="{x0}" y1="{y1}" y2="{y0}"/>', f'<line class="axis" x1="{x0}" x2="{x1}" y1="{y0}" y2="{y0}"/>']
    write_svg(output / "wikitext_menu_order_change_rate.svg", lines)


def plot_divergence_positions(artifact_root: Path, output: Path) -> None:
    width, height = 820, 900
    lines = svg_start(width, height, "Observed divergence positions in WikiText matched trajectories")
    panel_h = 155
    for i, model in enumerate(MODELS):
        data = pair_rows(artifact_root / DIRS[model])
        positions = [
            int(row["first_name_difference"])
            for row in data
            if row["status"] == "different_tools_diverge_at_name"
            and row["first_name_difference"]
        ]
        counts = Counter(positions)
        xs = sorted(counts)
        ytop=45+i*panel_h; ymax=max(counts.values(), default=1); x0,x1=80,790; y0=ytop+105
        for tick in range(0,4):
            y=y0-90*tick/3; lines.append(f'<line class="grid" x1="{x0}" x2="{x1}" y1="{y}" y2="{y}"/>')
        for x in xs:
            px=x0+(x/max(xs+[1]))*(x1-x0); bh=90*counts[x]/ymax
            lines.append(f'<rect x="{px-4}" y="{y0-bh}" width="8" height="{bh}" fill="#f58518"/>')
        lines.append(f'<text x="35" y="{ytop+55}" text-anchor="middle">{model}</text>')
    lines.append(f'<text x="{width/2}" y="{height-10}" text-anchor="middle">First differing tool-name token position from &lt;tool_call&gt;</text>')
    write_svg(output / "wikitext_menu_order_divergence_positions.svg", lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/oscar/scratch/zliu328/llm_tool_ckpt/artifacts"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/oscar/scratch/zliu328/llm_tool_ckpt/figures/wikitext_jacobian"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plot_readout(args.artifact_root, args.output)
    plot_order_change(args.artifact_root, args.output)
    plot_divergence_positions(args.artifact_root, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
