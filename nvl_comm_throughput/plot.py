#!/usr/bin/env python3
"""Plot measured remote receive bandwidth and utilization from benchmark CSVs."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nvl-comm-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("assets/bandwidth.png"))
    parser.add_argument("--x-axis", choices=("bytes", "ctas"), default="bytes")
    parser.add_argument("--sizes", type=int, nargs="+", help="filter OUTPUT bytes per rank")
    args = parser.parse_args()
    rows = []
    for path in args.csv:
        with path.open() as f:
            rows.extend(r for r in csv.DictReader(f) if r["correct"] == "True")
    if args.sizes:
        rows = [r for r in rows if int(r["output_bytes"]) in args.sizes]
    if args.x_axis == "ctas":
        rows = [r for r in rows if int(r["ctas"]) > 0]
    else:
        # Keep the message-size comparison readable; the CTA plot shows the full sweep.
        rows = [r for r in rows if not r["backend"].startswith("tma")
                or int(r["ctas"]) in (16, 32)]
    if not rows:
        parser.error("no validated measurements")
    panels = sorted({(int(r["world"]), r["pattern"]) for r in rows})
    fig, axes = plt.subplots(len(panels), 2, figsize=(13, 4 * len(panels)), squeeze=False)
    for axrow, (world, pattern) in zip(axes, panels):
        groups = defaultdict(list)
        for row in rows:
            if int(row["world"]) == world and row["pattern"] == pattern:
                label = row["backend"]
                if args.x_axis == "ctas":
                    label += f' / {int(row["output_bytes"]) / 2**20:g} MiB output/rank'
                elif int(row["ctas"]):
                    label += f' ({row["ctas"]} CTAs)'
                    if row.get("warps_per_cta"):
                        label += f' {row["warps_per_cta"]} warps/CTA'
                label += f' / {row["mode"]}'
                groups[label].append(row)
        for label, values in sorted(groups.items()):
            x_key = "ctas" if args.x_axis == "ctas" else "output_bytes"
            values.sort(key=lambda r: int(r[x_key]))
            x = [int(r[x_key]) for r in values]
            axrow[0].plot(x, [float(r["remote_GBps"]) for r in values], ".-", label=label)
            valid = [r for r in values if r["rx_util_pct"]]
            if valid:
                axrow[1].plot([int(r[x_key]) for r in valid],
                              [float(r["rx_util_pct"]) for r in valid], ".-", label=label)
        for ax in axrow:
            ax.set_xscale("log", base=2)
            ax.set_xlabel("CTAs (14 independent TMA warps each)" if args.x_axis == "ctas"
                          else "Output bytes per rank (B)")
            ax.set_title(f"{world} GPUs / {pattern}")
            ax.grid(True, alpha=0.25)
            ax.set_ylim(bottom=0)
        axrow[0].set_ylabel("Effective remote receive GB/s per GPU")
        axrow[0].legend(fontsize=8)
        axrow[1].set_ylabel("Receive utilization vs specified one-way peak (%)")
    fig.suptitle("NVLink collective throughput — CUDA event timing includes two barriers")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    svg = args.output.with_suffix(".svg")
    fig.savefig(svg)
    # Matplotlib emits trailing spaces in SVG paths; keep committed figures clean.
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
