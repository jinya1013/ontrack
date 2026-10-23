"""Accuracy/NFE Pareto frontiers and budget tables from sweep result markers.

    python scripts/pareto.py results/llada_ontrack results/fastdllm results/eb ...

For every dataset this prints (1) the best *measured* accuracy within each NFE
budget (gen_length/8, /4, /2, gen_length) and (2) each method's Pareto
frontier (minimize mean NFE, maximize accuracy).  Nothing is interpolated or
extrapolated.  ``--csv`` additionally writes every measured point.
"""

import argparse
import csv
import json
from pathlib import Path

DATASETS = ("gsm8k", "math", "humaneval", "mbpp")


def read_points(method_dir: Path):
    points = []
    for marker in sorted(method_dir.glob("*/*.json")):
        data = json.loads(marker.read_text())
        if not {"accuracy", "mean_nfe", "dataset"} <= set(data):
            continue
        knob = next((k for k in ("threshold", "gamma", "steps") if k in data), None)
        points.append({
            "method": method_dir.name, "dataset": data["dataset"], "knob": knob,
            "value": data.get(knob), "accuracy": float(data["accuracy"]),
            "mean_nfe": float(data["mean_nfe"]), "complete": bool(data.get("complete", True)),
        })
    return points


def best_within(points, budget):
    inside = [p for p in points if p["mean_nfe"] <= budget + 1e-9]
    return max(inside, key=lambda p: (p["accuracy"], -p["mean_nfe"])) if inside else None


def pareto_frontier(points):
    frontier, best = [], -1.0
    for point in sorted(points, key=lambda p: (p["mean_nfe"], -p["accuracy"])):
        if point["accuracy"] > best:
            frontier.append(point)
            best = point["accuracy"]
    return frontier


def label(point):
    return f"{100 * point['accuracy']:.2f}% ({point['mean_nfe']:.1f}; {point['knob']} {point['value']:g})"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("method_dirs", nargs="+", type=Path)
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    points = [p for directory in args.method_dirs for p in read_points(directory)]
    if args.csv:
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(points[0]))
            writer.writeheader()
            writer.writerows(points)
    methods = [directory.name for directory in args.method_dirs]
    budgets = [args.gen_length // 8, args.gen_length // 4, args.gen_length // 2, args.gen_length]
    for dataset in DATASETS:
        rows = [p for p in points if p["dataset"] == dataset and p["complete"]]
        if not rows:
            continue
        print(f"\n## {dataset}\n")
        print("| Method | points | " + " | ".join(f"NFE <= {b}" for b in budgets) + " |")
        print("|---|---:|" + "---:|" * len(budgets))
        for method in methods:
            measured = [p for p in rows if p["method"] == method]
            if not measured:
                continue
            cells = [best_within(measured, budget) for budget in budgets]
            print(f"| {method} | {len(measured)} | " + " | ".join(
                label(cell) if cell else "--" for cell in cells) + " |")
        print(f"\nPareto frontiers ({dataset}):")
        for method in methods:
            measured = [p for p in rows if p["method"] == method]
            if measured:
                frontier = pareto_frontier(measured)
                print(f"- {method}: " + ", ".join(
                    f"({p['mean_nfe']:.1f}, {100 * p['accuracy']:.2f}%)" for p in frontier))


if __name__ == "__main__":
    main()
