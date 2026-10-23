"""Threshold (or EB-Sampler gamma) sweep of one decoding method on one GPU.

Generation runs sequentially on the GPU.  After each dataset/threshold point
the unscored generations are saved, scoring is launched as a background CPU
process, and the next generation starts immediately.  Once a dataset reaches a
measured mean NFE equal to the generation length (one token per step), higher
thresholds are skipped for that dataset.  After all scorers finish, one result
marker ``<output>/<dataset>/<knob><value>.json`` is written per point.

Examples::

    # learned policy (ours), thresholds 0.5 ... 0.99
    python scripts/sweep.py --output results/llada_ontrack \
        --policy-config configs/llada/ontrack.yaml --policy-path outputs/llada_ontrack/final
    # Fast-dLLM with the prefix KV cache
    python scripts/sweep.py --output results/fastdllm_prefix --fastdllm --kv-cache prefix_cache
    # EB-Sampler (gamma grid, largest gamma = fewest NFE first)
    python scripts/sweep.py --output results/eb --eb-sampler
    # one-by-one reference (a single point)
    python scripts/sweep.py --output results/one_by_one --one-by-one
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.94, 0.96, 0.98, 0.99)
EB_GAMMAS = (10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0)
DATASETS = ("gsm8k", "math", "humaneval", "mbpp")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    method = parser.add_mutually_exclusive_group(required=True)
    method.add_argument("--policy-config", help="Training config of a learned policy.")
    method.add_argument("--fastdllm", action="store_true", help="Fast-dLLM confidence thresholds.")
    method.add_argument("--eb-sampler", action="store_true", help="EB-Sampler entropy budgets.")
    method.add_argument("--one-by-one", action="store_true", help="One token per step reference.")
    parser.add_argument("--policy-path", help="Policy checkpoint directory (with --policy-config).")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--values", nargs="+", type=float, default=None,
                        help="Thresholds (default 0.50-0.99) or EB gammas (default 10-0).")
    parser.add_argument("--protocol", default="zero-shot-tps-math500-v1")
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--kv-cache", choices=["no_cache", "prefix_cache"], default="no_cache")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--truncate-prompt", type=int, default=None,
                        help="Keep only the last N prompt tokens (default: full prompts).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-test", type=int, default=None, help="Debug: first N examples only.")
    args = parser.parse_args()
    if args.policy_config and not args.policy_path:
        parser.error("--policy-config requires --policy-path")
    return args


def method_arguments(args, value):
    if args.policy_config:
        return "threshold", ["--remasking", "policy", "--config", args.policy_config,
                             "--policy_path", args.policy_path,
                             "--policy_accept_threshold", str(value)]
    if args.fastdllm:
        return "threshold", ["--remasking", "fastdllm", "--thres", str(value)]
    if args.eb_sampler:
        return "gamma", ["--remasking", "eb_sampler", "--eb_gamma", str(value)]
    return "steps", ["--remasking", "low_confidence", "--diffusion_steps", str(int(value))]


def generation_file(point_dir: Path) -> Path:
    files = sorted(point_dir.rglob("*_generations.json"))
    if len(files) != 1:
        raise RuntimeError(f"expected one generation file under {point_dir}, found {files}")
    return files[0]


def main():
    args = parse_args()
    if args.one_by_one:
        values = [args.gen_length]
    elif args.values is not None:
        values = args.values
    else:
        values = EB_GAMMAS if args.eb_sampler else THRESHOLDS
    args.output.mkdir(parents=True, exist_ok=True)
    cpu_env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    scorers = []
    for dataset in args.datasets:
        for value in values:
            knob, extra = method_arguments(args, value)
            point_dir = args.output / dataset / f"{knob}{value:g}"
            marker = args.output / dataset / f"{knob}{value:g}.json"
            if marker.exists():
                mean_nfe = json.loads(marker.read_text())["mean_nfe"]
            else:
                if not any(point_dir.rglob("*_generations.json")):
                    command = [
                        sys.executable, "-m", "eval.eval", "--dataset", dataset,
                        "--eval_protocol", args.protocol, "--gen_length", str(args.gen_length),
                        "--kv_cache", args.kv_cache, "--seed", str(args.seed),
                        "--output_dir", str(point_dir), "--defer_scoring", *extra,
                    ]
                    for flag, option in (("--model_path", args.model_path),
                                         ("--model_revision", args.model_revision),
                                         ("--n_test", args.n_test)):
                        if option is not None:
                            command += [flag, str(option)]
                    if args.truncate_prompt is not None:
                        command += ["--truncate_prompt", str(args.truncate_prompt)]
                    print(" ".join(command), flush=True)
                    subprocess.run(command, cwd=REPO, check=True)
                path = generation_file(point_dir)
                payload = json.loads(path.read_text())
                mean_nfe = sum(r["steps"] for r in payload["generations"]) / len(payload["generations"])
                if payload.get("scoring_status", {}).get("state") == "deferred":
                    log = (point_dir / "scoring.log").open("w")
                    scorers.append((subprocess.Popen(
                        [sys.executable, "-m", "eval.scoring", str(path)],
                        cwd=REPO, env=cpu_env, stdout=log, stderr=subprocess.STDOUT,
                    ), dataset, value, knob, point_dir, marker))
                else:
                    scorers.append((None, dataset, value, knob, point_dir, marker))
            print(f"{dataset} {knob}={value:g}: mean NFE {mean_nfe:.1f}", flush=True)
            if mean_nfe >= args.gen_length:
                print(f"{dataset}: one token per step reached; skipping higher values", flush=True)
                break

    failed = False
    for process, dataset, value, knob, point_dir, marker in scorers:
        if process is not None and process.wait() != 0:
            print(f"scoring failed: {point_dir} (see scoring.log)", flush=True)
            failed = True
            continue
        payload = json.loads(generation_file(point_dir).read_text())
        rows = payload["generations"]
        if dataset in {"gsm8k", "math"}:
            accuracy = payload["math_scoring"]["accuracy"]
        else:
            accuracy = payload["code_eval_results"]["pass@1"]
        summary = {
            "method": args.output.name,
            "dataset": dataset,
            knob: value,
            "accuracy": accuracy,
            "mean_nfe": sum(r["steps"] for r in rows) / len(rows),
            "examples": len(rows),
            "complete": not payload["partial_evaluation"],
            "sampler_tokens_per_second": payload["throughput"]["sampler_tokens_per_second"],
            "generations": str(generation_file(point_dir).relative_to(args.output)),
        }
        temporary = marker.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(marker)
        print(f"{dataset} {knob}={value:g}: accuracy {accuracy:.4f} NFE {summary['mean_nfe']:.1f}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
