"""Scoring of saved generations: math-verify for GSM8K/MATH500, pass@1 for code.

    python -m eval.scoring <generations.json>

atomically scores a file written by ``python -m eval.eval --defer_scoring``.
"""

import argparse
import hashlib
import json
import os
import re
import socket
import time
from importlib.metadata import version
from pathlib import Path

from common.math_protocol import protocol_metadata
from common.math_protocol import score_answer
from eval.protocols import DATASET_SIZES
from eval.protocols import FOUR_SHOT_PROTOCOLS
from eval.protocols import PROTOCOLS
from eval.protocols import digest

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
SUPPORTED_DATASETS = {"gsm8k", "math", "humaneval", "mbpp"}


def _normalize_eval_gold(gold):
    """Normalize dataset-only LaTeX typography unsupported by math-verify."""
    # Remove only an unambiguous thousands separator; general commas may
    # separate tuple/list elements and must remain intact.
    normalized = re.sub(r"(?<=\d),(?:\\!)?(?=\d{3}(?:\D|$))", "", str(gold))
    # TeX's delimiter-free shorthand such as ``\frac \pi4``.
    return re.sub(
        r"\\frac\s+(\\[A-Za-z]+|\d+)\s*(\d)",
        r"\\frac{\1}{\2}",
        normalized,
    )


def score_results(results, protocol, dataset):
    """Add math-verify correctness to every generation and summary metrics."""
    rows = results["generations"]
    if not rows:
        raise ValueError("Cannot score an empty evaluation")
    scored = [
        {
            **row,
            "correct_math_verify": score_answer(
                row["generations"], _normalize_eval_gold(row["ground_truth"])
            ),
        }
        for row in rows
    ]
    n = len(scored)
    scoring_protocol = {**protocol_metadata(None), "name": protocol}
    if protocol in FOUR_SHOT_PROTOCOLS:
        scoring_protocol["benchmark_suite"] = {dataset: DATASET_SIZES[dataset]}
    else:
        scoring_protocol["benchmark_suite"] = {
            "gsm8k": DATASET_SIZES["gsm8k"], "math": DATASET_SIZES["math"],
            "humaneval": DATASET_SIZES["humaneval"], "mbpp_sanitized_test": DATASET_SIZES["mbpp"],
        }
    output = {
        **results,
        "generations": scored,
        "math_scoring": {
            "name": "math-verify",
            "shared_protocol": scoring_protocol,
            "version": version("math-verify"),
            "extraction": "last explicit answer, otherwise default prediction; latex gold; no_fallback",
            "correct": sum(r["correct_math_verify"] for r in scored),
            "total": n,
            "accuracy": sum(r["correct_math_verify"] for r in scored) / n,
            "mean_nfe": sum(r["steps"] for r in scored) / n,
            "prompt_sha256": digest([r["prompt_input"] for r in scored]),
            "question_gold_sha256": digest([(r["question"], r["ground_truth"]) for r in scored]),
        },
    }
    return output


def evaluate_code(generations, dataset_name):
    """pass@1 with Hugging Face ``code_eval``; annotates each generation."""
    import evaluate as hf_evaluate

    print(f"\n=== Running code evaluation for {dataset_name} ===")
    code_eval = hf_evaluate.load("code_eval")
    predictions = [[gen["generation_sanitized"]] for gen in generations]
    references = [gen["test_cases"] for gen in generations]
    pass_at_k, results = code_eval.compute(
        references=references, predictions=predictions, k=[1], num_workers=1
    )
    for task_id, task_results in results.items():
        if task_results:
            _, result_dict = task_results[0]
            generations[task_id]["pass@1"] = 1.0 if result_dict["passed"] else 0.0
        else:
            generations[task_id]["pass@1"] = 0.0
    print(f"  pass@1: {pass_at_k['pass@1']:.4f}")
    return {"pass@1": pass_at_k["pass@1"]}


def score_payload(payload: dict) -> dict:
    status = payload.get("scoring_status", {})
    if status.get("state") != "deferred":
        raise ValueError("generation file is not awaiting deferred scoring")
    dataset = payload["evaluation_arguments"]["dataset"]
    protocol = payload["eval_protocol"]
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported dataset: {dataset!r}")
    if protocol not in PROTOCOLS:
        raise ValueError(f"unsupported protocol: {protocol!r}")
    if not payload.get("generations"):
        raise ValueError("generation file has no generations")

    started = time.perf_counter()
    if dataset in {"gsm8k", "math"}:
        payload = score_results(payload, protocol=protocol, dataset=dataset)
    else:
        payload["code_eval_results"] = evaluate_code(payload["generations"], dataset)
    payload["scoring_status"] = {
        "state": "complete",
        "mode": "deferred_cpu",
        "dataset": dataset,
        "seconds": time.perf_counter() - started,
        "hostname": socket.gethostname(),
    }
    return payload


def score_file(path: Path) -> dict:
    original = path.read_bytes()
    scored = score_payload(json.loads(original))
    scored["scoring_status"]["unscored_sha256"] = hashlib.sha256(original).hexdigest()
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.scored.tmp")
    try:
        temporary.write_text(json.dumps(scored, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    result = score_file(args.source)
    print(json.dumps(result["scoring_status"], indent=2))


if __name__ == "__main__":
    main()
