"""Pinned evaluation protocols.

* ``zero-shot-tps-math500-v1``: GSM8K (1,319), MATH500 (500), HumanEval (164)
  and MBPP-sanitized (257); zero-shot; generation length 256, block 32.
* ``zero-shot-tps-length-transfer-v1``: the same suite at generation length 128 or 512.
* ``four-shot-tps-gsm8k-v1`` / ``four-shot-tps-math500-v1``: the common
  math prompt preceded by four fixed solved turns (no prompt truncation).

All protocols use batch size 1, token temperature 0 and no warmup.
"""

import hashlib
import json
import subprocess
from pathlib import Path

MATH500_TPS_PROTOCOL = "zero-shot-tps-math500-v1"
LENGTH_TRANSFER_PROTOCOL = "zero-shot-tps-length-transfer-v1"
FOUR_SHOT_GSM8K_PROTOCOL = "four-shot-tps-gsm8k-v1"
FOUR_SHOT_MATH500_PROTOCOL = "four-shot-tps-math500-v1"
PROTOCOLS = (
    MATH500_TPS_PROTOCOL, LENGTH_TRANSFER_PROTOCOL,
    FOUR_SHOT_GSM8K_PROTOCOL, FOUR_SHOT_MATH500_PROTOCOL,
)
FOUR_SHOT_PROTOCOLS = {FOUR_SHOT_GSM8K_PROTOCOL, FOUR_SHOT_MATH500_PROTOCOL}
DATASET_SIZES = {"gsm8k": 1319, "math": 500, "humaneval": 164, "mbpp": 257}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def code_provenance():
    """Hash of the evaluation source (and git commit when available)."""
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    files = {}
    for folder in ("eval", "common", "data"):
        for path in sorted((root / folder).rglob("*.py")):
            files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"commit": commit, "source_sha256": digest(files), "files": files}


def configure(args):
    """Validate the decode settings for ``args.eval_protocol`` and set few-shot."""
    protocol = args.eval_protocol
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}")
    if protocol == FOUR_SHOT_GSM8K_PROTOCOL:
        datasets, lengths, shots = {"gsm8k"}, {256}, 4
    elif protocol == FOUR_SHOT_MATH500_PROTOCOL:
        datasets, lengths, shots = {"math"}, {256}, 4
    elif protocol == LENGTH_TRANSFER_PROTOCOL:
        datasets, lengths, shots = set(DATASET_SIZES), {128, 512}, 0
    else:
        datasets, lengths, shots = set(DATASET_SIZES), {256}, 0
    if args.dataset not in datasets:
        raise ValueError(f"{protocol} supports datasets {sorted(datasets)}")
    if args.few_shot not in {-1, shots}:
        raise ValueError(f"{protocol} requires --few_shot {shots}")
    if args.gen_length not in lengths or args.block_length != 32:
        raise ValueError(f"{protocol} requires length in {sorted(lengths)} and block length 32")
    if args.temperature != 0:
        raise ValueError(f"{protocol} requires token temperature 0")
    args.few_shot = shots
