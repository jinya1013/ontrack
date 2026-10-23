"""Import the released Learn2PD LLaDA filter into the local HF wrapper."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from common.models.learn2pd import Learn2PDPolicy
from common.models.policy import PolicyHFWrapper


OFFICIAL_SHA256 = "0d375569970208524fd14ce0a71ed3d1c9b51a237c8c397f6601a8ff7adbab5f"
OFFICIAL_URL = (
    "https://raw.githubusercontent.com/ims-kdks/"
    "Learning-to-Parallel-Decoding/main/llada/layer_2_flan.pth"
)


def main(source: Path, output: Path) -> None:
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != OFFICIAL_SHA256:
        raise RuntimeError(f"official checkpoint hash mismatch: {digest}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    # This is the official project checkpoint downloaded from OFFICIAL_URL.
    state = torch.load(source, map_location="cpu", weights_only=True)
    model = Learn2PDPolicy(block_length=32)
    model.load_state_dict(state, strict=True)
    output.mkdir(parents=True)
    PolicyHFWrapper(model, "learn2pd").save_pretrained(
        output, safe_serialization=True
    )
    (output / "release_provenance.json").write_text(
        json.dumps(
            {
                "complete": True,
                "source_url": OFFICIAL_URL,
                "source_sha256": digest,
                "architecture": "32-ReLU-32 Learn2PD MLP",
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "block_length": 32,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.output)
