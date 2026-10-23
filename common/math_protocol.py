"""Shared common-math-v1 input and correctness contract (train/eval/calibration)."""

import re
from common.parsing.final_answer import final_answer_text

PROTOCOL = "common-math-v1"
PROMPT = (
    "You are a math expert. Solve the question step by step. "
    "Put your final answer in \\boxed{}."
)


def prompt_messages(question):
    return [{"role": "user", "content": PROMPT + "\n\n" + question}]


def render_prompt(tokenizer, question):
    return tokenizer.apply_chat_template(
        prompt_messages(question), add_generation_prompt=True, tokenize=False
    )


def score_answer(prediction, gold):
    from math_verify import LatexExtractionConfig, parse, verify

    # Gold is a dataset answer, not a model response; wrap bare mathematical text.
    gold_text = str(gold).strip()
    if "\\boxed" not in gold_text and not re.search(r"(?<!\\)\$", gold_text):
        gold_text = "\\boxed{" + gold_text + "}"
    target = parse(gold_text, extraction_config=[LatexExtractionConfig()], fallback_mode="no_fallback")
    if not target:
        raise ValueError(f"Unparseable ground truth: {gold!r}")
    # Evaluate the last explicit answer only; multiple boxes can otherwise be
    # interpreted as a tuple by math-verify. Never rescue an invalid final box
    # with an earlier number from the reasoning.
    prediction = str(prediction)
    final = final_answer_text(prediction)
    if final is not None:
        answer = parse("\\boxed{" + final + "}", extraction_config=[LatexExtractionConfig()], fallback_mode="no_fallback")
    elif "\\boxed" in prediction or "\\fbox" in prediction or "<answer>" in prediction:
        return False
    else:
        answer = parse(prediction, fallback_mode="no_fallback")
    return bool(verify(target, answer))


def protocol_metadata(prompt_truncation=None):
    import hashlib
    from pathlib import Path
    from importlib.metadata import version
    return {"name": PROTOCOL, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "math_verify_version": version("math-verify"), "prompt_truncation": prompt_truncation,
            "final_answer_source_sha256": hashlib.sha256((Path(__file__).parent / "parsing/final_answer.py").read_bytes()).hexdigest()}
