"""Few-shot rendering for the common-math-v1 prompt.

Kept out of ``common.math_protocol`` so that module's source hash, recorded in
contracted G_trace teacher provenance, stays unchanged.
"""

from common.math_protocol import prompt_messages


def render_few_shot_prompt(tokenizer, question, examples):
    """``render_prompt`` preceded by solved (question, solution) chat turns."""
    messages = []
    for example_question, solution in examples:
        messages += prompt_messages(example_question)
        messages.append({"role": "assistant", "content": solution})
    return tokenizer.apply_chat_template(
        messages + prompt_messages(question), add_generation_prompt=True, tokenize=False
    )
