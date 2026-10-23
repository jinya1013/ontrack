# Adapted from https://github.com/dllm-reasoning/d1 (Apache 2.0)
import os
import random
import re
from pathlib import Path

import torch
from datasets import load_dataset
from datasets import load_from_disk

from common.few_shot_prompt import render_few_shot_prompt
from common.math_protocol import render_prompt
from common.parsing.answers import extract_gsm8k_gold

# Pinned local copies (HumanEval, MBPP) saved with ``save_to_disk``; the
# Hugging Face Hub is used for datasets not found here.
DATASETS_PATH = Path(os.environ.get("DLLM_DATASETS_DIR", Path(__file__).resolve().parents[1] / "datasets"))
FOUR_SHOT_GSM8K = "four-shot-tps-gsm8k-v1"


def load_gsm8k(split):
    local_path = DATASETS_PATH / "gsm8k"
    if local_path.exists():
        return load_from_disk(str(local_path))[split]
    return load_dataset("openai/gsm8k", "main")[split]


class GSM8KDataset(torch.utils.data.Dataset):
    """GSM8K test set with the common-math-v1 prompt (zero-shot or four-shot)."""

    def __init__(self, tokenizer, num_examples=0, prompt_protocol=None, fewshot_seed=1234):
        self.tokenizer = tokenizer
        self.prompt_protocol = prompt_protocol
        self.fewshot_seed = fewshot_seed
        if (num_examples == 4) != (prompt_protocol is not None):
            raise ValueError("use zero-shot prompts, or four shots with a four-shot protocol")
        self.actual_num_examples = num_examples
        self.dataset = self.load_test_dataset()
        self.protocol_examples = self.load_four_shot_examples() if prompt_protocol else None
        print(f"evaluating {len(self.dataset)} examples")

    def __len__(self):
        return len(self.dataset)

    def load_test_dataset(self):
        return load_gsm8k("test")

    def load_four_shot_examples(self):
        """Four fixed train solutions shared by every question.

        Calculator annotations ``<<...>>`` are removed and the ``#### N`` line
        becomes ``\\boxed{N}``, the answer format of the common-math-v1 prompt.
        """
        if self.prompt_protocol != FOUR_SHOT_GSM8K:
            raise ValueError(f"Unknown prompt protocol: {self.prompt_protocol}")
        train_data = load_gsm8k("train")
        indices = random.Random(self.fewshot_seed).sample(range(len(train_data)), 4)
        examples = []
        for index in indices:
            reasoning, final = train_data[index]["answer"].split("####")
            reasoning = re.sub(r"<<[^>]*>>", "", reasoning).strip()
            final = final.strip().replace(",", "")
            examples.append(
                (train_data[index]["question"], f"{reasoning}\nThe final answer is \\boxed{{{final}}}.")
            )
        return examples

    def create_prompt(self, question):
        if self.protocol_examples is not None:
            return render_few_shot_prompt(self.tokenizer, question, self.protocol_examples)
        return render_prompt(self.tokenizer, question)

    def __getitem__(self, idx):
        item = self.dataset[int(idx)]
        question = item["question"]
        answer = extract_gsm8k_gold(item["answer"])
        return self.create_prompt(question), question, answer

    def collate_fn(self, batch):
        prompts = [item[0] for item in batch]
        encoded = self.tokenizer(
            prompts, padding_side="left", return_tensors="pt", padding="longest"
        )
        return {
            "input_ids": encoded.input_ids,
            "questions": [item[1] for item in batch],
            "answers": [item[2] for item in batch],
            "prompts": prompts,
            "attention_mask": encoded.attention_mask,
        }
