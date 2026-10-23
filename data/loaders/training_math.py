"""GSM8K/MATH training prompts for teacher-trajectory collection."""

import torch
from trl.data_utils import maybe_apply_chat_template

from data.data_utils import get_gsm8k_questions
from data.data_utils import get_math_questions


class TrainingMathDataset(torch.utils.data.Dataset):
    """GSM8K or MATH training prompts under the common-math-v1 protocol."""

    def __init__(self, tokenizer, dataset_name: str, max_prompt_length: int | None = 256, protocol: str = "common-math-v1", invalid_gold: str = "exclude"):
        if dataset_name == "gsm8k":
            self.dataset = get_gsm8k_questions("train")
        elif dataset_name == "math":
            self.dataset = get_math_questions("train")
        else:
            raise ValueError("TrainingMathDataset supports only gsm8k or math")
        if protocol != "common-math-v1":
            raise ValueError("Unknown training math protocol")
        from data.data_utils import apply_common_math_protocol
        from data.data_utils import validate_common_math_gold
        self.dataset = apply_common_math_protocol(self.dataset)
        self.dataset = validate_common_math_gold(self.dataset, invalid_gold=invalid_gold)
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[int(idx)]
        prompt = maybe_apply_chat_template(example, self.tokenizer)["prompt"]
        question = example["prompt"][0]["content"]
        return prompt, question, example["answer"]

    def collate_fn(self, batch):
        prompts = [item[0] for item in batch]
        encoded = self.tokenizer(
            prompts,
            padding_side="left",
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        )
        input_ids, attention_mask = encoded.input_ids, encoded.attention_mask
        if self.max_prompt_length is not None:
            # Retain the final prompt tokens, including the assistant
            # generation header, when a formatted prompt is long.
            input_ids = input_ids[:, -self.max_prompt_length :]
            attention_mask = attention_mask[:, -self.max_prompt_length :]
        return {
            "input_ids": input_ids,
            "questions": [item[1] for item in batch],
            "answers": [item[2] for item in batch],
            "prompts": prompts,
            "attention_mask": attention_mask,
        }
