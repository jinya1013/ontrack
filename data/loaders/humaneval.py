# Adapted from https://github.com/dllm-reasoning/d1 (Apache 2.0)
import torch
from datasets import load_dataset
from datasets import load_from_disk

from data.loaders.gsm8k import DATASETS_PATH


class HumanEvalDataset(torch.utils.data.Dataset):
    """HumanEval, zero-shot, in lm-eval's ``humaneval_instruct`` format."""

    def __init__(self, tokenizer, num_examples=0):
        if num_examples:
            raise ValueError("HumanEval is evaluated zero-shot")
        self.tokenizer = tokenizer
        self.actual_num_examples = 0
        local_path = DATASETS_PATH / "humaneval"
        self.dataset = (
            load_from_disk(str(local_path))["test"]
            if local_path.exists()
            else load_dataset("openai/openai_humaneval")["test"]
        )
        print(f"evaluating {len(self.dataset)} examples")

    def __len__(self):
        return len(self.dataset)

    def create_prompt(self, prompt_text):
        doc_to_text = (
            f"Write a solution to the following problem and make sure that it passes the tests:\n"
            f"```python\n{prompt_text}\n```\n"
        )
        user_input = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": doc_to_text}], add_generation_prompt=True, tokenize=False
        )
        # The model continues from lm-eval's generation prefix.
        return user_input + f"Here is the completed function:\n```python\n{prompt_text}\n"

    def __getitem__(self, idx):
        item = self.dataset[int(idx)]
        prompt = item["prompt"]  # Function signature + docstring.
        test_cases = f"{item['test']}\ncheck({item['entry_point']})\n"
        return self.create_prompt(prompt), prompt, item["task_id"], test_cases, item["entry_point"]

    def collate_fn(self, batch):
        prompts = [item[0] for item in batch]
        encoded = self.tokenizer(
            prompts, padding_side="left", return_tensors="pt", padding="longest"
        )
        return {
            "input_ids": encoded.input_ids,
            "attention_mask": encoded.attention_mask,
            "prompts": prompts,
            "raw_prompts": [item[1] for item in batch],
            "task_ids": [item[2] for item in batch],
            "test_cases": [item[3] for item in batch],
            "entry_points": [item[4] for item in batch],
        }
