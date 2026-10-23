import torch
from datasets import load_dataset
from datasets import load_from_disk

from data.loaders.gsm8k import DATASETS_PATH


class MBPPDataset(torch.utils.data.Dataset):
    """MBPP (sanitized test split, 257 tasks), zero-shot, ``mbpp_instruct`` format."""

    def __init__(self, tokenizer, num_examples=0, dataset_config="sanitized", split="test"):
        if num_examples:
            raise ValueError("MBPP is evaluated zero-shot")
        if dataset_config not in {"full", "sanitized"}:
            raise ValueError(f"Unsupported MBPP config: {dataset_config}")
        self.tokenizer = tokenizer
        self.actual_num_examples = 0
        local_path = DATASETS_PATH / ("mbpp" if dataset_config == "full" else "mbpp_sanitized")
        dataset_dict = (
            load_from_disk(str(local_path))
            if local_path.exists()
            else load_dataset("google-research-datasets/mbpp", dataset_config)
        )
        self.dataset = dataset_dict[split]
        print(f"evaluating {len(self.dataset)} examples")

    def __len__(self):
        return len(self.dataset)

    def create_prompt(self, text, test_list):
        tests_str = "\n".join(test_list[:3])
        task_prompt = (
            f"You are an expert Python programmer, and here is your task:\n{text}\n"
            f"Your code should pass these tests:\n{tests_str}"
        )
        user_input = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": task_prompt}], add_generation_prompt=True, tokenize=False
        )
        # The model continues from lm-eval's generation prefix.
        return user_input + "\n```python\n"

    def __getitem__(self, idx):
        item = self.dataset[int(idx)]
        text = item.get("text", item.get("prompt"))  # Sanitized uses ``prompt``.
        test_list = item["test_list"]
        test_cases = "\n".join([*item.get("test_imports", []), *test_list])
        return self.create_prompt(text, test_list), text, item["task_id"], test_cases

    def collate_fn(self, batch):
        prompts = [item[0] for item in batch]
        encoded = self.tokenizer(
            prompts, padding_side="left", return_tensors="pt", padding="longest"
        )
        return {
            "input_ids": encoded.input_ids,
            "attention_mask": encoded.attention_mask,
            "prompts": prompts,
            "texts": [item[1] for item in batch],
            "task_ids": [item[2] for item in batch],
            "test_cases": [item[3] for item in batch],
        }
