from datasets import load_dataset
from datasets import load_from_disk

from common.parsing.answers import extract_math_answer
from data.loaders.gsm8k import DATASETS_PATH
from data.loaders.gsm8k import GSM8KDataset

FOUR_SHOT_MATH500 = "four-shot-tps-math500-v1"

# The four fixed Minerva solutions of lm-eval 0.4.9.1 minerva_math/utils.py.
MINERVA_MATH_FEWSHOT = [
    {
        "problem": r"Find the domain of the expression  $\frac{\sqrt{x-2}}{\sqrt{5-x}}$.}",
        "solution": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.",
    },
    {
        "problem": r"If $\det \mathbf{A} = 2$ and $\det \mathbf{B} = 12,$ then find $\det (\mathbf{A} \mathbf{B}).$",
        "solution": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $24$. I hope it is correct.",
    },
    {
        "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
        "solution": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nFinal Answer: The final answer is $16$. I hope it is correct.",
    },
    {
        "problem": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
        "solution": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.",
    },
]


class MATH500Dataset(GSM8KDataset):
    """MATH500 (HuggingFaceH4/MATH-500) with the common-math-v1 prompt."""

    def __init__(self, tokenizer, num_examples=0, dataset_id="HuggingFaceH4/MATH-500",
                 prompt_protocol=None, fewshot_seed=1234):
        self.dataset_id = dataset_id
        super().__init__(tokenizer, num_examples, prompt_protocol=prompt_protocol,
                         fewshot_seed=fewshot_seed)

    def load_test_dataset(self):
        local_path = DATASETS_PATH / "math500_h4"
        if local_path.exists():
            return load_from_disk(str(local_path))["test"]
        return load_dataset(self.dataset_id)["test"]

    def load_four_shot_examples(self):
        """The four Minerva solutions as common-math-v1 chat turns.

        Minerva's closing ``Final Answer: The final answer is $X$. I hope it is
        correct.`` line becomes ``The final answer is \\boxed{X}.``, matching
        the four-shot GSM8K protocol. None of the four problems is in MATH500.
        """
        if self.prompt_protocol != FOUR_SHOT_MATH500:
            raise ValueError(f"Unknown prompt protocol: {self.prompt_protocol}")
        examples = []
        for example in MINERVA_MATH_FEWSHOT:
            reasoning, final = example["solution"].split("\nFinal Answer: The final answer is $")
            final = final.removesuffix("$. I hope it is correct.")
            examples.append(
                (example["problem"], f"{reasoning.strip()}\nThe final answer is \\boxed{{{final}}}.")
            )
        return examples

    def __getitem__(self, idx):
        item = self.dataset[int(idx)]
        question = item["problem"]
        answer = item.get("answer")
        if answer is None:
            answer = extract_math_answer(item["solution"])
        return self.create_prompt(question), question, answer
