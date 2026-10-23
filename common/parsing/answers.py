# Adapted from https://github.com/dllm-reasoning/d1 (Apache 2.0)
"""Gold-answer extraction for GSM8K and MATH (model outputs are scored with math-verify)."""
import re


def extract_gsm8k_gold(generated_text):
    """Extract the first numerical answer following '####' in the generated text.

    :param generated_text: Generated text containing #### delimiter
    :return: Extracted float answer, or None if no valid answer found
    """
    try:
        # Use regex to find the first occurrence of #### followed by a number
        # Handle formats like: #### 18, #### $18, #### 18.0
        match = re.search(r"####\s*\$?([\d,]+(?:\.\d+)?)", generated_text)
        if match:
            return float(match.group(1).replace(",", ""))
    except Exception as e:
        print(f"Error extracting answer: {e}, Text: {generated_text[:100]}")
    return None


def extract_hash_answer(text: str) -> str | None:
    """Extract answer from text that has #### delimiter.

    :param text: Text containing answer after #### delimiter
    :return: Extracted answer string, or None if no #### delimiter found
    """
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def extract_math_answer(raw_generation: str) -> str | None:
    """Extract LaTeX answer from MATH generation.

    :param raw_generation: Generated text to parse
    :return: Extracted LaTeX answer string, or None if no valid answer found
    """
    parsed_answer = None

    # Try \boxed{} format first
    try:
        parsed_answer = remove_boxed(last_boxed_only_string(raw_generation))
    except Exception:
        pass

    # Try <answer></answer> format
    if not parsed_answer:
        answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
        if answer_match:
            parsed_answer = answer_match.group(1).strip()

    return parsed_answer


def last_boxed_only_string(string):
    """Extract the last LaTeX boxed expression from a string.

    :param string: Input string containing LaTeX markup
    :return: Last occurrence of \\boxed{...} or \\fbox{...}, or original string if not found
    """
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return string

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    """Remove LaTeX boxed wrapper from a string.

    :param s: String containing \\boxed{...} or \\boxed ...
    :return: Contents inside the boxed command, or original string if format invalid
    """
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"

        return s[len(left) : -1]
    except (AssertionError, IndexError):
        return s
