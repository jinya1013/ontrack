"""Select the final explicit answer without flattening nested LaTeX braces."""

import re


def final_answer_text(text):
    boxes = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    if boxes:
        start = boxes[-1].end()
        depth = 1
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index].strip()
        return None
    answers = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return answers[-1].strip() if answers else None
