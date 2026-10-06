"""Pull the final answer (CoT) or the program (PoT) out of a model's reply."""

from __future__ import annotations

import re
from dataclasses import dataclass

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_LINE = re.compile(
    r"(?im)^[\s>*#-]*(?:nihai\s+cevap|son\s+cevap|cevap|sonuç|yanıt|final\s+answer|answer)"
    r"\s*(?:\*\*)?\s*[:：]\s*(?:\*\*)?\s*(?P<answer>.+?)\s*(?:\*\*)?\s*$"
)
_NUMBER = re.compile(r"[+-]?\d+(?:[.,]\d+)*")
_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(?P<code>.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class Extraction:
    answer: str | None
    method: str   # boxed | answer_line | last_number | none


def remove_thinking(text: str) -> str:
    return _THINK_BLOCK.sub("", text or "").strip()


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (or \\fbox{...}), with nested braces."""
    starts = [m.end() for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text)]
    if not starts:
        return None
    start = starts[-1]
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None


def extract_answer(text: str) -> Extraction:
    body = remove_thinking(text)
    if not body:
        return Extraction(None, "none")
    boxed = last_boxed(body)
    if boxed:
        return Extraction(boxed, "boxed")
    lines = list(_ANSWER_LINE.finditer(body))
    if lines:
        return Extraction(lines[-1]["answer"].strip().rstrip("."), "answer_line")
    numbers = _NUMBER.findall(body)
    if numbers:
        return Extraction(numbers[-1], "last_number")
    return Extraction(None, "none")


def extract_code(text: str) -> str | None:
    """The last fenced Python block, or None if the reply has no code block."""
    blocks = _CODE_BLOCK.findall(remove_thinking(text))
    return blocks[-1].strip() if blocks else None
