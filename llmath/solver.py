"""Solve one question with one model and one strategy."""

from __future__ import annotations

from dataclasses import dataclass

from .config import ModelSpec
from .extract import extract_answer, extract_code
from .llm import LLMError, LLMResponse, chat
from .prompts import build_messages
from .sandbox import run_code


@dataclass
class SolveResult:
    strategy: str
    answer: str | None
    extraction_method: str
    response: LLMResponse | None = None
    code: str | None = None
    exec_stdout: str | None = None
    exec_error: str | None = None
    error: str | None = None


def solve(question: str, spec: ModelSpec, strategy: str, lang: str = "tr") -> SolveResult:
    try:
        response = chat(spec, build_messages(question, strategy, lang))
    except LLMError as exc:
        return SolveResult(strategy, None, "none", error=str(exc))

    if strategy == "cot":
        extraction = extract_answer(response.text)
        return SolveResult(strategy, extraction.answer, extraction.method, response)

    if strategy == "pot":
        code = extract_code(response.text)
        if code is None:
            return SolveResult(strategy, None, "no_code", response)
        result = run_code(code)
        method = "program_output" if result.ok else ("blocked" if result.blocked else "program_error")
        return SolveResult(strategy, result.answer if result.ok else None, method, response,
                           code=code, exec_stdout=result.stdout, exec_error=result.stderr or None)

    raise ValueError(f"bilinmeyen strateji: {strategy}")
