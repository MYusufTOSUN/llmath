"""Solve one question with one model and one strategy."""

from __future__ import annotations

from dataclasses import dataclass

from .config import ModelSpec
from .extract import extract_answer, extract_code
from .llm import LLMError, LLMResponse, chat
from .prompts import PROMPT_VERSION, build_messages, prompt_version_for
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
    prompt_version: str = PROMPT_VERSION


def solve(question: str, spec: ModelSpec, strategy: str, lang: str = "tr", variant: str = "base") -> SolveResult:
    version = prompt_version_for(variant)
    try:
        response = chat(spec, build_messages(question, strategy, lang, variant))
    except LLMError as exc:
        return SolveResult(strategy, None, "none", error=str(exc), prompt_version=version)

    if strategy == "cot":
        extraction = extract_answer(response.text)
        return SolveResult(strategy, extraction.answer, extraction.method, response, prompt_version=version)

    if strategy == "pot":
        code = extract_code(response.text)
        if code is None:
            return SolveResult(strategy, None, "no_code", response, prompt_version=version)
        result = run_code(code)
        return SolveResult(strategy, *program_answer(result), response, code=code, exec_stdout=result.stdout,
                           exec_error=result.stderr or None, prompt_version=version)

    raise ValueError(f"bilinmeyen strateji: {strategy}")


def program_answer(result) -> tuple[str | None, str]:
    """The answer and extraction method of an executed PoT program."""
    if result.ok:
        return result.answer, "program_output"
    return None, "blocked" if result.blocked else "program_error"
