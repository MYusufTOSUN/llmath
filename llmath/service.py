"""Operations shared by the batch runner and the API: solve, verify, store, summarize."""

from __future__ import annotations

from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import MODELS, ModelSpec, has_api_key
from .db import Question, Response, Run, Verification
from .llm import ollama_models
from .prompts import PROMPT_VERSION
from .solver import SolveResult, solve
from .verify import Verdict, verify


def model_status(spec: ModelSpec, local_models: set[str] | None = None) -> tuple[bool, str]:
    """Whether the model can be called right now, and why not if it cannot."""
    if spec.provider == "ollama":
        local = ollama_models() if local_models is None else local_models
        if not local:
            return False, "Ollama çalışmıyor"
        if spec.model not in local and f"{spec.model}:latest" not in local:
            return False, f"model indirilmemiş: ollama pull {spec.model}"
        return True, ""
    if not has_api_key(spec):
        return False, f"{spec.env_key} tanımlı değil (.env)"
    return True, ""


def available_models() -> list[dict]:
    local = ollama_models()
    rows = []
    for spec in MODELS.values():
        ok, reason = model_status(spec, local)
        rows.append({"key": spec.key, "label": spec.label, "available": ok, "reason": reason})
    return rows


def get_or_create_run(session: Session, name: str, config: dict | None = None) -> Run:
    run = session.scalars(select(Run).where(Run.name == name)).first()
    if run is None:
        run = Run(name=name, config={"prompt_version": PROMPT_VERSION, **(config or {})})
        session.add(run)
        session.flush()
    return run


def solve_and_verify(question_text: str, gold: str | None, spec: ModelSpec, strategy: str,
                     lang: str = "tr", answer_type: str | None = None) -> tuple[SolveResult, Verdict | None]:
    result = solve(question_text, spec, strategy, lang)
    if result.error:
        return result, Verdict("unparsed", "llm_error", result.error)
    if gold is None:
        return result, None
    return result, verify(gold, result.answer, answer_type)


def record(session: Session, run: Run, question_text: str, question_id: str | None,
           spec: ModelSpec, result: SolveResult, verdict: Verdict | None) -> Response:
    response = result.response
    row = Response(
        run_id=run.id, question_id=question_id, question_text=question_text,
        model_key=spec.key, strategy=result.strategy, prompt_version=PROMPT_VERSION,
        raw_output=response.text if response else None,
        reasoning=response.reasoning if response else None,
        extracted_answer=result.answer, extraction_method=result.extraction_method,
        code=result.code, exec_stdout=result.exec_stdout, exec_error=result.exec_error,
        input_tokens=response.input_tokens if response else None,
        output_tokens=response.output_tokens if response else None,
        cost_usd=response.cost_usd if response else None,
        latency_ms=response.latency_ms if response else None,
        finish_reason=response.finish_reason if response else None,
        error=result.error,
    )
    session.add(row)
    session.flush()
    if verdict is not None:
        session.add(Verification(response_id=row.id, status=verdict.status, layer=verdict.layer,
                                 detail=verdict.detail, ambiguous_format=verdict.ambiguous_format,
                                 normalized=verdict.normalized))
    return row


def summarize(session: Session, run_id: int) -> list[dict]:
    """Accuracy per model, strategy, subset and language for one run.

    Calls that failed at the API (quota, outage) are left out: they say nothing about the
    model and are retried when the run is resumed.
    """
    rows = session.execute(
        select(Response.model_key, Response.strategy, Question.subset, Question.lang,
               Verification.status, Response.latency_ms, Response.output_tokens, Response.cost_usd)
        .join(Question, Response.question_id == Question.id)
        .join(Verification, Verification.response_id == Response.id)
        .where(Response.run_id == run_id, Response.error.is_(None))
    ).all()
    groups: dict[tuple, list] = defaultdict(list)
    for model_key, strategy, subset, lang, status, latency, tokens, cost in rows:
        groups[(model_key, strategy, subset, lang)].append((status, latency, tokens, cost))
    summary = []
    for (model_key, strategy, subset, lang), items in sorted(groups.items()):
        n = len(items)
        correct = sum(1 for status, *_ in items if status == "correct")
        unparsed = sum(1 for status, *_ in items if status == "unparsed")
        latencies = [lat for _, lat, _, _ in items if lat is not None]
        tokens = [tok for _, _, tok, _ in items if tok is not None]
        costs = [c for *_, c in items if c is not None]
        summary.append({
            "model": model_key, "strategy": strategy, "subset": subset, "lang": lang, "n": n,
            "correct": correct, "accuracy": round(correct / n, 3) if n else 0.0,
            "unparsed": unparsed,
            "avg_latency_s": round(sum(latencies) / len(latencies) / 1000, 1) if latencies else None,
            "avg_output_tokens": round(sum(tokens) / len(tokens)) if tokens else None,
            "cost_usd": round(sum(costs), 4) if costs else None,
        })
    return summary
