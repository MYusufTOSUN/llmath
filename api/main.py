"""LLMath REST API.

Çalıştırma: python -m uvicorn api.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from llmath import __version__
from llmath.cascade import run_cascade
from llmath.config import MODELS, PILOT_FILE
from llmath.dataset import import_questions
from llmath.db import Question, Response, Run, SessionLocal, Verification, init_db
from llmath.service import available_models, get_or_create_run, model_status, record, solve_and_verify, summarize

INTERACTIVE_RUN = "etkileşimli"


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    if PILOT_FILE.exists():
        import_questions()
    yield


app = FastAPI(title="LLMath API", version=__version__, lifespan=lifespan)


class SolveRequest(BaseModel):
    model_key: str
    strategy: str = Field(default="cot", pattern="^(cot|pot)$")
    question_id: str | None = None
    text: str | None = None
    gold: str | None = None
    lang: str = "tr"


@app.get("/health")
def health() -> dict:
    return {"ok": True, "version": __version__}


@app.get("/models")
def models() -> list[dict]:
    return available_models()


@app.get("/questions")
def questions(subset: str | None = None, lang: str | None = None) -> list[dict]:
    query = select(Question).order_by(Question.subset, Question.id)
    if subset:
        query = query.where(Question.subset == subset)
    if lang:
        query = query.where(Question.lang == lang)
    with SessionLocal() as session:
        return [{"id": q.id, "subset": q.subset, "lang": q.lang, "text": q.text, "gold": q.gold,
                 "answer_type": q.answer_type, "topic": q.topic, "source": q.source}
                for q in session.scalars(query)]


@app.post("/solve")
def solve(request: SolveRequest) -> dict:
    spec = MODELS.get(request.model_key)
    if spec is None:
        raise HTTPException(404, f"bilinmeyen model: {request.model_key}")
    ok, reason = model_status(spec)
    if not ok:
        raise HTTPException(409, reason)
    with SessionLocal() as session:
        text, gold, lang, answer_type = request.text, request.gold, request.lang, None
        if request.question_id:
            question = session.get(Question, request.question_id)
            if question is None:
                raise HTTPException(404, f"bilinmeyen soru: {request.question_id}")
            text, gold, lang, answer_type = question.text, question.gold, question.lang, question.answer_type
        if not text:
            raise HTTPException(422, "soru metni ya da question_id gerekli")
        result, verdict = solve_and_verify(text, gold, spec, request.strategy, lang, answer_type)
        run = get_or_create_run(session, INTERACTIVE_RUN)
        row = record(session, run, text, request.question_id, spec, result, verdict)
        session.commit()
        response = result.response
        return {
            "response_id": row.id,
            "answer": result.answer,
            "extraction_method": result.extraction_method,
            "gold": gold,
            "verdict": verdict.__dict__ if verdict else None,
            "solution": response.text if response else None,
            "reasoning": response.reasoning if response else None,
            "code": result.code,
            "exec_stdout": result.exec_stdout,
            "exec_error": result.exec_error,
            "error": result.error,
            "input_tokens": response.input_tokens if response else None,
            "output_tokens": response.output_tokens if response else None,
            "latency_ms": response.latency_ms if response else None,
            "cost_usd": response.cost_usd if response else None,
        }


@app.get("/runs")
def runs() -> list[dict]:
    with SessionLocal() as session:
        counts = dict(session.execute(
            select(Response.run_id, func.count(Response.id)).group_by(Response.run_id)).all())
        return [{"id": r.id, "name": r.name, "created_at": r.created_at.isoformat(),
                 "responses": counts.get(r.id, 0), "config": r.config}
                for r in session.scalars(select(Run).order_by(Run.id.desc()))]


@app.get("/runs/{run_id}/summary")
def run_summary(run_id: int) -> list[dict]:
    with SessionLocal() as session:
        if session.get(Run, run_id) is None:
            raise HTTPException(404, "çalıştırma bulunamadı")
        return summarize(session, run_id)


@app.get("/runs/{run_id}/cascade")
def run_cascade_analysis(run_id: int, cheap: str, strong: str, strong_strategy: str = "cot") -> dict:
    """Agreement-gated cascade simulated on the run's stored answers (AS2)."""
    with SessionLocal() as session:
        if session.get(Run, run_id) is None:
            raise HTTPException(404, "çalıştırma bulunamadı")
        return run_cascade(session, run_id, cheap, strong, strong_strategy)


@app.get("/runs/{run_id}/responses")
def run_responses(run_id: int, status: str | None = None) -> list[dict]:
    query = (select(Response, Verification, Question)
             .join(Verification, Verification.response_id == Response.id)
             .outerjoin(Question, Question.id == Response.question_id)
             .where(Response.run_id == run_id).order_by(Response.id))
    if status:
        query = query.where(Verification.status == status)
    with SessionLocal() as session:
        return [{
            "response_id": resp.id, "question_id": resp.question_id, "subset": q.subset if q else None,
            "lang": q.lang if q else None, "model": resp.model_key, "strategy": resp.strategy,
            "question": resp.question_text, "gold": q.gold if q else None,
            "answer": resp.extracted_answer, "extraction_method": resp.extraction_method,
            "status": ver.status, "layer": ver.layer, "detail": ver.detail,
            "ambiguous_format": ver.ambiguous_format, "solution": resp.raw_output, "code": resp.code,
            "exec_error": resp.exec_error, "latency_ms": resp.latency_ms,
        } for resp, ver, q in session.execute(query)]
