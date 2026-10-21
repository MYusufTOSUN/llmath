import json

import pytest
from sqlalchemy import select

from llmath import runner
from llmath.db import Question, Run, SessionLocal
from llmath.solver import SolveResult
from llmath.verify import Verdict


@pytest.fixture
def question_file(tmp_path):
    rows = [{"id": f"t-{i}", "subset": "P3", "lang": "tr", "text": f"{i} + {i} kaçtır?", "gold": str(2 * i),
             "answer_type": "integer", "source": "test", "license": "CC0"} for i in (1, 2, 3)]
    path = tmp_path / "set.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    yield path
    with SessionLocal() as session:   # leave the shared test database as other tests expect it
        for run in session.scalars(select(Run).where(Run.name.like("test-runner%"))):
            for response in run.responses:
                if response.verification is not None:
                    session.delete(response.verification)
                session.delete(response)
            session.delete(run)
        for question in session.scalars(select(Question).where(Question.id.like("t-%"))):
            session.delete(question)
        session.commit()


def test_runner_solves_only_the_questions_in_the_given_file(question_file, monkeypatch):
    texts = []

    def fake_solve(text, gold, spec, strategy, lang, answer_type, variant):
        texts.append(text)
        return SolveResult(strategy, gold, "boxed"), Verdict("correct", "exact")

    monkeypatch.setattr(runner, "model_status", lambda spec: (True, ""))
    monkeypatch.setattr(runner, "solve_and_verify", fake_solve)
    runner.main(["--name", "test-runner", "--data", str(question_file), "--models", "qwen3-8b",
                 "--strategies", "cot"])
    assert sorted(texts) == ["1 + 1 kaçtır?", "2 + 2 kaçtır?", "3 + 3 kaçtır?"]


def test_runner_stops_a_model_after_repeated_api_errors(question_file, monkeypatch):
    strategies = []

    def failing_solve(text, gold, spec, strategy, lang, answer_type, variant):
        strategies.append(strategy)
        return (SolveResult(strategy, None, "none", error="429 RESOURCE_EXHAUSTED"),
                Verdict("unparsed", "llm_error"))

    monkeypatch.setattr(runner, "model_status", lambda spec: (True, ""))
    monkeypatch.setattr(runner, "solve_and_verify", failing_solve)
    runner.main(["--name", "test-runner-errors", "--data", str(question_file), "--models", "qwen3-8b",
                 "--max-errors", "2"])
    assert strategies == ["cot", "cot"]   # stopped after two failures; PoT never started
