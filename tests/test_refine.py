import pytest
from sqlalchemy import select

from llmath import refine
from llmath.db import Question, Response, Run, SessionLocal, Verification, init_db
from llmath.llm import LLMError, LLMResponse
from llmath.prompts import REFINE_VERSION

GOLD = "1234"   # appears nowhere else, so a prompt that leaks the answer key is easy to catch


def question(lang="tr"):
    return Question(id="r-1", subset="P3", lang=lang, text="Bir sayının iki katı 2468'dir. Sayı kaçtır?",
                    gold=GOLD, answer_type="integer", source="test", license="CC0")


def cot(answer, text=None):
    return Response(strategy="cot", extracted_answer=answer, extraction_method="boxed",
                    raw_output=text or f"Adımlar... \\boxed{{{answer}}}")


def pot(answer, method="program_output", code="print(1234)", error=None):
    return Response(strategy="pot", extracted_answer=answer, extraction_method=method, code=code,
                    exec_error=error, raw_output=f"```python\n{code}\n```" if code else "kod yok")


def replies(*texts):
    """A fake model that answers with the given texts in order and records what it was sent."""
    sent = []

    def call(messages):
        sent.append(list(messages))   # the conversation grows after the call
        return LLMResponse(texts[len(sent) - 1], None, 10, 5, 100, 0.0)
    return call, sent


def test_pot_problem_kinds():
    assert refine.pot_problem("no_code", None, None) == ("no_code", "")
    assert refine.pot_problem("blocked", None, "izin verilmeyen modül: os") == ("blocked", "izin verilmeyen modül: os")
    assert refine.pot_problem("program_error", None, "zaman aşımı") == ("timeout", "")
    assert refine.pot_problem("program_error", None, "NameError: x")[0] == "error"
    assert refine.pot_problem("program_output", None, None) == ("empty", "")
    assert refine.pot_problem("program_output", "5", None) is None


def test_agreeing_answers_need_no_call():
    call, sent = replies()
    steps = refine.refine_question(question(), cot("1234"), pot("1234"), call)
    assert sent == [] and len(steps) == 1
    decision, verdict = steps[0]
    assert (decision.strategy, decision.extraction_method, decision.answer) == ("refine", "agree", "1234")
    assert verdict.status == "correct" and decision.prompt_version == REFINE_VERSION


def test_failed_program_gets_the_error_and_agreement_after_the_fix_is_accepted():
    call, sent = replies("```python\nprint(1234)\n```")
    original = pot(None, "program_error", code="print(x)", error="NameError: name 'x' is not defined")
    steps = refine.refine_question(question(), cot("1234"), original, call)
    assert len(sent) == 1
    messages = sent[0]
    assert messages[-2] == {"role": "assistant", "content": original.raw_output}   # the model sees its own program
    assert "NameError" in messages[-1]["content"] and "python" in messages[-1]["content"]
    fix, decision = steps[0][0], steps[-1][0]
    assert (fix.strategy, fix.extraction_method, fix.answer) == ("pot_fix", "program_output", "1234")
    assert (decision.extraction_method, decision.answer) == ("pot_fix", "1234")


def test_disagreement_shows_both_solutions_but_never_the_answer_key():
    call, sent = replies("Kontrol ettim. \\boxed{1234}")
    steps = refine.refine_question(question(), cot("617", "2468 / 4 = 617, \\boxed{617}"),
                                   pot("1240", code="print(1240)"), call)
    prompt = sent[0][1]["content"]
    assert "617" in prompt and "print(1240)" in prompt and "1240" in prompt
    assert GOLD not in prompt and all(GOLD not in m["content"] for m in sent[0])
    assert [s[0].strategy for s in steps] == ["consistency", "refine"]
    decision, verdict = steps[-1]
    assert (decision.extraction_method, decision.answer, verdict.status) == ("consistency", "1234", "correct")


def test_program_that_stays_broken_goes_to_f2_and_f2_without_answer_falls_back_to_cot():
    call, sent = replies("```python\nprint(y)\n```", "yine yanlış", "Bilemedim.")
    steps = refine.refine_question(question("en"), cot("617"), pot(None, "no_code", code=None), call,
                                   max_rounds=2)
    assert [s[0].strategy for s in steps] == ["pot_fix", "pot_fix", "consistency", "refine"]
    assert "NameError" in sent[1][-1]["content"]            # the second round hears why the first fix failed
    assert "print(y)" in sent[2][1]["content"]              # F2 sees the last program that was written
    decision = steps[-1][0]
    assert (decision.extraction_method, decision.answer) == ("cot_fallback", "617")


@pytest.fixture
def base_run():
    init_db()
    with SessionLocal() as session:
        session.add(Question(id="r-db-1", subset="P3", lang="tr", text="2 + 2 kaçtır?", gold="4",
                             answer_type="integer", source="test", license="CC0"))
        session.add(Question(id="r-db-2", subset="P3", lang="tr", text="3 + 3 kaçtır?", gold="6",
                             answer_type="integer", source="test", license="CC0"))
        run = Run(name="test-refine-base", config={})
        session.add(run)
        session.flush()
        for qid, gold, c, p in [("r-db-1", "4", "4", "4"), ("r-db-2", "6", "5", "6")]:
            for strategy, answer in (("cot", c), ("pot", p)):
                row = Response(run_id=run.id, question_id=qid, question_text="-", model_key="gemini-flash-lite",
                               strategy=strategy, prompt_version="v0.1", extracted_answer=answer,
                               extraction_method="boxed" if strategy == "cot" else "program_output",
                               raw_output=f"\\boxed{{{answer}}}", code="print(6)" if strategy == "pot" else None)
                session.add(row)
                session.flush()
                session.add(Verification(response_id=row.id, status="correct" if answer == gold else "incorrect",
                                         layer="exact"))
        session.commit()
    yield
    with SessionLocal() as session:
        for run in session.scalars(select(Run).where(Run.name.like("test-refine%"))):
            for response in run.responses:
                if response.verification is not None:
                    session.delete(response.verification)
                session.delete(response)
            session.delete(run)
        for q in session.scalars(select(Question).where(Question.id.like("r-db-%"))):
            session.delete(q)
        session.commit()


def test_run_stores_every_call_and_resumes_without_repeating(base_run, monkeypatch, capsys):
    calls = []

    def fake_chat(spec, messages):
        calls.append(messages)
        return LLMResponse("\\boxed{6}", None, 10, 5, 100, 0.0)

    monkeypatch.setattr(refine, "chat", fake_chat)
    monkeypatch.setattr(refine, "model_status", lambda spec: (True, ""))
    args = ["--base", "test-refine-base", "--name", "test-refine-1", "--models", "gemini-flash-lite", "--pause", "0"]
    assert refine.main(args) == 0
    assert len(calls) == 1                                    # only the disagreeing question needs F2
    with SessionLocal() as session:
        run = session.scalars(select(Run).where(Run.name == "test-refine-1")).one()
        rows = sorted((r.question_id, r.strategy, r.extraction_method) for r in run.responses)
    assert rows == [("r-db-1", "refine", "agree"), ("r-db-2", "consistency", "boxed"),
                    ("r-db-2", "refine", "consistency")]
    assert "düzeltilmiş %100.0 (2/2" in capsys.readouterr().out
    assert refine.main(args) == 0 and len(calls) == 1          # resumed run: nothing left to do


def test_api_error_stores_nothing_for_the_question(base_run, monkeypatch):
    def failing_chat(spec, messages):
        raise LLMError("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(refine, "chat", failing_chat)
    monkeypatch.setattr(refine, "model_status", lambda spec: (True, ""))
    refine.main(["--base", "test-refine-base", "--name", "test-refine-2", "--models", "gemini-flash-lite",
                 "--pause", "0"])
    with SessionLocal() as session:
        run = session.scalars(select(Run).where(Run.name == "test-refine-2")).one()
        assert [(r.question_id, r.strategy) for r in run.responses] == [("r-db-1", "refine")]
