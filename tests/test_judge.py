import pytest
from sqlalchemy import select

from llmath import judge
from llmath.config import MODELS
from llmath.db import Annotation, Question, Response, Run, SessionLocal, Verification, init_db
from llmath.llm import LLMResponse
from llmath.taxonomy import ERROR_TYPES, solution_steps


def test_parse_label_checks_the_codebook():
    text = 'Tamam.\n```json\n{"error_type": "h10", "newman_stage": "Anlama", "first_error_step": 2, ' \
           '"explanation": "6,5 sayısını iki sayı okudu."}\n```'
    assert judge.parse_label(text, 4) == {"error_type": "H10", "newman_stage": "anlama", "first_error_step": 2,
                                          "note": "6,5 sayısını iki sayı okudu."}
    assert judge.parse_label('{"error_type": "H1", "newman_stage": "işlem"}', 3)["newman_stage"] == "islem"
    # an unknown stage falls back to the type's usual one, a step outside the solution is dropped
    label = judge.parse_label('{"error_type": "H6", "newman_stage": "?", "first_error_step": 9}', 3)
    assert (label["newman_stage"], label["first_error_step"]) == ("donusturme", None)
    assert judge.parse_label('{"error_type": "Y1", "newman_stage": "islem"}', 3)["newman_stage"] is None
    assert judge.parse_label('{"error_type": "H99"}', 3) is None
    assert judge.parse_label("Bence H1.", 3) is None


def test_judge_sees_numbered_steps_the_key_and_the_codebook():
    solution = "Önce toplarız.\n\n2 + 3 = 6\n\nCevap \\boxed{6}"
    q = Question(id="j-1", text="2 + 3 kaçtır?", gold="5", lang="tr")
    r = Response(raw_output=solution, extracted_answer="6", code=None, finish_reason="length")
    v = Verification(status="incorrect", detail="beklenen 5, bulunan 6")
    messages, count = judge.judge_messages(q, r, v)
    prompt = messages[1]["content"]
    assert count == len(solution_steps(solution)) == 3
    assert "[2] 2 + 3 = 6" in prompt and "## Cevap anahtarı\n5" in prompt and "beklenen 5, bulunan 6" in prompt
    assert all(code in prompt for code in ERROR_TYPES) and "token sınırında" in prompt

    pot = Response(code="x = 2\nprint(x + 4)", exec_stdout="6\n", exec_error=None, extracted_answer="6")
    messages, count = judge.judge_messages(q, pot, v)
    assert count == 2 and "  2| print(x + 4)" in messages[1]["content"] and "kod satırı" in messages[1]["content"]


def test_codebook_sends_turkish_number_misreadings_to_h10():
    # j0.1 used a decimal in H3's example, and the judge filed "8,4 -> Rational(8, 4)" under H3
    assert "H10" in ERROR_TYPES["H3"].definition and "," not in ERROR_TYPES["H3"].example.split(";")[0]
    assert "ondalık virgül" in ERROR_TYPES["H10"].definition
    assert judge.annotator_name("gemini-flash-lite") == f"llm:gemini-flash-lite/{judge.JUDGE_VERSION}"


def test_a_judge_may_not_label_its_own_family():
    gemini = MODELS["gemini-flash"]
    assert judge.same_family(gemini, "gemini-flash-lite")
    assert not judge.same_family(gemini, "qwen3-8b")
    assert judge.same_family(gemini, "silinmis-model")   # unknown models are refused too
    # DeepSeek-R1 8B is distilled onto Qwen3-8B, so Qwen may not judge it, and it may not judge Qwen
    assert judge.same_family(MODELS["qwen3-8b-think"], "deepseek-r1-8b")
    assert judge.same_family(MODELS["deepseek-r1-8b"], "qwen3-8b")
    assert not judge.same_family(gemini, "deepseek-r1-8b")
    assert not judge.same_family(MODELS["qwen3-8b-think"], "gemini-flash-lite")


def test_cohen_kappa():
    assert judge.cohen_kappa(["H1", "H3", "H1"], ["H1", "H3", "H1"]) == 1.0
    assert judge.cohen_kappa(["H1", "H1", "H3", "H3"], ["H1", "H3", "H1", "H3"]) == 0.0
    assert judge.cohen_kappa([], []) is None


@pytest.fixture
def wrong_answers():
    init_db()
    with SessionLocal() as session:
        session.add(Question(id="j-db-1", subset="P3", lang="tr", text="2 + 3 kaçtır?", gold="5",
                             answer_type="integer", source="test", license="CC0"))
        run = Run(name="test-judge", config={})
        session.add(run)
        session.flush()
        for model in ("qwen3-8b", "gemini-flash-lite"):
            row = Response(run_id=run.id, question_id="j-db-1", question_text="-", model_key=model, strategy="cot",
                           prompt_version="v0.1", raw_output="2 + 3 = 6\n\\boxed{6}", extracted_answer="6",
                           extraction_method="boxed")
            session.add(row)
            session.flush()
            session.add(Verification(response_id=row.id, status="incorrect", layer="numeric"))
        session.commit()
    yield
    with SessionLocal() as session:
        run = session.scalars(select(Run).where(Run.name == "test-judge")).one()
        for response in run.responses:
            for a in session.scalars(select(Annotation).where(Annotation.response_id == response.id)):
                session.delete(a)
            session.delete(response.verification)
            session.delete(response)
        session.delete(run)
        session.delete(session.get(Question, "j-db-1"))
        session.commit()


def test_judge_labels_other_families_and_reminds_once_after_a_bad_reply(wrong_answers, monkeypatch):
    replies = iter(["H1 bence", '{"error_type": "H1", "newman_stage": "islem", "first_error_step": 1, '
                                '"explanation": "2 + 3 yanlış toplandı."}'])
    sent = []

    def fake_chat(spec, messages):
        sent.append(messages)
        return LLMResponse(next(replies), None, 10, 5, 100, 0.0)

    monkeypatch.setattr(judge, "chat", fake_chat)
    monkeypatch.setattr(judge, "model_status", lambda spec: (True, ""))
    assert judge.main(["--run", "test-judge", "--judge", "gemini-flash", "--pause", "0"]) == 0
    assert len(sent) == 2 and sent[1][-1]["content"] == judge.REMINDER
    with SessionLocal() as session:
        labels = session.execute(select(Annotation, Response.model_key)
                                 .join(Response, Response.id == Annotation.response_id)
                                 .where(Annotation.annotator == judge.annotator_name("gemini-flash"))).all()
    assert [(a.error_type, a.newman_stage, a.first_error_step, model) for a, model in labels] == [
        ("H1", "islem", 1, "qwen3-8b")]   # Gemini's own answer is left for a judge from another family
    assert judge.main(["--run", "test-judge", "--judge", "gemini-flash", "--pause", "0"]) == 0
    assert len(sent) == 2                  # nothing left to label
