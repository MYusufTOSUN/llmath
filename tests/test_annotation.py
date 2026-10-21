from fastapi.testclient import TestClient
from sqlalchemy import select

from api.main import app
from llmath.config import MODELS
from llmath.db import Annotation, Response, Run, SessionLocal
from llmath.llm import LLMResponse
from llmath.service import get_or_create_run, record
from llmath.solver import SolveResult
from llmath.taxonomy import ERROR_TYPES, MEASUREMENT_CODES, NEWMAN_STAGES, codebook_markdown, solution_steps
from llmath.verify import Verdict


def test_taxonomy_is_complete():
    assert [code for code in ERROR_TYPES if code.startswith("H")] == [f"H{i}" for i in range(1, 14)]
    assert MEASUREMENT_CODES == {"Y1", "Y2"}
    for t in ERROR_TYPES.values():
        assert t.definition and t.example
        assert (t.stage is None) == (t.code in MEASUREMENT_CODES)
        assert t.stage is None or t.stage in NEWMAN_STAGES
    assert "| H10 | Türkçe ifadeyi yanlış yorumlama |" in codebook_markdown()


def test_a_displayed_formula_is_never_cut_into_two_steps():
    solution = "Önce denklemi yazarız.\n\n\\[\nx = 2\n\n+ 3\n\\]\n\nSonra $$a$$ buluruz.\n\nCevap \\boxed{5}"
    assert solution_steps(solution) == ["Önce denklemi yazarız.", "\\[\nx = 2\n\n+ 3\n\\]",
                                        "Sonra $$a$$ buluruz.", "Cevap \\boxed{5}"]
    assert solution_steps("Tek paragraf\nikinci satır") == ["Tek paragraf", "ikinci satır"]


def _remove_run(name: str) -> None:
    with SessionLocal() as session:
        for run in session.scalars(select(Run).where(Run.name == name)):
            for response in session.scalars(select(Response).where(Response.run_id == run.id)):
                for annotation in session.scalars(select(Annotation).where(Annotation.response_id == response.id)):
                    session.delete(annotation)
                if response.verification is not None:
                    session.delete(response.verification)
                session.delete(response)
            session.delete(run)
        session.commit()


def test_annotation_queue_is_blind_and_labels_can_be_replaced():
    with TestClient(app) as client:
        question = client.get("/questions").json()[0]   # the pilot set is loaded at start-up
        with SessionLocal() as session:
            run = get_or_create_run(session, "test-annotation")
            for answer, status in (("1", "incorrect"), (question["gold"], "correct"), (None, "unparsed")):
                record(session, run, question["text"], question["id"], MODELS["qwen3-8b"],
                       SolveResult("cot", answer, "boxed", LLMResponse(f"Çözüm... {answer}", None, 1, 1, 1, 0.0)),
                       Verdict(status, "exact"))
            # a refinement decision has no solution to read, so it is never queued
            record(session, run, question["text"], question["id"], MODELS["qwen3-8b"],
                   SolveResult("refine", "1", "consistency"), Verdict("incorrect", "exact"))
            session.commit()
            url = f"/runs/{run.id}/annotation/next"
        try:
            first = client.get(url, params={"annotator": "human"}).json()
            assert (first["done"], first["remaining"]) == (0, 2)        # the correct answer is not queued
            rid = first["item"]["response_id"]
            skipped = client.get(url, params={"annotator": "human", "skip": str(rid)}).json()
            assert skipped["item"]["response_id"] != rid

            body = {"response_id": rid, "annotator": "human", "error_type": "H1", "newman_stage": "islem"}
            assert client.post("/annotations", json=body).status_code == 200
            assert client.post("/annotations", json={**body, "error_type": "Y1"}).status_code == 200
            with SessionLocal() as session:
                labels = session.scalars(select(Annotation).where(Annotation.response_id == rid)).all()
                assert [(a.error_type, a.newman_stage) for a in labels] == [("Y1", None)]   # replaced

            after = client.get(url, params={"annotator": "human"}).json()
            assert (after["done"], after["remaining"], after["done_in_run"]) == (1, 1, 1)
            assert after["item"]["response_id"] != rid
            # filters narrow the queue; the run-wide count stays
            other = "P2" if question["subset"] != "P2" else "P1"
            filtered = client.get(url, params={"annotator": "human", "subset": other}).json()
            assert (filtered["remaining"], filtered["item"], filtered["done_in_run"]) == (0, None, 1)
            assert client.get(url, params={"annotator": "human", "strategy": "pot"}).json()["remaining"] == 0
            several = f"{other},{question['subset']}"   # comma-separated subsets, as in "P1,P3"
            assert client.get(url, params={"annotator": "human", "subset": several}).json()["remaining"] == 1
            assert client.get(url, params={"annotator": "human", "subset": question["subset"],
                                           "strategy": "cot"}).json()["remaining"] == 1
            assert client.get(url, params={"annotator": "llm-hakem"}).json()["done"] == 0   # labels per annotator
            assert client.post("/annotations", json={**body, "error_type": "H99"}).status_code == 422
            assert client.post("/annotations", json={**body, "newman_stage": "yok"}).status_code == 422
        finally:
            _remove_run("test-annotation")
