import json
import sys
from pathlib import Path

import pytest
from sqlalchemy import select

from llmath.db import Annotation, Question, Response, Run, SessionLocal, Verification, init_db

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import check_coverage  # noqa: E402
import copy_answers  # noqa: E402

QUESTIONS = [("v2t-p1-1-en", "P1", "en"), ("v2t-p1-1-tr", "P1", "tr"), ("v2t-p1-1-tr2", "P1", "tr"),
             ("v2t-p3-1", "P3", "tr")]


@pytest.fixture
def source_run(tmp_path):
    init_db()
    with SessionLocal() as s:
        for qid, subset, lang in QUESTIONS:
            s.merge(Question(id=qid, subset=subset, lang=lang, pair_id="v2t-p1-1" if subset == "P1" else None,
                             text=qid, gold="1", answer_type="integer", source="test", license="CC0"))
        run = Run(name="test-v2-source", config={})
        s.add(run)
        s.flush()
        for qid in ("v2t-p1-1-en", "v2t-p1-1-tr", "v2t-p3-1"):   # main-1 knows only the old Turkish
            r = Response(run_id=run.id, question_id=qid, question_text=qid, model_key="qwen3-8b", strategy="cot",
                         prompt_version="v0.1", extracted_answer="1", extraction_method="boxed", raw_output="1")
            s.add(r)
            s.flush()
            s.add(Verification(response_id=r.id, status="correct", layer="exact"))
            s.add(Annotation(response_id=r.id, annotator="human", error_type="H1"))
        s.commit()
    data = tmp_path / "v2.jsonl"
    data.write_text("\n".join(json.dumps({"id": q, "subset": sub, "lang": lang}) for q, sub, lang in QUESTIONS
                              if q != "v2t-p1-1-tr"), encoding="utf-8")
    yield data
    with SessionLocal() as s:
        for run in s.scalars(select(Run).where(Run.name.like("test-v2-%"))):
            for r in run.responses:
                for a in s.scalars(select(Annotation).where(Annotation.response_id == r.id)):
                    s.delete(a)
                if r.verification is not None:
                    s.delete(r.verification)
                s.delete(r)
            s.delete(run)
        for qid, _, _ in QUESTIONS:
            if (q := s.get(Question, qid)) is not None:
                s.delete(q)
        s.commit()


def test_only_unchanged_answers_are_copied_with_their_labels_and_only_once(source_run):
    assert copy_answers.unchanged_ids(source_run) == {"v2t-p1-1-en", "v2t-p3-1"}
    ids = copy_answers.unchanged_ids(source_run)
    with SessionLocal() as s:
        source = s.scalars(select(Run).where(Run.name == "test-v2-source")).one()
        target = Run(name="test-v2-target", config={})
        s.add(target)
        s.flush()
        first = copy_answers.copy(s, source, target, ids)
        english = s.scalars(select(Response).where(Response.run_id == source.id,
                                                   Response.question_id == "v2t-p1-1-en")).one()
        s.add(Annotation(response_id=english.id, annotator="llm:judge/j0.5", error_type="H4"))  # labelled later
        s.flush()
        again = copy_answers.copy(s, source, target, ids)
        third = copy_answers.copy(s, source, target, ids)
        s.commit()
        rows = s.scalars(select(Response).where(Response.run_id == target.id)).all()
        assert (sum(first[0].values()), first[1]) == (2, 2)
        assert (sum(again[0].values()), again[1]) == (0, 1)     # the later label follows, the answer does not
        assert (sum(third[0].values()), third[1]) == (0, 0)
        assert sorted(r.question_id for r in rows) == ["v2t-p1-1-en", "v2t-p3-1"]   # never the old Turkish
        assert all(r.verification.status == "correct" for r in rows)
        labels = s.scalars(select(Annotation).where(Annotation.response_id.in_([r.id for r in rows]))).all()
        assert sorted(a.annotator for a in labels) == ["human", "human", "llm:judge/j0.5"]


def test_coverage_shows_the_new_turkish_still_missing(source_run):
    rows = [json.loads(line) for line in source_run.read_text(encoding="utf-8").splitlines()]
    answered = {("v2t-p1-1-en", "qwen3-8b", "cot"), ("v2t-p3-1", "qwen3-8b", "cot")}
    table = {group: (got, expected) for _, _, group, got, expected in check_coverage.coverage(rows, answered)}
    assert table == {"P1-en": (1, 1), "P1-tr": (0, 1), "P3-tr": (1, 1)}
