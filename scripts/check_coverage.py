"""Which questions of a data file each model has answered in a run, by subset and language.

Shows answered/expected per model, strategy, subset and language, and counts answers to Turkish P1/P2
questions that are not the re-translated "-tr2" items (machine translations), which must be zero in main-2.
It also counts answers given to a text that has changed since, e.g. when the author corrects a translation
after a model has answered it (main_v2 rebuilt and imported again); --drop-stale deletes them with their
verdicts and labels, so that the next run asks the new text. Exit code 1 when something is missing, an old
translation is found or an answer is stale, so the check can gate a report.

Çalıştırma: python scripts/check_coverage.py --name main-2 --data data/main_v2.jsonl [--drop-stale]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from llmath.config import OUT_OF_SCOPE  # noqa: E402
from llmath.dataset import read_jsonl  # noqa: E402
from llmath.db import Annotation, Question, Response, Run, SessionLocal  # noqa: E402


def coverage(rows: list[dict], answered: set[tuple[str, str, str]]) -> list[tuple]:
    """(model, strategy, group, answered, expected) for every model and strategy found in the run."""
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups[f"{row['subset']}-{row['lang']}"].add(row["id"])
    combos = sorted({(model, strategy) for _, model, strategy in answered})
    table = []
    for model, strategy in combos:
        mine = {qid for qid, m, s in answered if (m, s) == (model, strategy)}
        for group, ids in sorted(groups.items()):
            table.append((model, strategy, group, len(ids & mine), len(ids)))
    return table


def status(model: str, group: str, got: int, expected: int) -> str:
    """Empty when complete; "kapsam dışı" for a part the model is not run on by decision (OUT_OF_SCOPE)."""
    subset, lang = group.split("-")
    if got < expected and (model, subset, lang) in OUT_OF_SCOPE:
        return "kapsam dışı"
    return "" if got == expected else "← eksik"


def stale_answers(session: Session, run_id: int) -> list[Response]:
    """Answers whose question text has changed since they were given."""
    return list(session.scalars(select(Response).join(Question, Question.id == Response.question_id)
                                .where(Response.run_id == run_id, Response.question_text != Question.text)))


def drop(session: Session, responses: list[Response]) -> None:
    for response in responses:
        for a in session.scalars(select(Annotation).where(Annotation.response_id == response.id)):
            session.delete(a)
        if response.verification is not None:
            session.delete(response.verification)
        session.delete(response)


def main() -> None:
    parser = argparse.ArgumentParser(description="Çalıştırmanın soru kapsamı")
    parser.add_argument("--name", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--drop-stale", action="store_true",
                        help="metni sonradan değişmiş sorulara verilmiş cevapları karar ve etiketleriyle sil")
    args = parser.parse_args()
    rows = read_jsonl(args.data)
    with SessionLocal() as session:
        run = session.scalars(select(Run).where(Run.name == args.name)).first()
        if run is None:
            sys.exit(f"çalıştırma bulunamadı: {args.name}")
        stale = stale_answers(session, run.id)
        stale_keys = sorted((r.question_id, r.model_key, r.strategy) for r in stale)
        if stale and args.drop_stale:
            drop(session, stale)
            session.commit()
            print(f"metni değişmiş sorulara verilmiş {len(stale_keys)} cevap silindi: {stale_keys}")
            stale_keys = []
        answered = {tuple(r) for r in session.execute(
            select(Response.question_id, Response.model_key, Response.strategy)
            .where(Response.run_id == run.id, Response.error.is_(None)))}
        old = session.execute(
            select(Response.model_key, Response.strategy).join(Question, Question.id == Response.question_id)
            .where(Response.run_id == run.id, Question.lang == "tr", Question.subset.in_(["P1", "P2"]),
                   ~Question.id.like("%-tr2"))).all()
    table = coverage(rows, answered)
    missing = 0
    print(f"{'model':<22}{'strateji':<9}{'grup':<8}cevaplanan/beklenen")
    for model, strategy, group, got, expected in table:
        flag = status(model, group, got, expected)
        if flag == "← eksik":
            missing += expected - got
        print(f"{model:<22}{strategy:<9}{group:<8}{got}/{expected}  {flag}".rstrip())
    old_counts = Counter(old)
    print(f"\neski çeviriyle verilmiş Türkçe P1/P2 cevabı: {len(old)} {dict(old_counts) if old else ''}")
    print(f"eksik cevap: {missing}")
    print(f"metni sonradan değişmiş soruya verilmiş cevap: {len(stale_keys)} {stale_keys[:10] if stale_keys else ''}")
    if missing or old or stale_keys:
        sys.exit(1)


if __name__ == "__main__":
    main()
