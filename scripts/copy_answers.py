"""Copy the answers to unchanged questions from one run into another (main-1 -> main-2).

main_v2 differs from main_v1 only in its re-translated Turkish P1/P2 items ("-tr2"); the English items and
the template items (P3) are identical. Their answers in main-1 are therefore copied into main-2 together with
their verdicts and labels (LLM judge and human), and only the new Turkish items are answered again. A
(question, model, strategy, prompt version) that already exists in the target run is skipped, so the copy
can be repeated safely, e.g. after a model has answered more English items in main-1.

Çalıştırma: python scripts/copy_answers.py --source main-1 --target main-2 --data data/main_v2.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from llmath.dataset import import_questions, read_jsonl  # noqa: E402
from llmath.db import Annotation, Response, Run, SessionLocal, Verification  # noqa: E402
from llmath.service import get_or_create_run  # noqa: E402

_SKIP = {"id", "run_id", "run", "question", "verification"}


def unchanged_ids(data: Path) -> set[str]:
    """Questions that read the same in both versions: everything but the re-translated "-tr2" items."""
    return {r["id"] for r in read_jsonl(data) if not r["id"].endswith("-tr2")}


def copy(session: Session, source: Run, target: Run, ids: set[str]) -> Counter:
    existing = {tuple(row) for row in session.execute(
        select(Response.question_id, Response.model_key, Response.strategy, Response.prompt_version)
        .where(Response.run_id == target.id))}
    copied: Counter = Counter()
    columns = [c.key for c in Response.__table__.columns if c.key not in _SKIP]
    for response in session.scalars(select(Response).where(Response.run_id == source.id, Response.error.is_(None),
                                                            Response.question_id.in_(ids))):
        key = (response.question_id, response.model_key, response.strategy, response.prompt_version)
        if key in existing:
            continue
        clone = Response(run_id=target.id, **{c: getattr(response, c) for c in columns})
        session.add(clone)
        session.flush()
        if response.verification is not None:
            v = response.verification
            session.add(Verification(response_id=clone.id, status=v.status, layer=v.layer, detail=v.detail,
                                     ambiguous_format=v.ambiguous_format, normalized=v.normalized))
        for a in session.scalars(select(Annotation).where(Annotation.response_id == response.id)):
            session.add(Annotation(response_id=clone.id, annotator=a.annotator, newman_stage=a.newman_stage,
                                   error_type=a.error_type, first_error_step=a.first_error_step, note=a.note,
                                   created_at=a.created_at))
        copied[(response.model_key, response.strategy)] += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Değişmeyen soruların cevaplarını başka bir çalıştırmaya kopyala")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--data", type=Path, required=True, help="hedef çalıştırmanın soru dosyası")
    args = parser.parse_args()
    import_questions(args.data)   # the -tr2 items must exist as questions
    with SessionLocal() as session:
        source = session.scalars(select(Run).where(Run.name == args.source)).first()
        if source is None:
            sys.exit(f"çalıştırma bulunamadı: {args.source}")
        target = get_or_create_run(session, args.target, {"data": args.data.name, "copied_from": args.source})
        copied = copy(session, source, target, unchanged_ids(args.data))
        session.commit()
    for (model, strategy), n in sorted(copied.items()):
        print(f"  {model:<22} {strategy:<4} {n} cevap kopyalandı")
    print(f"toplam {sum(copied.values())} cevap {args.source} → {args.target}")


if __name__ == "__main__":
    main()
