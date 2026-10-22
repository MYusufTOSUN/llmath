"""Copy the answers to unchanged questions from one run into another (main-1 -> main-2).

main_v2 differs from main_v1 only in its re-translated Turkish P1/P2 items ("-tr2"); the English items and
the template items (P3) are identical. Their answers in main-1 are therefore copied into main-2 together with
their verdicts and labels (LLM judge and human), and only the new Turkish items are answered again. A
(question, model, strategy, prompt version) that already exists in the target run is not copied again; it
only gets the labels it lacks. The copy can therefore be repeated safely, and should be repeated after a
model has answered more English items in main-1 or a judge has labelled more of its errors there.

A re-translated item whose text is byte for byte the old machine translation (only m-p2-069) is the same
input to a model, so its main-1 answers to the old item are copied too, under the new id. Items that only
mean the same are answered again (the author's decision of 2026-10-22): models read text, not meaning.

Çalıştırma: python scripts/copy_answers.py --source main-1 --target main-2 --data data/main_v2.jsonl
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

from llmath.dataset import import_questions, read_jsonl  # noqa: E402
from llmath.db import Annotation, Question, Response, Run, SessionLocal, Verification  # noqa: E402
from llmath.service import get_or_create_run  # noqa: E402

_SKIP = {"id", "run_id", "run", "question", "verification"}


def unchanged_ids(data: Path) -> set[str]:
    """Questions that read the same in both versions: everything but the re-translated "-tr2" items."""
    return {r["id"] for r in read_jsonl(data) if not r["id"].endswith("-tr2")}


def same_text_ids(session: Session, data: Path) -> dict[str, str]:
    """{old id: new id} for the re-translated items whose text equals the old translation byte for byte."""
    aliases = {}
    for row in read_jsonl(data):
        if row["id"].endswith("-tr2"):
            old = session.get(Question, row["id"][:-1])
            if old is not None and old.text == row["text"]:
                aliases[old.id] = row["id"]
    return aliases


def _labels(session: Session, run: Run) -> dict[int, list[Annotation]]:
    labels: dict[int, list[Annotation]] = defaultdict(list)
    for a in session.scalars(select(Annotation).join(Response, Response.id == Annotation.response_id)
                             .where(Response.run_id == run.id)):
        labels[a.response_id].append(a)
    return labels


def copy(session: Session, source: Run, target: Run, ids: set[str],
         aliases: dict[str, str] | None = None) -> tuple[Counter, int]:
    """Copied answers per (model, strategy), and the number of labels added. An answer to an id in `aliases`
    is copied under the new id. A label is identified by its annotator and time, which the copy keeps."""
    aliases = aliases or {}
    clones = {(r.question_id, r.model_key, r.strategy, r.prompt_version): r
              for r in session.scalars(select(Response).where(Response.run_id == target.id))}
    source_labels, target_labels = _labels(session, source), _labels(session, target)
    copied: Counter = Counter()
    added = 0
    columns = [c.key for c in Response.__table__.columns if c.key not in _SKIP]
    for response in session.scalars(select(Response).where(Response.run_id == source.id, Response.error.is_(None),
                                                            Response.question_id.in_(set(ids) | set(aliases)))):
        question_id = aliases.get(response.question_id, response.question_id)
        key = (question_id, response.model_key, response.strategy, response.prompt_version)
        clone = clones.get(key)
        if clone is None:
            clone = Response(run_id=target.id,
                             **{c: getattr(response, c) for c in columns} | {"question_id": question_id})
            session.add(clone)
            session.flush()
            if response.verification is not None:
                v = response.verification
                session.add(Verification(response_id=clone.id, status=v.status, layer=v.layer, detail=v.detail,
                                         ambiguous_format=v.ambiguous_format, normalized=v.normalized))
            copied[(response.model_key, response.strategy)] += 1
        have = {(a.annotator, a.created_at) for a in target_labels.get(clone.id, [])}
        for a in source_labels.get(response.id, []):
            if (a.annotator, a.created_at) not in have:
                session.add(Annotation(response_id=clone.id, annotator=a.annotator, newman_stage=a.newman_stage,
                                       error_type=a.error_type, first_error_step=a.first_error_step, note=a.note,
                                       created_at=a.created_at))
                added += 1
    return copied, added


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
        aliases = same_text_ids(session, args.data)
        copied, added = copy(session, source, target, unchanged_ids(args.data), aliases)
        session.commit()
    for (model, strategy), n in sorted(copied.items()):
        print(f"  {model:<22} {strategy:<4} {n} cevap kopyalandı")
    print(f"toplam {sum(copied.values())} cevap ve {added} etiket {args.source} → {args.target}")
    print(f"metni eski çeviriyle birebir aynı yeni soru: {', '.join(sorted(aliases.values())) or 'yok'}")


if __name__ == "__main__":
    main()
