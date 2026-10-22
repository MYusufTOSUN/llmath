"""Which questions of a data file each model has answered in a run, by subset and language.

Shows answered/expected per model, strategy, subset and language, and counts answers to Turkish P1/P2
questions that are not the re-translated "-tr2" items (machine translations), which must be zero in main-2.
Exit code 1 when something is missing or an old translation is found, so the check can gate a report.

Çalıştırma: python scripts/check_coverage.py --name main-2 --data data/main_v2.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import select  # noqa: E402

from llmath.dataset import read_jsonl  # noqa: E402
from llmath.db import Question, Response, Run, SessionLocal  # noqa: E402


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Çalıştırmanın soru kapsamı")
    parser.add_argument("--name", required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.data)
    with SessionLocal() as session:
        run = session.scalars(select(Run).where(Run.name == args.name)).first()
        if run is None:
            sys.exit(f"çalıştırma bulunamadı: {args.name}")
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
        flag = "" if got == expected else "  ← eksik"
        missing += expected - got
        print(f"{model:<22}{strategy:<9}{group:<8}{got}/{expected}{flag}")
    old_counts = Counter(old)
    print(f"\neski çeviriyle verilmiş Türkçe P1/P2 cevabı: {len(old)} {dict(old_counts) if old else ''}")
    print(f"eksik cevap: {missing}")
    if missing or old:
        sys.exit(1)


if __name__ == "__main__":
    main()
