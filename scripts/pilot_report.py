"""Write a Markdown summary of one run: accuracy tables and every wrong answer.

Çalıştırma: python scripts/pilot_report.py --name pilot-1 > outputs/pilot-1.md
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from llmath.db import Question, Response, Run, SessionLocal, Verification  # noqa: E402
from llmath.service import summarize  # noqa: E402

STRATEGY = {"cot": "CoT", "pot": "PoT"}


def pct(correct: int, n: int) -> str:
    return f"%{100 * correct / n:.0f} ({correct}/{n})" if n else "—"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="çalıştırma adı")
    args = parser.parse_args()
    with SessionLocal() as session:
        run = session.scalars(select(Run).where(Run.name == args.name)).first()
        if run is None:
            sys.exit(f"çalıştırma bulunamadı: {args.name}")
        rows = summarize(session, run.id)
        wrong = session.execute(
            select(Response, Verification, Question)
            .join(Verification, Verification.response_id == Response.id)
            .join(Question, Question.id == Response.question_id)
            .where(Response.run_id == run.id, Verification.status != "correct", Response.error.is_(None))
            .order_by(Question.id, Response.model_key, Response.strategy)).all()

    totals: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    by_subset: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        totals[(r["model"], r["strategy"])][0] += r["correct"]
        totals[(r["model"], r["strategy"])][1] += r["n"]
        by_subset[(r["model"], r["strategy"], f"{r['subset']}-{r['lang']}")] = [r["correct"], r["n"]]
    columns = sorted({key[2] for key in by_subset})

    print(f"# Çalıştırma #{run.id}: {run.name}\n")
    print("| Model | Strateji | Toplam | " + " | ".join(columns) + " |")
    print("| --- | --- | --- | " + " | ".join("---" for _ in columns) + " |")
    for (model, strategy), (correct, n) in sorted(totals.items()):
        cells = [pct(*by_subset.get((model, strategy, c), [0, 0])) for c in columns]
        print(f"| {model} | {STRATEGY.get(strategy, strategy)} | {pct(correct, n)} | " + " | ".join(cells) + " |")

    print(f"\n## Yanlış ve doğrulanamayan cevaplar ({len(wrong)})\n")
    print("| Soru | Model | Strateji | Beklenen | Model cevabı | Karar | Katman |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for response, verification, question in wrong:
        answer = (response.extracted_answer or "—").replace("|", "\\|")
        print(f"| {question.id} | {response.model_key} | {STRATEGY.get(response.strategy, response.strategy)} | "
              f"{question.gold} | {answer} | {verification.status} | {verification.layer} |")


if __name__ == "__main__":
    main()
