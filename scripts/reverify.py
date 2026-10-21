"""Re-verify every answer of a run with the current sandbox and verifier.

Use it after a change to the sandbox or the verifier, so that all answers of a run are judged by
the same rules. Models are not called again: stored outputs and programs are reused. Programs that
the sandbox blocked or that timed out are executed again (a timeout depends on how busy the
machine was). Changed verdicts are written to outputs/reverify-<run>-<time>.csv for the record.

Çalıştırma: python scripts/reverify.py --name main-1 [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import select  # noqa: E402

from llmath.db import Question, Response, Run, SessionLocal, Verification  # noqa: E402
from llmath.sandbox import TIMEOUT_MESSAGE as TIMEOUT  # noqa: E402
from llmath.sandbox import run_code  # noqa: E402
from llmath.solver import program_answer  # noqa: E402
from llmath.verify import verify  # noqa: E402


def rerun(response: Response) -> None:
    result = run_code(response.code)
    response.extracted_answer, response.extraction_method = program_answer(result)
    response.exec_stdout, response.exec_error = result.stdout, result.stderr or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Bir çalıştırmanın cevaplarını güncel kodla yeniden doğrula")
    parser.add_argument("--name", required=True, help="çalıştırma adı")
    parser.add_argument("--dry-run", action="store_true", help="değişiklikleri göster ama kaydetme")
    args = parser.parse_args()

    with SessionLocal() as session, session.no_autoflush:
        run = session.scalars(select(Run).where(Run.name == args.name)).first()
        if run is None:
            sys.exit(f"çalıştırma bulunamadı: {args.name}")
        rows = session.execute(
            select(Response, Verification, Question)
            .join(Verification, Verification.response_id == Response.id)
            .join(Question, Question.id == Response.question_id)
            .where(Response.run_id == run.id, Response.error.is_(None))   # API failures have no answer
            .order_by(Response.id)).all()

        changes, reruns = [], 0
        for response, verification, question in rows:
            before = (verification.status, verification.layer, response.extracted_answer)
            if response.code and (response.extraction_method == "blocked" or response.exec_error == TIMEOUT):
                rerun(response)
                reruns += 1
            verdict = verify(question.gold, response.extracted_answer, question.answer_type)
            if (verdict.status, response.extracted_answer) != (before[0], before[2]):
                changes.append({"response_id": response.id, "question_id": question.id,
                                "model": response.model_key, "strategy": response.strategy,
                                "old_status": before[0], "new_status": verdict.status,
                                "old_layer": before[1], "new_layer": verdict.layer,
                                "old_answer": before[2], "new_answer": response.extracted_answer})
            verification.status, verification.layer, verification.detail = verdict.status, verdict.layer, verdict.detail
            verification.ambiguous_format, verification.normalized = verdict.ambiguous_format, verdict.normalized

        if args.dry_run:
            session.rollback()
        else:
            session.commit()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")   # every pass keeps its own record
    out = ROOT / "outputs" / f"reverify-{args.name}-{stamp}{'-onizleme' if args.dry_run else ''}.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(changes[0]) if changes else ["response_id"])
        writer.writeheader()
        writer.writerows(changes)
    print(f"{len(rows)} cevap yeniden doğrulandı, {reruns} program yeniden çalıştırıldı, "
          f"{len(changes)} karar ya da cevap değişti ({'kaydedilmedi' if args.dry_run else 'kaydedildi'}).")
    for (model, old, new), count in sorted(Counter((c["model"], c["old_status"], c["new_status"])
                                                   for c in changes).items()):
        print(f"  {model:18} {old} → {new}: {count}")
    print(f"ayrıntı: {out}")


if __name__ == "__main__":
    main()
