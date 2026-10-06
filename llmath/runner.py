"""Batch experiment runner.

Örnek:
    python -m llmath.runner --name pilot-1 --models qwen3-8b --strategies cot pot
A run can be resumed with the same --name; finished combinations are skipped.
"""

from __future__ import annotations

import argparse
import sys
import time

from sqlalchemy import select

from .config import MODELS
from .dataset import import_questions
from .db import Question, Response, SessionLocal
from .service import get_or_create_run, model_status, record, solve_and_verify, summarize

_MARK = {"correct": "✓", "incorrect": "✗", "unparsed": "?"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLMath toplu deney çalıştırıcı")
    parser.add_argument("--name", default="pilot", help="çalıştırma adı (aynı adla devam edilir)")
    parser.add_argument("--models", nargs="+", default=["qwen3-8b"], choices=sorted(MODELS))
    parser.add_argument("--strategies", nargs="+", default=["cot", "pot"], choices=["cot", "pot"])
    parser.add_argument("--subsets", nargs="+", default=["P1", "P2", "P3"])
    parser.add_argument("--langs", nargs="+", default=["tr", "en"])
    parser.add_argument("--limit", type=int, help="her model ve strateji için en fazla soru sayısı")
    parser.add_argument("--pause", type=float, default=4.0,
                        help="bulut modellerinde çağrılar arası bekleme (sn); ücretsiz katman dakika sınırı için")
    args = parser.parse_args(argv)

    import_questions()
    with SessionLocal() as session:
        run = get_or_create_run(session, args.name, {"models": args.models, "strategies": args.strategies,
                                                     "subsets": args.subsets, "langs": args.langs})
        session.commit()
        questions = list(session.scalars(
            select(Question).where(Question.subset.in_(args.subsets), Question.lang.in_(args.langs))
            .order_by(Question.subset, Question.id)))
        if args.limit:
            questions = questions[: args.limit]
        # Calls that failed at the API (rate limits, outages) are retried on the next run.
        failed = list(session.scalars(select(Response).where(Response.run_id == run.id,
                                                             Response.error.is_not(None))))
        for row in failed:
            if row.verification is not None:
                session.delete(row.verification)
            session.delete(row)
        session.commit()
        done = {tuple(row) for row in session.execute(
            select(Response.question_id, Response.model_key, Response.strategy)
            .where(Response.run_id == run.id))}

        for model_key in args.models:
            spec = MODELS[model_key]
            ok, reason = model_status(spec)
            if not ok:
                print(f"! {model_key} atlandı: {reason}", file=sys.stderr)
                continue
            for strategy in args.strategies:
                todo = [q for q in questions if (q.id, model_key, strategy) not in done]
                for i, question in enumerate(todo, start=1):
                    if spec.provider != "ollama" and i > 1:
                        time.sleep(args.pause)
                    started = time.perf_counter()
                    result, verdict = solve_and_verify(question.text, question.gold, spec, strategy,
                                                       question.lang, question.answer_type)
                    record(session, run, question.text, question.id, spec, result, verdict)
                    session.commit()
                    status = verdict.status if verdict else "unparsed"
                    print(f"[{i}/{len(todo)}] {model_key} {strategy} {question.id} {_MARK[status]} "
                          f"({verdict.layer if verdict else '-'}) cevap={result.answer!r} "
                          f"{time.perf_counter() - started:.1f} sn", flush=True)

        print(f"\nÇalıştırma #{run.id} ({run.name}) özeti:")
        for row in summarize(session, run.id):
            print(f"  {row['model']:<18} {row['strategy']:<4} {row['subset']} {row['lang']}  "
                  f"{row['correct']}/{row['n']} doğru (%{100 * row['accuracy']:.0f}), "
                  f"ayrıştırılamayan {row['unparsed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
