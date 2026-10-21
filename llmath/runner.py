"""Batch experiment runner.

Örnek:
    python -m llmath.runner --name pilot-1 --models qwen3-8b --strategies cot pot
    python -m llmath.runner --name main-1 --data data/main_v1.jsonl --models gemini-flash-lite
    python -m llmath.runner --name numfmt-1-trnote --data data/number_format_v1.jsonl \
        --models gemini-flash-lite --strategies pot --langs tr --prompt-variant trnote
A run can be resumed with the same --name; finished combinations are skipped.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from sqlalchemy import select

from .config import MODELS, PILOT_FILE
from .dataset import import_questions
from .db import Question, Response, SessionLocal
from .prompts import VARIANTS, prompt_version_for
from .service import get_or_create_run, model_status, record, solve_and_verify, summarize

_MARK = {"correct": "✓", "incorrect": "✗", "unparsed": "?"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLMath toplu deney çalıştırıcı")
    parser.add_argument("--name", default="pilot", help="çalıştırma adı (aynı adla devam edilir)")
    parser.add_argument("--data", type=Path, default=PILOT_FILE,
                        help="soru dosyası (JSONL); yalnız bu dosyadaki sorular çözülür")
    parser.add_argument("--models", nargs="+", default=["qwen3-8b"], choices=sorted(MODELS))
    parser.add_argument("--strategies", nargs="+", default=["cot", "pot"], choices=["cot", "pot"])
    parser.add_argument("--subsets", nargs="+", help="yalnız bu alt setler (varsayılan: dosyadaki hepsi)")
    parser.add_argument("--langs", nargs="+", default=["tr", "en"])
    parser.add_argument("--limit", type=int, help="her model ve strateji için en fazla soru sayısı")
    parser.add_argument("--pause", type=float, default=4.0,
                        help="bulut modellerinde çağrılar arası bekleme (sn); ücretsiz katman dakika sınırı için")
    parser.add_argument("--max-errors", type=int, default=5,
                        help="art arda bu kadar API hatasında model durdurulur (ör. günlük kota doldu)")
    parser.add_argument("--prompt-variant", default="base", choices=VARIANTS,
                        help="PoT istem varyantı (sayı yazımı deneyi için); her varyant ayrı --name ile çalıştırılmalı")
    args = parser.parse_args(argv)
    version = prompt_version_for(args.prompt_variant)

    ids = import_questions(args.data)
    with SessionLocal() as session:
        run = get_or_create_run(session, args.name, {"data": args.data.name, "models": args.models,
                                                     "strategies": args.strategies, "prompt_version": version,
                                                     "subsets": args.subsets, "langs": args.langs})
        session.commit()
        query = select(Question).where(Question.id.in_(ids), Question.lang.in_(args.langs))
        if args.subsets:
            query = query.where(Question.subset.in_(args.subsets))
        questions = list(session.scalars(query.order_by(Question.subset, Question.id)))
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
            select(Response.question_id, Response.model_key, Response.strategy, Response.prompt_version)
            .where(Response.run_id == run.id))}

        for model_key in args.models:
            spec = MODELS[model_key]
            ok, reason = model_status(spec)
            if not ok:
                print(f"! {model_key} atlandı: {reason}", file=sys.stderr)
                continue
            errors = 0   # consecutive API failures; a full daily quota fails every call that follows
            for strategy in args.strategies:
                todo = [q for q in questions if (q.id, model_key, strategy, version) not in done]
                for i, question in enumerate(todo, start=1):
                    if spec.provider != "ollama" and i > 1:
                        time.sleep(args.pause)
                    started = time.perf_counter()
                    result, verdict = solve_and_verify(question.text, question.gold, spec, strategy,
                                                       question.lang, question.answer_type, args.prompt_variant)
                    record(session, run, question.text, question.id, spec, result, verdict)
                    session.commit()
                    errors = errors + 1 if result.error else 0
                    status = verdict.status if verdict else "unparsed"
                    print(f"[{i}/{len(todo)}] {model_key} {strategy} {question.id} {_MARK[status]} "
                          f"({verdict.layer if verdict else '-'}) cevap={result.answer!r} "
                          f"{time.perf_counter() - started:.1f} sn", flush=True)
                    if errors >= args.max_errors:
                        break
                if errors >= args.max_errors:
                    print(f"! {model_key} durduruldu: art arda {errors} API hatası (kota dolmuş olabilir); "
                          f"aynı komutla sonra devam edin", file=sys.stderr)
                    break

        print(f"\nÇalıştırma #{run.id} ({run.name}) özeti:")
        for row in summarize(session, run.id):
            print(f"  {row['model']:<18} {row['strategy']:<4} {row['subset']} {row['lang']}  "
                  f"{row['correct']}/{row['n']} doğru (%{100 * row['accuracy']:.0f}), "
                  f"ayrıştırılamayan {row['unparsed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
