"""The old machine translation against the new translation of the Turkish items, answer by answer.

main-1 holds the answers to the machine-translated Turkish of main_v1 (GSM8K-TR, MCLM MT-MATH100); main-2 holds
the answers of the same models to the re-translated items ("-tr2", see llmath/translation_v2.py). Items are
paired by their pair id, and every model and strategy that answered both versions is compared with the exact
McNemar test. Only the Turkish P1/P2 items differ between the two runs.

Çalıştırma: python scripts/compare_translations.py --old main-1 --new main-2 --out outputs/ceviri-karsilastirma.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from llmath.cascade import mcnemar_exact  # noqa: E402
from llmath.db import Question, Response, Run, SessionLocal, Verification  # noqa: E402

STRATEGY = {"cot": "CoT", "pot": "PoT"}


def turkish_outcomes(session, run_name: str) -> dict[tuple[str, str, str, str], bool]:
    """{(model, strategy, subset, pair id): correct} for the translated Turkish items (P1, P2) of a run."""
    run = session.scalars(select(Run).where(Run.name == run_name)).one()
    rows = session.execute(select(Response.model_key, Response.strategy, Question.subset, Question.pair_id,
                                  Verification.status)
                           .join(Question, Question.id == Response.question_id)
                           .join(Verification, Verification.response_id == Response.id)
                           .where(Response.run_id == run.id, Response.error.is_(None), Question.lang == "tr",
                                  Question.subset.in_(["P1", "P2"])))
    return {(model, strategy, subset, pair): status == "correct" for model, strategy, subset, pair, status in rows}


def compare(old: dict, new: dict) -> list[dict]:
    rows = []
    for model, strategy, subset in sorted({key[:3] for key in old} & {key[:3] for key in new}):
        pairs = sorted(key[3] for key in new if key[:3] == (model, strategy, subset) and key in old)
        before = [old[(model, strategy, subset, p)] for p in pairs]
        after = [new[(model, strategy, subset, p)] for p in pairs]
        only_old, only_new, p = mcnemar_exact(before, after)
        rows.append({"model": model, "strategy": strategy, "subset": subset, "n": len(pairs), "old": sum(before),
                     "new": sum(after), "only_old": only_old, "only_new": only_new, "p": p})
    return rows


def to_markdown(rows: list[dict], old_name: str, new_name: str) -> str:
    lines = [f"# Eski makine çevirisi ({old_name}) ve yeni çeviri ({new_name})", "",
             "Aynı Türkçe sorular, aynı modeller ve stratejiler; McNemar kesin testi.", "",
             "| Model | Strateji | Alt küme | Soru | Eski çeviri | Yeni çeviri | Fark (puan) | Yalnız eskide doğru "
             "| Yalnız yenide doğru | McNemar p |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        diff = f"{100 * (r['new'] - r['old']) / r['n']:+.1f}".replace(".", ",")
        p = "<0,001" if r["p"] < 0.001 else f"{r['p']:.3f}".replace(".", ",")
        lines.append(f"| {r['model']} | {STRATEGY[r['strategy']]} | {r['subset']} | {r['n']} | {r['old']} | {r['new']} "
                     f"| {diff} | {r['only_old']} | {r['only_new']} | {p} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Eski ve yeni Türkçe çevirinin cevap cevap karşılaştırması")
    parser.add_argument("--old", default="main-1")
    parser.add_argument("--new", default="main-2")
    parser.add_argument("--out", type=Path, default=Path("outputs/ceviri-karsilastirma.md"))
    args = parser.parse_args()
    with SessionLocal() as session:
        rows = compare(turkish_outcomes(session, args.old), turkish_outcomes(session, args.new))
    args.out.write_text(to_markdown(rows, args.old, args.new), encoding="utf-8")
    print(f"rapor yazıldı: {args.out}")


if __name__ == "__main__":
    main()
