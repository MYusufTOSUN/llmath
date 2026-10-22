"""Write a Markdown report of one run.

Sections: accuracy with 95% intervals, paired Turkish–English and CoT–PoT comparisons (exact
McNemar), accuracy by MATH level, which verifier layer decided, and every wrong answer.

Çalıştırma: python scripts/report.py --name main-1 --cascade qwen3-8b gemini-flash-lite --out outputs/main-1.md
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from llmath.cascade import mcnemar_exact, run_cascade, to_markdown, wilson  # noqa: E402
from llmath.db import Question, Response, Run, SessionLocal, Verification  # noqa: E402

STRATEGY = {"cot": "CoT", "pot": "PoT"}


def pct(correct: int, n: int) -> str:
    return f"%{100 * correct / n:.0f} ({correct}/{n})" if n else "—"


def interval(values: list[bool]) -> str:
    low, high = wilson(sum(values), len(values))
    return f"%{100 * low:.0f}–{100 * high:.0f}"


def p_value(p: float) -> str:
    return "<0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


def accuracy_table(rows) -> list[str]:
    groups: dict[tuple, list[bool]] = defaultdict(list)
    for response, verification, question in rows:
        ok = verification.status == "correct"
        groups[(response.model_key, response.strategy, "Toplam")].append(ok)
        groups[(response.model_key, response.strategy, f"{question.subset}-{question.lang}")].append(ok)
    columns = sorted({key[2] for key in groups} - {"Toplam"})
    lines = ["## Doğruluk", "", "| Model | Strateji | Toplam | %95 GA | " + " | ".join(columns) + " |",
             "| --- | --- | --- | --- | " + " | ".join("---" for _ in columns) + " |"]
    for model, strategy in sorted({key[:2] for key in groups}):
        total = groups[(model, strategy, "Toplam")]
        cells = [pct(sum(v), len(v)) for v in (groups.get((model, strategy, c), []) for c in columns)]
        lines.append(f"| {model} | {STRATEGY.get(strategy, strategy)} | {pct(sum(total), len(total))} | "
                     f"{interval(total)} | " + " | ".join(cells) + " |")
    return lines


def language_table(rows) -> list[str]:
    """The same problem in English and Turkish: does the language change the outcome?"""
    pairs: dict[tuple, dict[str, bool]] = defaultdict(dict)
    for response, verification, question in rows:
        if question.pair_id:
            key = (response.model_key, response.strategy, question.subset, question.pair_id)
            pairs[key][question.lang] = verification.status == "correct"
    groups: dict[tuple, tuple[list[bool], list[bool]]] = defaultdict(lambda: ([], []))
    for (model, strategy, subset, _), langs in pairs.items():
        if "en" in langs and "tr" in langs:
            groups[(model, strategy, subset)][0].append(langs["en"])
            groups[(model, strategy, subset)][1].append(langs["tr"])
    if not groups:
        return []
    lines = ["## Türkçe–İngilizce (aynı soru, eşleştirilmiş)", "",
             "| Model | Strateji | Alt küme | Çift | İngilizce | Türkçe | Fark (puan) | Yalnız EN doğru | "
             "Yalnız TR doğru | McNemar p |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for (model, strategy, subset), (en, tr) in sorted(groups.items()):
        only_en, only_tr, p = mcnemar_exact(en, tr)
        diff = 100 * (sum(tr) - sum(en)) / len(en)
        lines.append(f"| {model} | {STRATEGY.get(strategy, strategy)} | {subset} | {len(en)} | "
                     f"{pct(sum(en), len(en))} | {pct(sum(tr), len(tr))} | {f'{diff:+.1f}'.replace('.', ',')} | "
                     f"{only_en} | {only_tr} | {p_value(p)} |")
    return lines


def strategy_table(rows) -> list[str]:
    """CoT and PoT on the same question."""
    by_question: dict[tuple, dict[str, bool]] = defaultdict(dict)
    for response, verification, question in rows:
        by_question[(response.model_key, question.lang, question.id)][response.strategy] = \
            verification.status == "correct"
    groups: dict[tuple, tuple[list[bool], list[bool]]] = defaultdict(lambda: ([], []))
    for (model, lang, _), strategies in by_question.items():
        if "cot" in strategies and "pot" in strategies:
            groups[(model, lang)][0].append(strategies["cot"])
            groups[(model, lang)][1].append(strategies["pot"])
    if not groups:
        return []
    lines = ["## CoT–PoT (aynı soru, eşleştirilmiş)", "",
             "| Model | Dil | Soru | CoT | PoT | Yalnız CoT doğru | Yalnız PoT doğru | McNemar p |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for (model, lang), (cot, pot) in sorted(groups.items()):
        only_cot, only_pot, p = mcnemar_exact(cot, pot)
        lines.append(f"| {model} | {lang} | {len(cot)} | {pct(sum(cot), len(cot))} | {pct(sum(pot), len(pot))} | "
                     f"{only_cot} | {only_pot} | {p_value(p)} |")
    return lines


def level_table(rows) -> list[str]:
    groups: dict[tuple, list[bool]] = defaultdict(list)
    for response, verification, question in rows:
        if question.difficulty:
            groups[(response.model_key, response.strategy, question.lang, question.difficulty)].append(
                verification.status == "correct")
    if not groups:
        return []
    levels = sorted({key[3] for key in groups})
    lines = ["## MATH düzeyine göre (P2)", "",
             "| Model | Strateji | Dil | " + " | ".join(f"Düzey {level}" for level in levels) + " |",
             "| --- | --- | --- | " + " | ".join("---" for _ in levels) + " |"]
    for model, strategy, lang in sorted({key[:3] for key in groups}):
        cells = [pct(sum(v), len(v)) for v in (groups.get((model, strategy, lang, lv), []) for lv in levels)]
        lines.append(f"| {model} | {STRATEGY.get(strategy, strategy)} | {lang} | " + " | ".join(cells) + " |")
    return lines


def verifier_table(rows) -> list[str]:
    layers: dict[tuple, Counter] = defaultdict(Counter)
    extra: dict[tuple, Counter] = defaultdict(Counter)
    for response, verification, _ in rows:
        key = (response.model_key, response.strategy)
        layers[key][f"{verification.status}/{verification.layer}"] += 1
        extra[key]["belirsiz biçim"] += verification.ambiguous_format
        extra[key]["token sınırında kesilen"] += response.finish_reason == "length"
    names = sorted({name for counter in layers.values() for name in counter})
    lines = ["## Kararı veren doğrulama katmanı", "",
             "| Model | Strateji | " + " | ".join(names) + " | Belirsiz biçim | Token sınırında kesilen |",
             "| --- | --- | " + " | ".join("---" for _ in names) + " | --- | --- |"]
    for key in sorted(layers):
        cells = [str(layers[key][name]) for name in names]
        lines.append(f"| {key[0]} | {STRATEGY.get(key[1], key[1])} | " + " | ".join(cells) +
                     f" | {extra[key]['belirsiz biçim']} | {extra[key]['token sınırında kesilen']} |")
    return lines


def wrong_answers(rows) -> list[str]:
    wrong = [(r, v, q) for r, v, q in rows if v.status != "correct"]
    lines = [f"## Yanlış ve doğrulanamayan cevaplar ({len(wrong)})", "",
             "| Soru | Model | Strateji | Beklenen | Model cevabı | Karar | Katman |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for response, verification, question in wrong:
        answer = (response.extracted_answer or "—").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {question.id} | {response.model_key} | {STRATEGY.get(response.strategy, response.strategy)} | "
                     f"{question.gold} | {answer} | {verification.status} | {verification.layer} |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="çalıştırma adı")
    parser.add_argument("--cascade", nargs=2, metavar=("UCUZ", "GÜÇLÜ"),
                        help="kademeli çözüm analizini de ekle (ör. qwen3-8b gemini-flash-lite)")
    parser.add_argument("--out", type=Path,
                        help="UTF-8 Markdown dosyası (PowerShell'in > yönlendirmesi UTF-16 yazar)")
    args = parser.parse_args()
    with SessionLocal() as session:
        run = session.scalars(select(Run).where(Run.name == args.name)).first()
        if run is None:
            sys.exit(f"çalıştırma bulunamadı: {args.name}")
        rows = session.execute(
            select(Response, Verification, Question)
            .join(Verification, Verification.response_id == Response.id)
            .join(Question, Question.id == Response.question_id)
            .where(Response.run_id == run.id, Response.error.is_(None))   # API failures say nothing
            .order_by(Question.id, Response.model_key, Response.strategy)).all()
        cascade = to_markdown(run_cascade(session, run.id, *args.cascade)) if args.cascade else None

    parts = [f"# Çalıştırma #{run.id}: {run.name}",
             f"Veri: {run.config.get('data', '—')}, istem sürümü: {run.config.get('prompt_version', '—')}, "
             f"cevap sayısı: {len(rows)}"]
    for section in (accuracy_table(rows), language_table(rows), strategy_table(rows), level_table(rows),
                    verifier_table(rows)):
        if section:
            parts.append("\n".join(section))
    if cascade:
        parts.append(cascade)
    parts.append("\n".join(wrong_answers(rows)))
    text = "\n\n".join(parts) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"rapor yazıldı: {args.out}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
