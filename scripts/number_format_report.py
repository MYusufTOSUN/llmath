"""Markdown report of the number-format experiment (data/number_format_v1.jsonl).

Conditions: CoT and PoT with the base prompt (run numfmt-1), PoT without the Rational hint
(numfmt-1-norational) and PoT with the Turkish number note (numfmt-1-trnote, Turkish questions
only). The report compares the conditions on the same questions and the Turkish and English
versions of each problem (exact McNemar), and searches every program for the two format
misreadings: a Turkish decimal split at its comma (8,4 -> Rational(8, 4)) and a thousands
separator read as a decimal point (12.450 -> 12.45).

Çalıştırma: python scripts/number_format_report.py --model gemini-flash-lite --out outputs/numfmt-1-gemini.md
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sqlalchemy import select  # noqa: E402

from llmath.cascade import mcnemar_exact  # noqa: E402
from llmath.db import Question, Response, Run, SessionLocal, Verification  # noqa: E402
from report import interval, p_value, pct  # noqa: E402

CONDITIONS = [   # label, run, strategy
    ("CoT", "numfmt-1", "cot"),
    ("PoT", "numfmt-1", "pot"),
    ("PoT, Rational ipucu yok", "numfmt-1-norational", "pot"),
    ("PoT, Türkçe sayı notu", "numfmt-1-trnote", "pot"),
]
GROUPS = ["D1-tr", "D1-en", "D2-tr", "D2-en"]

Row = tuple[Response, Verification, Question]


def misreadings(question: Question, code: str | None) -> list[str]:
    """Number-format misreadings visible in a program, e.g. '8,4 → (8, 4)' or '12.450 → 12.45'."""
    if not code:
        return []
    code = re.sub(r"#.*", "", code)   # a comment may quote the question's number as it is written
    found = []
    # "(3,0)" is a point, not a decimal, so a number right after a parenthesis is skipped
    split = r"(?<![\d.,(])(\d+),(\d+)(?![\d,])" if question.lang == "tr" else r"(?<![\d.,])(\d{1,3}),(\d{3})(?![\d,])"
    for whole, part in re.findall(split, question.text):
        if re.search(rf"\(\s*{whole}\s*,\s*{part}\s*\)", code):
            found.append(f"{whole},{part} → ({whole}, {part})")
    if question.lang == "tr":
        for whole, part in re.findall(r"(?<![\d.,])(\d{1,3})\.(\d{3})(?![\d.])", question.text):
            if re.search(rf"(?<![\d.]){whole}\.{part.rstrip('0') or '0'}\d*(?!\d)", code):
                found.append(f"{whole}.{part} → {whole}.{part.rstrip('0') or '0'}")
    return found


def load(session, run_name: str, strategy: str, model: str) -> dict[str, Row]:
    run = session.scalars(select(Run).where(Run.name == run_name)).first()
    if run is None:
        return {}
    rows = session.execute(
        select(Response, Verification, Question)
        .join(Verification, Verification.response_id == Response.id)
        .join(Question, Question.id == Response.question_id)
        .where(Response.run_id == run.id, Response.model_key == model, Response.strategy == strategy,
               Response.error.is_(None))).all()
    return {q.id: (r, v, q) for r, v, q in rows}


def ok(row: Row) -> bool:
    return row[1].status == "correct"


def accuracy_table(data: dict[str, dict[str, Row]]) -> list[str]:
    lines = ["## Doğruluk", "", "| Koşul | " + " | ".join(GROUPS) + " | Toplam | %95 GA |",
             "| --- |" + " --- |" * (len(GROUPS) + 2)]
    for label, rows in data.items():
        if not rows:
            continue
        cells = []
        for group in GROUPS:
            values = [ok(r) for r in rows.values() if f"{r[2].subset}-{r[2].lang}" == group]
            cells.append(pct(sum(values), len(values)))
        total = [ok(r) for r in rows.values()]
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {pct(sum(total), len(total))} | {interval(total)} |")
    return lines


def language_table(data: dict[str, dict[str, Row]]) -> list[str]:
    lines = ["## Türkçe–İngilizce (aynı problem, eşleştirilmiş)", "",
             "| Koşul | Çift | İngilizce | Türkçe | Yalnız EN doğru | Yalnız TR doğru | McNemar p |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for label, rows in data.items():
        pairs: dict[str, dict[str, bool]] = defaultdict(dict)
        for r in rows.values():
            pairs[r[2].pair_id][r[2].lang] = ok(r)
        both = [p for p in pairs.values() if len(p) == 2]
        if not both:
            continue
        en, tr = [p["en"] for p in both], [p["tr"] for p in both]
        only_en, only_tr, p = mcnemar_exact(en, tr)
        lines.append(f"| {label} | {len(both)} | {pct(sum(en), len(en))} | {pct(sum(tr), len(tr))} | "
                     f"{only_en} | {only_tr} | {p_value(p)} |")
    return lines


def variant_table(data: dict[str, dict[str, Row]]) -> list[str]:
    base = data["PoT"]
    lines = ["## İstem varyantları (PoT, aynı soru, eşleştirilmiş)", "",
             "| Karşılaştırma | Dil | Soru | PoT | Varyant | Yalnız PoT doğru | Yalnız varyant doğru | McNemar p |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for label in ("PoT, Rational ipucu yok", "PoT, Türkçe sayı notu"):
        for lang in ("tr", "en"):
            ids = [q for q, r in data[label].items() if r[2].lang == lang and q in base]
            if not ids:
                continue
            a, b = [ok(base[q]) for q in ids], [ok(data[label][q]) for q in ids]
            only_a, only_b, p = mcnemar_exact(a, b)
            lines.append(f"| PoT ↔ {label} | {lang} | {len(ids)} | {pct(sum(a), len(a))} | {pct(sum(b), len(b))} | "
                         f"{only_a} | {only_b} | {p_value(p)} |")
    return lines


def misreading_section(data: dict[str, dict[str, Row]]) -> list[str]:
    lines = ["## Sayı yazımını yanlış okuma (programlarda)", "",
             "| Koşul | Dil | Program | Yanlış okuma görülen | Bunların yanlış cevabı |", "| --- | --- | --- | --- | --- |"]
    found = []
    for label, rows in data.items():
        if label == "CoT":
            continue
        for lang in ("tr", "en"):
            group = [r for r in rows.values() if r[2].lang == lang]
            if not group:
                continue
            hits = [(r, misreadings(r[2], r[0].code)) for r in group]
            hits = [(r, m) for r, m in hits if m]
            found += [(label, r, m) for r, m in hits]
            lines.append(f"| {label} | {lang} | {len(group)} | {len(hits)} | {sum(not ok(r) for r, _ in hits)} |")
    if found:
        lines += ["", "| Koşul | Soru | Beklenen | Program cevabı | Yanlış okuma |", "| --- | --- | --- | --- | --- |"]
        for label, r, m in found:
            lines.append(f"| {label} | {r[2].id} | {r[2].gold} | {r[0].extracted_answer or '—'} | {'; '.join(m)} |")
    return lines


def wrong_answers(data: dict[str, dict[str, Row]]) -> list[str]:
    wrong = [(label, r) for label, rows in data.items() for r in rows.values() if not ok(r)]
    lines = [f"## Yanlış ve doğrulanamayan cevaplar ({len(wrong)})", ""]
    if not wrong:
        return lines + ["Yok."]
    lines += ["| Koşul | Soru | Beklenen | Model cevabı | Karar | Katman |", "| --- | --- | --- | --- | --- | --- |"]
    for label, (response, verification, question) in sorted(wrong, key=lambda w: (w[0], w[1][2].id)):
        answer = (response.extracted_answer or "—").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {label} | {question.id} | {question.gold} | {answer} | {verification.status} | "
                     f"{verification.layer} |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Sayı yazımı deneyinin raporu")
    parser.add_argument("--model", required=True, help="model anahtarı (ör. gemini-flash-lite)")
    parser.add_argument("--out", type=Path, help="UTF-8 Markdown dosyası")
    args = parser.parse_args()
    with SessionLocal() as session:
        data = {label: load(session, run, strategy, args.model) for label, run, strategy in CONDITIONS}
    if not any(data.values()):
        sys.exit(f"{args.model} için sayı yazımı deneyi cevabı yok")
    parts = [f"# Sayı yazımı deneyi: {args.model}",
             "Veri: data/number_format_v1.jsonl (40 TR/EN çifti; D1 ondalık, D2 binlik ayırıcı)"]
    for section in (accuracy_table(data), language_table(data), variant_table(data), misreading_section(data),
                    wrong_answers(data)):
        parts.append("\n".join(section))
    text = "\n\n".join(parts) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"rapor yazıldı: {args.out}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
