"""Markdown report of the number-format experiment (data/number_format_v1.jsonl).

Conditions: CoT and PoT with the base prompt (run numfmt-1), PoT without the Rational hint
(numfmt-1-norational) and PoT with the Turkish number note (numfmt-1-trnote, Turkish questions
only). The report compares the conditions on the same questions and the Turkish and English
versions of each problem (exact McNemar), and searches every program for the two format
misreadings: a Turkish decimal split at its comma (8,4 -> Rational(8, 4)) and a thousands
separator read as a decimal point (12.450 -> 12.45).

When the fabric follow-up has been run (data/number_format_kumas_v1.jsonl, runs numfmt-kumas-1
and numfmt-kumas-1-norational), a section compares its cells: comma list against "ve" on the
same numbers (McNemar), short against two-digit numbers (Fisher's exact test, different numbers)
and Turkish against English.

Çalıştırma: python scripts/number_format_report.py --model gemini-flash-lite --out outputs/numfmt-1-gemini.md
"""

from __future__ import annotations

import argparse
import math
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
FABRIC_CONDITIONS = [("PoT", "numfmt-kumas-1", "pot"),
                     ("PoT, Rational ipucu yok", "numfmt-kumas-1-norational", "pot")]
FABRIC_CELLS = [("K0", "tr", "main-1'in 5 kumaş sorusu, aynen"),
                ("K1", "tr", "kısa sayılar (4,2), virgüllü sıralama"),
                ("K2", "tr", "kısa sayılar, \"ve\" ile"),
                ("K3", "tr", "iki basamaklı sayılar (14,2), virgüllü sıralama"),
                ("K4", "tr", "iki basamaklı sayılar, \"ve\" ile"),
                ("K1", "en", "K1'in İngilizcesi (kontrol)")]

Row = tuple[Response, Verification, Question]


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value for the 2×2 table [[a, b], [c, d]]."""
    n1, n2, k = a + b, c + d, a + c
    total = math.comb(n1 + n2, k)

    def prob(x: int) -> float:
        return math.comb(n1, x) * math.comb(n2, k - x) / total

    observed = prob(a)
    return min(1.0, sum(p for x in range(max(0, k - n2), min(k, n1) + 1)
                        if (p := prob(x)) <= observed * (1 + 1e-9)))


def misreadings(question: Question, code: str | None) -> list[str]:
    """Number-format misreadings visible in a program.

    Three kinds: a number split at its comma ('8,4 → (8, 4)'), a Turkish decimal kept as text that
    Python cannot read ("'7,5' metin"), and a thousands dot read as a decimal point ('12.450 → 12.45').
    """
    if not code:
        return []
    code = re.sub(r"#.*", "", code)   # a comment may quote the question's number as it is written
    found = []
    # "(3,0)" is a point, not a decimal, so a number right after a parenthesis is skipped
    split = r"(?<![\d.,(])(\d+),(\d+)(?![\d,])" if question.lang == "tr" else r"(?<![\d.,])(\d{1,3}),(\d{3})(?![\d,])"
    for whole, part in re.findall(split, question.text):
        if re.search(rf"\(\s*{whole}\s*,\s*{part}\s*\)", code):
            found.append(f"{whole},{part} → ({whole}, {part})")
        # only text handed to a number type as it is; '7,5'.replace(',', '.') is the right way
        elif question.lang == "tr" and re.search(
                rf"\b(?:Rational|Float|float|Fraction|Decimal|S|sympify|nsimplify)\(\s*['\"]{whole},{part}['\"]\s*\)", code):
            found.append(f"'{whole},{part}' metin")
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


def fabric_section(fabric: dict[str, dict[str, Row]]) -> list[str]:
    """Which feature of the main set's fabric template brings out the decimal-comma misreading?"""
    labels = [label for label, rows in fabric.items() if rows]
    lines = ["## Kumaş kalıbı izleme seti (PoT)", "",
             "| Hücre | Dil | İçerik | " + " | ".join(f"{label}: doğru, yanlış okuma" for label in labels) + " |",
             "| --- | --- | --- |" + " --- |" * len(labels)]
    for cell, lang, content in FABRIC_CELLS:
        cells = []
        for label in labels:
            group = [r for r in fabric[label].values() if r[2].subset == cell and r[2].lang == lang]
            misread = sum(bool(misreadings(r[2], r[0].code)) for r in group)
            cells.append(f"{pct(sum(ok(r) for r in group), len(group))}, {misread}" if group else "—")
        lines.append(f"| {cell} | {lang} | {content} | " + " | ".join(cells) + " |")

    base = fabric.get("PoT") or {}
    comparisons = []

    def paired(name: str, first: dict[str, Row], second: dict[str, Row], counterpart) -> None:
        ids = [q for q in first if counterpart(q) in second]
        if ids:
            a, b = [ok(first[q]) for q in ids], [ok(second[counterpart(q)]) for q in ids]
            only_a, only_b, p = mcnemar_exact(a, b)
            comparisons.append(f"| {name} | {len(ids)} | {pct(sum(a), len(a))} | {pct(sum(b), len(b))} | "
                               f"{only_a} | {only_b} | McNemar {p_value(p)} |")

    comma = {q: r for q, r in base.items() if q.endswith("-virgul-tr")}
    for size, name in (("kisa", "K1 virgül ↔ K2 \"ve\", aynı sayılar"), ("uzun", "K3 virgül ↔ K4 \"ve\", aynı sayılar")):
        paired(name, {q: r for q, r in comma.items() if f"-{size}-" in q}, base,
               lambda q: q.replace("-virgul-tr", "-ve-tr"))
    paired("K1 Türkçe ↔ İngilizce", {q: r for q, r in comma.items() if "-kisa-" in q}, base,
           lambda q: q.replace("-virgul-tr", "-en"))
    k1 = [ok(r) for r in base.values() if r[2].subset == "K1" and r[2].lang == "tr"]
    k3 = [ok(r) for r in base.values() if r[2].subset == "K3"]
    if k1 and k3:
        p = fisher_exact(sum(k1), len(k1) - sum(k1), sum(k3), len(k3) - sum(k3))
        comparisons.append(f"| K1 kısa ↔ K3 iki basamaklı, farklı sayılar | {len(k1)} + {len(k3)} | "
                           f"{pct(sum(k1), len(k1))} | {pct(sum(k3), len(k3))} | — | — | Fisher {p_value(p)} |")
    for label in labels[1:]:
        paired(f"PoT ↔ {label}", {q: r for q, r in base.items() if q in fabric[label]}, fabric[label], lambda q: q)
    if comparisons:
        lines += ["", "| Karşılaştırma | Soru | A | B | Yalnız A doğru | Yalnız B doğru | p |",
                  "| --- | --- | --- | --- | --- | --- | --- |"] + comparisons
    return lines


def wrong_answers(data: dict[str, dict[str, Row]], title: str = "Yanlış ve doğrulanamayan cevaplar") -> list[str]:
    wrong = [(label, r) for label, rows in data.items() for r in rows.values() if not ok(r)]
    lines = [f"## {title} ({len(wrong)})", ""]
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
        fabric = {label: load(session, run, strategy, args.model) for label, run, strategy in FABRIC_CONDITIONS}
    if not any(data.values()):
        sys.exit(f"{args.model} için sayı yazımı deneyi cevabı yok")
    parts = [f"# Sayı yazımı deneyi: {args.model}",
             "Veri: data/number_format_v1.jsonl (40 TR/EN çifti; D1 ondalık, D2 binlik ayırıcı)"]
    sections = [accuracy_table(data), language_table(data), variant_table(data), misreading_section(data),
                wrong_answers(data)]
    if any(fabric.values()):
        sections += [fabric_section(fabric), wrong_answers(fabric, "Kumaş setinde yanlış ve doğrulanamayan cevaplar")]
    for section in sections:
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
