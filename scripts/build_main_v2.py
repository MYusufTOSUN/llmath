"""Build data/main_v2.jsonl: the main set with the Turkish P1/P2 texts re-translated by Claude.

The author rejected the machine-translated Turkish of main_v1 (GSM8K-TR, made with Gemini 1.5; MCLM
MT-MATH100, made with GPT-4o) on 2026-10-22. The new translations live in data/tr_v2/ (see
llmath/translation_v2.py): no localization (names, currency and units as in the English), Turkish number
writing outside math, LaTeX math kept exactly as in the source. The author decides on every item on the
"Çeviri inceleme" page; corrections replace the translation. Each new item's source field says whether the
author read it ("okuyup onayladı", "okuyup düzeltti") or approved it unread ("okumadan toplu onayladı").

Checks, also on the author's corrections: a P1 translation has exactly the numbers of its English original
(separators aside), a P2 translation has exactly the same math segments, every [ŞEKİL] mark matches a figure.
main_v2 is written only when all 200 items have a decision and pass. The new Turkish items get the id
"<pair id>-tr2", so main_v1, its Turkish items and main-1 stay as they were; English and P3 (template)
items are copied unchanged.

Çalıştırma: python scripts/build_main_v2.py [--check]   (--check: yalnız denetler, dosya yazmaz)
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llmath.config import DATA_DIR, MAIN_FILE  # noqa: E402
from llmath.dataset import read_jsonl, write_jsonl  # noqa: E402
from llmath.translation_v2 import READ, current_text, load_items, load_reviews, with_figures  # noqa: E402

OUT = DATA_DIR / "main_v2.jsonl"
TRANSLATOR = "Claude (Anthropic) çevirisi, 2026-10-22"
DECISION = {"onay": "yazar okuyup onayladı", "duzeltme": "yazar okuyup düzeltti",
            "toplu_onay": "yazar okumadan toplu onayladı"}


def source_of(status: str, english_source: str) -> str:
    """The source field of a new Turkish item: who translated it and whether the author read it."""
    return f"{TRANSLATOR}, {DECISION[status]}; İngilizce asıl: {english_source}"

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
# Math as it appears in MATH problems; "\$" is a dollar sign in text, not a delimiter.
_MATH = re.compile(r"\$\$.*?\$\$|(?<!\\)\$(?:\\.|[^\\$])*\$|\\\(.*?\\\)|\\\[.*?\\\]|\[asy\].*?\[/asy\]",
                   re.DOTALL)
_ORDINAL = re.compile(r"\^\{\\(?:text|mathrm)\{(?:st|nd|rd|th)\}\}")
# English words that MATH typesets inside a formula, and their Turkish translation; the comparison maps the
# Turkish back, so these are the only intended changes inside formulas.
TEXT_IN_MATH = {
    r"\textbf{tek}": r"\textbf{odd}",                                           # m-p2-048
    r"\textbf{Juan'ın Eski Pulları}": r"\textbf{Juan's Old Stamping Grounds}",  # m-p2-057 (title)
    r"70\text{'li}": r"70\text{'s}",                                            # m-p2-057 (1970s)
}


def numbers(text: str, math: bool) -> Counter:
    """Numbers (outside math when the text has math), separators and leading zeros removed: 2.74 and 2,74
    are both 274, 88,000 and 88.000 are 88000, $.25 and 0,25 are 25. GSM8K has no math, and its dollar
    signs would look like math delimiters, so P1 text is taken whole."""
    if math:
        text = _MATH.sub(" ", text)
    return Counter(re.sub(r"[.,]", "", n).lstrip("0") or "0" for n in _NUMBER.findall(text))


def _normal(segment: str) -> str:
    """A math segment as compared: sentence punctuation at its end and an English ordinal mark (n^{th})
    may move out of the formula in Turkish ("$w.$" -> "$w$'yi …", "$n^{\\text{th}}$" -> "$n$'inci"), and the
    words listed in TEXT_IN_MATH are compared with their English originals."""
    segment = segment.strip()
    for turkish, english in TEXT_IN_MATH.items():
        segment = segment.replace(turkish, english)
    for opening, closing in (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$")):
        if segment.startswith(opening) and segment.endswith(closing) and len(segment) > len(opening) + len(closing):
            inner = segment[len(opening):len(segment) - len(closing)]
            inner = _ORDINAL.sub("", inner).rstrip().rstrip(".,;:").rstrip()
            return f"{opening}{inner}{closing}"
    return segment


def math_segments(text: str) -> Counter:
    return Counter(_normal(m) for m in _MATH.findall(text))


def problems(subset: str, english: str, turkish: str) -> list[str]:
    """What is wrong with a translation (after its figures are filled in); empty when it passes."""
    found = []
    en_numbers, tr_numbers = numbers(english, subset == "P2"), numbers(turkish, subset == "P2")
    if en_numbers != tr_numbers:
        found.append(f"sayılar farklı (eksik {dict(en_numbers - tr_numbers)}, fazla {dict(tr_numbers - en_numbers)})")
    if subset == "P2" and math_segments(english) != math_segments(turkish):
        found.append(f"formüller farklı: {list(math_segments(english) - math_segments(turkish))[:3]}")
    if not turkish.strip():
        found.append("çeviri boş")
    return found


def final_texts(items: list[dict], reviews: dict[str, dict]) -> tuple[dict[str, str], list[tuple[str, list[str]]]]:
    """The Turkish of every item as it goes into main_v2, and the problems found in it."""
    texts, report = {}, []
    for item in items:
        try:
            text = with_figures(current_text(item, reviews), item["en"])
        except ValueError as exc:
            report.append((item["pair_id"], [str(exc)]))
            continue
        issues = problems(item["subset"], item["en"], text)
        if issues:
            report.append((item["pair_id"], issues))
        texts[item["pair_id"]] = text
    return texts, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Türkçesi yeniden çevrilmiş ana set (main_v2)")
    parser.add_argument("--check", action="store_true", help="yalnız denetle, dosya yazma")
    args = parser.parse_args()

    v1 = read_jsonl(MAIN_FILE)
    items, reviews = load_items(), load_reviews()
    english = {r["pair_id"] for r in v1 if r["lang"] == "en"}
    translated = {item["pair_id"] for item in items}
    texts, report = final_texts(items, reviews)
    unreviewed = sorted(translated - set(reviews))
    decisions = Counter(reviews[pid]["status"] for pid in translated if pid in reviews)

    print(f"{len(items)} çeviri; {len(english - translated)} eksik, {len(report)} sorunlu; "
          f"{sum(decisions[s] for s in READ)} okundu ({decisions['duzeltme']} düzeltme), "
          f"{decisions['toplu_onay']} okunmadan onaylandı, {len(unreviewed)} kararsız")
    for pid, issues in report:
        print(f"  {pid}: {'; '.join(issues)}")
    if args.check:
        return
    if english - translated or report or unreviewed:
        sys.exit("her çeviri için karar verilmeden ve sorunlar çözülmeden main_v2 yazılmaz")

    subsets = {item["subset"] for item in items}
    rows = []
    for row in v1:
        if row["lang"] == "tr" and row["subset"] in subsets:
            continue   # the machine-translated Turkish of v1
        rows.append(row)
        if row["lang"] == "en":
            pid = row["pair_id"]
            item = next(i for i in items if i["pair_id"] == pid)
            review = reviews[pid]
            note = " ".join(item["notes"] + ([f"Yazarın notu: {review['comment']}"] if review.get("comment") else []))
            rows.append({**row, "id": f"{pid}-tr2", "lang": "tr", "text": texts[pid],
                         "source": source_of(review["status"], row["source"]), "note": note or None})
    write_jsonl(OUT, rows)
    print(f"{len(rows)} öğe yazıldı: {OUT}")


if __name__ == "__main__":
    main()
