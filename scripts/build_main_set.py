"""Build data/main_v1.jsonl, the ~500-item main question set, and a review list of flagged items.

P1  120 GSM8K test problems, English + Turkish. The Turkish text comes from GSM8K-TR, but only
    where its numbers and answer match the English original (GSM8K-TR is localized: some items
    have different numbers, and some have wrong answer keys). The gold answer is the original.
P2  80 MATH-500 problems, English + Turkish, from MCLM MT-MATH100; subject and level from MATH-500.
P3  100 fresh Turkish questions, 20 templates x 5, with answers computed by SymPy.
Questions used in the pilot are left out.

Çalıştırma: python scripts/build_main_set.py
"""

from __future__ import annotations

import io
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llmath.config import DATA_DIR, PILOT_FILE  # noqa: E402
from llmath.dataset import read_jsonl, write_jsonl  # noqa: E402
from llmath.templates import MAIN_TEMPLATES, generate  # noqa: E402
from llmath.verify import verify  # noqa: E402

MAIN_FILE = DATA_DIR / "main_v1.jsonl"
REVIEW_FILE = ROOT / "outputs" / "main_v1_review.md"
SEED = 2027
PILOT_P2 = {16, 30, 42, 60, 71, 72, 78, 84, 87, 90}
P2_EXCLUDE = {14: "cevap 9 tabanında", 43: "çoktan seçmeli şık cevabı", 56: "vektör/matris cevabı"}
P2_GOLD = {   # canonical forms of answers the verifier cannot read as written
    49: "864", 57: "5.4", 53: "\\{3+2\\sqrt{2}, 3-2\\sqrt{2}\\}", 87: "\\{1+\\sqrt{5}, 1-\\sqrt{5}, -2\\}",
    39: "11111111100",   # "11,\! 111,\! 111,\! 100" reads as a list of four numbers
}
# Manual review decisions (2026-10-08): the flagged items in outputs/main_v1_review.md, plus a full
# reading of every Turkish text against its English original.
EXCLUDE: dict[str, str] = {
    "m-p1-1119": "İngilizce asıl belirsiz: \"well over 4 hours longer\" cevabı belirlemiyor",
}
CORRECTIONS: dict[str, dict] = {
    "m-p1-0074-tr": {"replace": ("çiçekçi tezbanı işletiyor", "çiçek tezgâhı işletiyor"),
                     "note": "yazım hatası düzeltildi"},
    "m-p1-0163-tr": {"replace": ("oyunca kahraman", "oyuncak kahraman"), "note": "yazım hatası düzeltildi"},
    "m-p1-0312-tr": {"replace": ("Elif, Can'dan 3 yaş küçük, Can ise Arda'dan 5 yaş büyüktür.",
                                 "Elif, Can'dan 5 yaş büyük; Can ise Arda'dan 3 yaş küçüktür."),
                     "note": "ilişkiler İngilizce aslındaki sırayla kuruldu"},
    "m-p1-0610-tr": {"replace": ("metrekaresi 194 TL'ye", "metresi 194 TL'ye"),
                     "note": "\"per meter\" metrekare diye çevrilmişti; yalnız uzunluk verildiği için çözülemezdi"},
    "m-p1-0437-tr": {"replace": ("portakalarını kendi aralarında değil de 6 arkadaşlarıyla",
                                 "portakallarını diğer 6 arkadaşlarıyla"),
                     "note": "\"kendi aralarında değil de\" iki arkadaşı paylaşımın dışında bırakıyordu"},
    "m-p1-1164-tr": {"replace": ("az buldu. ve Maya", "az buldu ve Maya"), "note": "noktalama düzeltildi"},
    "m-p2-025-tr": {"replace": ("\\[-1 \\le x,\\] \\[y \\le 1\\] koşuluyla", "$-1 \\le x, y \\le 1$ koşuluyla"),
                    "note": "bölünmüş koşul formülü birleştirildi"},
    "m-p2-006-tr": {"replace": ("çakışana kadar kaç derece", "çakışana kadar en az kaç derece"),
                    "note": "\"minimum\" çeviride düşmüştü"},
    "m-p2-007-tr": {"replace": ("değerini nedir?", "değeri nedir?"), "note": "dil bilgisi düzeltildi"},
    "m-p2-036-tr": {"replace": ("Paul ve Jesse, ilk", "Paul ve Jesse'nin her biri, ilk"),
                    "note": "\"each\" çeviride düşmüştü; iki ayrı seçim olduğu belirtildi"},
    "m-p2-061-tr": {"replace": ("$CD$ doğrularının", "$CD$ doğru parçalarının"),
                    "note": "\"segments\" doğru diye çevrilmişti; doğrular hemen her zaman kesişir, cevap değişirdi"},
    "m-p2-062-tr": {"restore_figure": True, "note": "çeviride kaybolan şekil (asy) geri kondu"},
    "m-p2-077-tr": {"replace": ("dikdörtgen bir alan, bir atletizm pisti ile çevrilmiştir. Pist, alanın iki kenarından "
                                "ve iki yarım daireden oluşmaktadır. Pist uzunluğu 400 metredir. Alanın en büyük",
                                "dikdörtgen bir saha, bir atletizm pisti ile çevrilmiştir. Pist, sahanın iki kenarından "
                                "ve iki yarım daireden oluşmaktadır. Pist uzunluğu 400 metredir. Sahanın en büyük"),
                    "restore_figure": True,
                    "note": "çeviride kaybolan şekil (asy) geri kondu; \"field\" için \"alan\" yerine \"saha\" "
                            "(\"alanın alanı\" karışıklığı)"},
    "m-p2-095-tr": {"replace": ("değerini nedir?", "değeri nedir?"), "restore_figure": True,
                    "note": "çeviride kaybolan şekil (asy) geri kondu; dil bilgisi düzeltildi"},
}
CHECKED: dict[str, str] = {   # flagged items that were read and need no change
    "m-p1-0651-tr": "sorun yok: \"belirli bir ayda\" takvim ayı, anlam aynı",
    "m-p1-0792-tr": "sorun yok: \"30 günlük bir ay\" açıkça verilmiş",
    "m-p1-0992-tr": "sorun yok: \"ayda iki kez\" ve \"üç ayda bir\" doğru",
    "m-p2-021-tr": "sorun yok: formüller yeniden dizilmiş, anlam aynı",
    "m-p2-038-tr": "sorun yok: yalnızca şekil kodundaki yorumlar çevrilmiş",
    "m-p2-044-tr": "sorun yok: noktalama, anlam aynı",
    "m-p2-048-tr": "sorun yok: \\textbf{odd} → \\textbf{tek}",
    "m-p2-057-tr": "sorun yok: tablo başlığı ve \"70'ler\" çevrilmiş",
    **dict.fromkeys(["m-p2-019-tr", "m-p2-080-tr", "m-p2-092-tr"], "sorun yok: \\text{if} → \\text{eğer}"),
    **dict.fromkeys(["m-p2-001-tr", "m-p2-011-tr", "m-p2-026-tr", "m-p2-045-tr", "m-p2-053-tr", "m-p2-074-tr",
                     "m-p2-085-tr", "m-p2-097-tr"], "sorun yok: noktalama formülün dışına alınmış"),
}


# --- downloading -------------------------------------------------------------------------

def parquet(dataset: str, config: str, split: str) -> pd.DataFrame:
    listing = httpx.get("https://datasets-server.huggingface.co/parquet", params={"dataset": dataset},
                        timeout=60).json()["parquet_files"]
    url = next(f["url"] for f in listing if f["config"] == config and f["split"] == split)
    return pd.read_parquet(io.BytesIO(httpx.get(url, follow_redirects=True, timeout=180).content))


def mclm_rows() -> list[dict]:
    response = httpx.get("https://datasets-server.huggingface.co/rows", timeout=60, params={
        "dataset": "amphora/MCLM", "config": "MT-MATH100", "split": "test", "offset": 0, "length": 100})
    return [item["row"] for item in response.json()["rows"]]


# --- Turkish number formats ----------------------------------------------------------------

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
# "\$" is a literal dollar sign, not a math delimiter
_MATH = re.compile(r"\$\$.*?\$\$|(?<!\\)\$(?:\\.|[^\\$])*\$|\\\(.*?\\\)|\\\[.*?\\\]|\[asy\].*?\[/asy\]",
                   re.DOTALL)


def number_signature(text: str) -> tuple[str, ...]:
    return tuple(sorted(re.sub(r"[.,]", "", n) for n in _NUMBER.findall(text)))


def turkish_number_fixes(text: str) -> tuple[str, list[str]]:
    """Outside math mode: 2.25 -> 2,25 (English decimal) and 1,250 -> 1.250 (English thousands)."""
    notes: list[str] = []

    def fix(segment: str) -> str:
        def decimal(m: re.Match) -> str:
            notes.append(f"{m[0]} → {m[1]},{m[2]}")
            return f"{m[1]},{m[2]}"

        def thousands(m: re.Match) -> str:
            notes.append(f"{m[0]} → {m[0].replace(',', '.')}")
            return m[0].replace(",", ".")

        # a sentence-final period after the number is fine; another digit group is not
        segment = re.sub(r"(?<![\d.,])(\d+)\.(\d{1,2})(?!\d|[.,]\d)", decimal, segment)
        return re.sub(r"(?<![\d.,])\d{1,3}(?:,\d{3})+(?!\d|[.,]\d)", thousands, segment)

    pieces, last = [], 0
    for m in _MATH.finditer(text):
        pieces += [fix(text[last:m.start()]), m[0]]
        last = m.end()
    pieces.append(fix(text[last:]))
    return "".join(pieces), notes


# --- subsets -------------------------------------------------------------------------------

def gsm8k_gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def build_p1(count: int, pilot_tr_indices: set[int], flags: list[tuple[str, str]]) -> list[dict]:
    en, tr = parquet("openai/gsm8k", "main", "test"), parquet("bezir/gsm8k-tr", "default", "test")
    by_signature = defaultdict(list)
    for i, row in tr.iterrows():
        by_signature[(number_signature(row["question"]), gsm8k_gold(row["answer"]).replace(".", ""))].append(i)
    pairs = []
    for i, row in en.iterrows():
        gold = gsm8k_gold(row["answer"])
        hits = by_signature.get((number_signature(row["question"]), gold), [])
        if len(hits) == 1 and hits[0] not in pilot_tr_indices and re.fullmatch(r"-?\d+", gold):
            pairs.append((i, hits[0], gold))
    # The file holds the train translations first and then the test ones, with a few rows missing, so
    # the translation of English test row i sits at i + offset for a nearly constant offset. A match
    # far from that offset is a different problem that happens to share its few numbers and answer.
    offset = Counter(t - e for e, t, _ in pairs).most_common(1)[0][0]
    aligned = {p for p in pairs if abs(p[1] - p[0] - offset) <= 5}
    print(f"P1: {len(en)} İngilizce test sorusundan {len(aligned)} tanesi Türkçe çevirisiyle eşleşti "
          f"({len(pairs) - len(aligned)} rastlantısal eşleşme atıldı)")
    # shuffle all matches and keep the first aligned ones that were not excluded on review
    order = random.Random(SEED).sample(pairs, len(pairs))
    chosen = [p for p in order if p in aligned and f"m-p1-{p[0]:04d}" not in EXCLUDE][:count]
    rows = []
    for en_index, tr_index, gold in sorted(chosen):
        pair = f"m-p1-{en_index:04d}"
        tr_text, notes = turkish_number_fixes(tr.loc[tr_index, "question"].strip())
        risky = [w for w in (r"\bayda\b", r"\bkat fazla", r"\bkatı fazla") if re.search(w, tr_text.lower())]
        if notes or risky:
            flags.append((f"{pair}-tr", "; ".join(notes + [f"kontrol: {w}" for w in risky])))
        base = {"subset": "P1", "pair_id": pair, "gold": gold, "answer_type": "integer",
                "topic": "sözel problem", "license": "MIT"}
        rows.append({**base, "id": f"{pair}-en", "lang": "en", "text": en.loc[en_index, "question"].strip(),
                     "source": "GSM8K (openai/gsm8k)"})
        rows.append({**base, "id": f"{pair}-tr", "lang": "tr", "text": tr_text,
                     "source": f"GSM8K-TR (bezir/gsm8k-tr #{tr_index})",
                     "note": "; ".join(f"biçim: {n}" for n in notes) or None})
    return rows


def build_p2(count: int, flags: list[tuple[str, str]]) -> list[dict]:
    rows = mclm_rows()
    math500 = parquet("HuggingFaceH4/MATH-500", "default", "test")
    meta = {" ".join(p.split()): (s, int(l)) for p, s, l in zip(math500["problem"], math500["subject"],
                                                              math500["level"])}
    candidates = [i for i in range(len(rows)) if i not in PILOT_P2 and i not in P2_EXCLUDE]
    chosen = sorted(random.Random(SEED).sample(candidates, count))
    items = []
    for i in chosen:
        row = rows[i]
        gold = P2_GOLD.get(i, row["answer"].strip())
        if verify(gold, gold).status != "correct":
            raise ValueError(f"p2 #{i}: cevap anahtarı kendini doğrulamıyor: {gold}")
        subject, level = meta.get(" ".join(row["en"].split()), (None, None))
        if subject is None:
            flags.append((f"m-p2-{i:03d}", "MATH-500'de eşleşme yok (konu/zorluk bilinmiyor)"))
        # MATH texts are LaTeX and "$" also marks dollars, so numbers are kept as in the source;
        # the Turkish decimal-comma effect is measured on P1 and P3 instead.
        tr_text = row["tr"].strip()
        en_math = sorted(_MATH.findall(row["en"]))
        tr_math = sorted(_MATH.findall(tr_text))
        if en_math != tr_math:
            flags.append((f"m-p2-{i:03d}-tr", f"formüller İngilizceden farklı: {len(set(en_math) ^ set(tr_math))} parça"))
        base = {"subset": "P2", "pair_id": f"m-p2-{i:03d}", "gold": gold,
                "answer_type": "integer" if re.fullmatch(r"-?\d+", gold) else "expression",
                "topic": subject or "MATH-500", "difficulty": level,
                "source": "MCLM MT-MATH100 (amphora/MCLM)", "license": "MIT"}
        note = "kanonik cevap anahtarı" if i in P2_GOLD else None
        items.append({**base, "id": f"m-p2-{i:03d}-en", "lang": "en", "text": row["en"].strip(), "note": note})
        items.append({**base, "id": f"m-p2-{i:03d}-tr", "lang": "tr", "text": tr_text, "note": note})
    return items


def apply_review(rows: list[dict]) -> list[dict]:
    texts = {row["id"]: row["text"] for row in rows}
    kept = []
    for row in rows:
        if row["id"] in EXCLUDE or row.get("pair_id") in EXCLUDE:
            continue
        for key in (row["id"], row.get("pair_id")):
            fix = CORRECTIONS.get(key)
            if fix:
                if "replace" in fix:
                    old, new = fix["replace"]
                    if old not in row["text"]:
                        raise ValueError(f"{row['id']}: düzeltilecek metin yok: {old!r}")
                    row["text"] = row["text"].replace(old, new)
                if fix.get("restore_figure"):
                    figure = re.search(r"\[asy\].*?\[/asy\]", texts[row["pair_id"] + "-en"], re.DOTALL)[0]
                    row["text"] = f"{row['text']}\n\n{figure}"
                row["gold"] = fix.get("gold", row["gold"])
                row["note"] = "; ".join(filter(None, [row.get("note"), fix["note"]]))
        kept.append(row)
    return kept


def decision(item: str) -> str:
    if item in EXCLUDE:
        return f"çıkarıldı: {EXCLUDE[item]}"
    if item in CORRECTIONS:
        return f"düzeltildi: {CORRECTIONS[item]['note']}"
    return CHECKED.get(item, "incelenmedi")


def main() -> None:
    pilot = read_jsonl(PILOT_FILE)
    pilot_tr = {int(r["source"].split("#")[-1]) for r in pilot if "#" in (r.get("source") or "")}
    pilot_tr |= {int(r["id"].split("-")[1]) for r in pilot if r["subset"] == "P1"}
    flags: list[tuple[str, str]] = []
    rows = build_p1(120, pilot_tr, flags) + build_p2(80, flags)
    rows += generate(MAIN_TEMPLATES, per_template=5, seed=SEED, id_prefix="m-p3",
                     exclude_texts={r["text"] for r in pilot})
    rows = apply_review(rows)
    write_jsonl(MAIN_FILE, rows)

    counts: dict[tuple, int] = defaultdict(int)
    for row in rows:
        counts[(row["subset"], row["lang"])] += 1
    print(f"{len(rows)} öğe yazıldı: {MAIN_FILE}")
    for key, value in sorted(counts.items()):
        print(f"  {key[0]} {key[1]}: {value}")

    lines = [f"| {item} | {reason} | {decision(item)} |" for item, reason in flags]
    unreviewed = sum(decision(item) == "incelenmedi" for item, _ in flags)
    flagged = {item for item, _ in flags}
    found = [f"| {item} | {decision(item)} |" for item in [*EXCLUDE, *CORRECTIONS] if item not in flagged]
    REVIEW_FILE.parent.mkdir(exist_ok=True)
    REVIEW_FILE.write_text("# Ana set: inceleme kararları\n\n## Otomatik kontrollerin işaretledikleri\n\n"
                           "| Öğe | Neden | Karar |\n| --- | --- | --- |\n" + "\n".join(lines) + "\n\n"
                           "## Tam okumada bulunanlar\n\n| Öğe | Karar |\n| --- | --- |\n" + "\n".join(found) + "\n",
                           encoding="utf-8")
    print(f"{len(flags)} işaretli öğe, {unreviewed} tanesi incelenmedi: {REVIEW_FILE}")


if __name__ == "__main__":
    main()
