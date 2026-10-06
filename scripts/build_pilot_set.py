"""Build data/pilot.jsonl: 10 GSM8K-TR, 10 MATH-500 (Turkish + English) and 10 template questions.

Sources (both MIT licensed):
  - GSM8K-TR  https://huggingface.co/datasets/bezir/gsm8k-tr
  - MCLM      https://huggingface.co/datasets/amphora/MCLM  (MT-MATH100, a MATH-500 subset)

Çalıştırma: python scripts/build_pilot_set.py
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmath.config import PILOT_FILE  # noqa: E402
from llmath.dataset import write_jsonl  # noqa: E402
from llmath.templates import PILOT_TEMPLATES, generate  # noqa: E402
from llmath.verify import to_fraction, to_sympy  # noqa: E402

ROWS_API = "https://datasets-server.huggingface.co/rows"
SEED = 2026

# Manual review of the machine-translated items (2026-10-06). Kept in code so the set can be
# rebuilt exactly; every change carries its reason into the `note` field.
EXCLUDE = {
    "p2-014": "cevap 9 tabanında (40_9); doğrulayıcı taban gösterimini okumuyor",
}
CORRECTIONS = {
    "p1-1951": {"gold": "550",
                "note": "GSM8K-TR çözümü alış fiyatını yanlış hesaplıyor (600); orijinal GSM8K cevabı 550"},
    "p1-1681": {"append": " (Bir ayı 4 hafta kabul edin.)",
                "note": "ayın uzunluğu belirsizdi; cevap anahtarı 4 hafta varsayıyor"},
    "p1-8385": {"replace": ("2.25 saat", "2,25 saat"), "note": "Türkçe ondalık virgül"},
    "p2-030-tr": {"replace": ("6.5 birimdir", "6,5 birimdir"), "note": "Türkçe ondalık virgül"},
    "p2-087": {"gold": "\\{1+\\sqrt{5}, 1-\\sqrt{5}, -2\\}", "note": "± işaretli cevap açık yazıldı"},
    "p2-060-tr": {"replace": ("basit kesirleri için geçerlidir. $\\frac{7}{30}@\\frac{10}{21}$ işleminin basit hali nedir?",
                              "sadeleştirilmiş kesirleri için geçerlidir. $\\frac{7}{30}@\\frac{10}{21}$ işleminin "
                              "sadeleştirilmiş değeri nedir?"),
                  "note": "'simplified fraction' yanlışlıkla 'basit kesir' (müfredatta: payı paydasından küçük) çevrilmişti"},
}


def apply_corrections(rows: list[dict]) -> list[dict]:
    for row in rows:
        for key in (row["id"], row.get("pair_id")):
            fix = CORRECTIONS.get(key)
            if not fix:
                continue
            if "gold" in fix:
                row["gold"] = fix["gold"]
            if "append" in fix:
                row["text"] += fix["append"]
            if "replace" in fix:
                old, new = fix["replace"]
                if old not in row["text"]:
                    raise ValueError(f"{row['id']}: düzeltilecek metin bulunamadı: {old!r}")
                row["text"] = row["text"].replace(old, new)
            row["note"] = "; ".join(filter(None, [row.get("note"), fix["note"]]))
    return rows


def fetch_rows(dataset: str, config: str, split: str, offset: int, length: int) -> list[dict]:
    response = httpx.get(ROWS_API, params={"dataset": dataset, "config": config, "split": split,
                                           "offset": offset, "length": length}, timeout=60)
    response.raise_for_status()
    return [item["row"] for item in response.json()["rows"]]


def gsm8k_tr(count: int) -> list[dict]:
    total = 8760
    rng = random.Random(SEED)
    rows = []
    for index in rng.sample(range(total), count * 3):
        row = fetch_rows("bezir/gsm8k-tr", "default", "test", index, 1)[0]
        gold = row["answer"].split("####")[-1].strip()
        if not re.fullmatch(r"-?\d+", gold):   # skip separators and decimals: gold must be canonical
            continue
        rows.append({"id": f"p1-{index:04d}", "subset": "P1", "lang": "tr", "pair_id": None,
                     "text": row["question"].strip(), "gold": gold, "answer_type": "integer",
                     "topic": "sözel problem", "source": "GSM8K-TR (bezir/gsm8k-tr)", "license": "MIT"})
        if len(rows) == count:
            return rows
    raise RuntimeError("yeterli GSM8K-TR sorusu bulunamadı")


def math100(count: int) -> list[dict]:
    rows = fetch_rows("amphora/MCLM", "MT-MATH100", "test", 0, 100)
    usable = []
    for index, row in enumerate(rows):
        gold = row["answer"].strip()
        if "\\text" in gold or re.search(r"[a-zA-Z]{2,}", re.sub(r"\\[a-zA-Z]+", "", gold)):
            continue                             # textual answers need a human grader
        if to_fraction(gold) is None and to_sympy(gold) is None:
            continue
        usable.append((index, row))
    rng = random.Random(SEED)
    # sample() draws in a fixed order, so asking for a few extra keeps the earlier picks stable
    picked = [item for item in rng.sample(usable, count + 5) if f"p2-{item[0]:03d}" not in EXCLUDE]
    chosen = sorted(picked[:count], key=lambda item: item[0])
    items = []
    for index, row in chosen:
        gold = row["answer"].strip()
        answer_type = "integer" if re.fullmatch(r"-?\d+", gold) else "expression"
        for lang in ("tr", "en"):
            items.append({"id": f"p2-{index:03d}-{lang}", "subset": "P2", "lang": lang,
                          "pair_id": f"p2-{index:03d}", "text": row[lang].strip(), "gold": gold,
                          "answer_type": answer_type, "topic": "MATH-500",
                          "source": "MCLM MT-MATH100 (amphora/MCLM)", "license": "MIT"})
    return items


def main() -> None:
    rows = gsm8k_tr(10) + math100(10) + generate(PILOT_TEMPLATES, per_template=2, seed=SEED)
    write_jsonl(PILOT_FILE, apply_corrections(rows))
    counts: dict[tuple, int] = {}
    for row in rows:
        counts[(row["subset"], row["lang"])] = counts.get((row["subset"], row["lang"]), 0) + 1
    print(f"{len(rows)} soru yazıldı: {PILOT_FILE}")
    for key, value in sorted(counts.items()):
        print(f"  {key[0]} {key[1]}: {value}")


if __name__ == "__main__":
    main()
