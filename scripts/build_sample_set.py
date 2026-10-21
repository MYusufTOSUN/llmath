"""Build data/main_v1_sample20.jsonl: a fixed 20-question sample of the main set.

Used to measure the reasoning models (time per answer, truncation, accuracy) before deciding
how much of the main set they run. Four GSM8K pairs and four MATH pairs, in both languages, and
four template questions; the MATH pairs are one of each level 2–5. The items keep their main-set
ids, so their answers can be compared with main-1 question by question.

Çalıştırma: python scripts/build_sample_set.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llmath.config import DATA_DIR, MAIN_FILE  # noqa: E402
from llmath.dataset import read_jsonl, write_jsonl  # noqa: E402

OUT = DATA_DIR / "main_v1_sample20.jsonl"
SEED = 2026_10_21


def main() -> None:
    rows = read_jsonl(MAIN_FILE)
    rng = random.Random(SEED)
    by_pair: dict[str, list[dict]] = {}
    for row in rows:
        if row["pair_id"]:
            by_pair.setdefault(row["pair_id"], []).append(row)

    def pairs(subset: str, level: int | None = None) -> list[str]:
        return sorted(p for p, items in by_pair.items() if items[0]["subset"] == subset and len(items) == 2
                      and (level is None or items[0]["difficulty"] == level))

    chosen = rng.sample(pairs("P1"), 4) + [rng.choice(pairs("P2", level)) for level in (2, 3, 4, 5)]
    sample = [row for p in chosen for row in sorted(by_pair[p], key=lambda r: r["lang"], reverse=True)]
    sample += rng.sample(sorted((r for r in rows if r["subset"] == "P3"), key=lambda r: r["id"]), 4)
    write_jsonl(OUT, sample)
    print(f"{len(sample)} öğe yazıldı: {OUT}")
    for row in sample:
        print(f"  {row['id']:<28} {row['subset']} {row['lang']} düzey={row['difficulty']}")


if __name__ == "__main__":
    main()
