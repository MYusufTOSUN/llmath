"""Build data/number_format_v1.jsonl: 40 Turkish–English problem pairs that hinge on number writing.

Çalıştırma: python scripts/build_number_format_set.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llmath.config import DATA_DIR  # noqa: E402
from llmath.dataset import write_jsonl  # noqa: E402
from llmath.number_format_set import generate  # noqa: E402
from llmath.verify import verify  # noqa: E402

OUT = DATA_DIR / "number_format_v1.jsonl"


def main() -> None:
    rows = generate()
    for row in rows:
        if verify(row["gold"], row["gold"]).status != "correct":
            raise ValueError(f"{row['id']}: cevap anahtarı kendini doğrulamıyor: {row['gold']}")
    write_jsonl(OUT, rows)
    print(f"{len(rows)} öğe yazıldı: {OUT}")
    for (subset, lang), n in sorted(Counter((r["subset"], r["lang"]) for r in rows).items()):
        print(f"  {subset} {lang}: {n}")


if __name__ == "__main__":
    main()
