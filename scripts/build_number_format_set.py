"""Build data/number_format_v1.jsonl (40 Turkish–English problem pairs that hinge on number writing)
and data/number_format_kumas_v1.jsonl (the follow-up on the main set's fabric template).

Çalıştırma: python scripts/build_number_format_set.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llmath.config import DATA_DIR, MAIN_FILE  # noqa: E402
from llmath.dataset import read_jsonl, write_jsonl  # noqa: E402
from llmath.number_format_set import LICENSE, SOURCE, generate, generate_fabric  # noqa: E402
from llmath.verify import verify  # noqa: E402

OUT = DATA_DIR / "number_format_v1.jsonl"
FABRIC_OUT = DATA_DIR / "number_format_kumas_v1.jsonl"


def original_fabric() -> list[dict]:
    """K0: main-1's five fabric questions word for word, under new ids."""
    rows = [r for r in read_jsonl(MAIN_FILE) if r["id"].startswith("m-p3-ondalik_kumas-")]
    return [{"id": f"nf-kumas-orijinal-{r['id'].rsplit('-', 1)[1]}-tr", "subset": "K0", "pair_id": None,
             "lang": "tr", "text": r["text"], "gold": r["gold"], "answer_type": r["answer_type"],
             "topic": "ondalık sayılar (kumaş kalıbı)", "source": f"{SOURCE}; ana setteki {r['id']}",
             "license": LICENSE} for r in rows]


def write(path: Path, rows: list[dict]) -> None:
    for row in rows:
        if verify(row["gold"], row["gold"]).status != "correct":
            raise ValueError(f"{row['id']}: cevap anahtarı kendini doğrulamıyor: {row['gold']}")
    write_jsonl(path, rows)
    print(f"{len(rows)} öğe yazıldı: {path}")
    for (subset, lang), n in sorted(Counter((r["subset"], r["lang"]) for r in rows).items()):
        print(f"  {subset} {lang}: {n}")


def main() -> None:
    write(OUT, generate())
    fabric = original_fabric()
    if len(fabric) != 5:
        raise ValueError(f"ana sette 5 kumaş sorusu bekleniyordu, {len(fabric)} bulundu")
    write(FABRIC_OUT, fabric + generate_fabric())


if __name__ == "__main__":
    main()
