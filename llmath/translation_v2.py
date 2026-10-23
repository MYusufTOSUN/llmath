"""The second Turkish translation of the main set (P1 GSM8K, P2 MATH) and its review by the author.

Claude translated the 200 items from their English originals on 2026-10-22, after the author rejected the
machine-translated Turkish of main_v1. Translations live in data/tr_v2/p1.md and p2.md, one "### <pair id>"
block per item; "> " lines are notes for the reviewer, and "[ŞEKİL]" stands for the English item's [asy]
figure code, which is copied in unchanged. The author reads items next to the English on the
"Çeviri inceleme" page; each decision is appended to data/tr_v2/inceleme.jsonl (the latest one per item
counts), and an approved correction replaces the translation when main_v2 is built. Items the author
approves without reading them get their own decision, "toplu_onay", so that no item is counted as read by
a person when it was not: the author read the first 45 GSM8K items, found no errors and approved the
other 155 that way on 2026-10-22.

That first round was then set aside: the browser's automatic translation had been on and altered both
columns of the page, so the author restarted the review of all 200 items from scratch on 2026-10-23. The
first round's decisions (and six clicks of the restart day) are kept in inceleme_tur1.jsonl; only the
decisions in inceleme.jsonl count.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .config import DATA_DIR, MAIN_FILE
from .dataset import read_jsonl

FOLDER = DATA_DIR / "tr_v2"
FILES = {"P1": FOLDER / "p1.md", "P2": FOLDER / "p2.md"}
REVIEWS = FOLDER / "inceleme.jsonl"
ROUND1 = FOLDER / "inceleme_tur1.jsonl"   # the set-aside first round, kept for the record
FIGURE = "[ŞEKİL]"   # stands for the [asy] figure code of the English, copied in unchanged
_FIGURE = re.compile(r"\[asy\].*?\[/asy\]", re.DOTALL)
STATUSES = ("onay", "duzeltme", "toplu_onay")
READ = ("onay", "duzeltme")   # decisions taken after reading the item; "toplu_onay" is approval unread


def read_translations(path: Path) -> dict[str, dict]:
    """{pair id: {"text": ..., "notes": [...]}} from a "### <pair id>" file."""
    blocks: dict[str, dict] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("### "):
            current = line[4:].strip()
            if current in blocks:
                raise ValueError(f"{path.name}: {current} iki kez çevrilmiş")
            blocks[current] = {"lines": [], "notes": []}
        elif current is not None:
            if line.startswith("> "):
                blocks[current]["notes"].append(line[2:].strip())
            else:
                blocks[current]["lines"].append(line)
    return {pid: {"text": "\n".join(b["lines"]).strip(), "notes": b["notes"]} for pid, b in blocks.items()}


def with_figures(turkish: str, english: str) -> str:
    """The Turkish text with each [ŞEKİL] mark replaced by the English [asy] block in the same place."""
    figures, parts = _FIGURE.findall(english), turkish.split(FIGURE)
    if len(parts) - 1 != len(figures):
        raise ValueError(f"{len(figures)} şekil bekleniyordu, {len(parts) - 1} {FIGURE} işareti var")
    return parts[0] + "".join(figure + part for figure, part in zip(figures, parts[1:]))


def without_figures(text: str) -> str:
    """Figure code shown as the [ŞEKİL] mark, for reading."""
    return _FIGURE.sub(FIGURE, text)


def load_reviews(path: Path = REVIEWS) -> dict[str, dict]:
    """The latest review decision per pair id."""
    if not path.exists():
        return {}
    latest: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            latest[record["pair_id"]] = record
    return latest


def save_review(pair_id: str, status: str, text: str | None = None, comment: str | None = None,
                path: Path = REVIEWS) -> dict:
    """Append a decision: "onay" keeps Claude's translation, "duzeltme" replaces it with `text`,
    "toplu_onay" keeps it without the author having read it."""
    if status not in STATUSES:
        raise ValueError(f"bilinmeyen karar: {status}")
    if status == "duzeltme" and not (text or "").strip():
        raise ValueError("düzeltmede yeni Türkçe metin gerekli")
    record = {"pair_id": pair_id, "status": status, "text": text.strip() if status == "duzeltme" else None,
              "comment": (comment or "").strip() or None, "time": datetime.now(timezone.utc).isoformat()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_items(files: dict[str, Path] = FILES, main_file: Path = MAIN_FILE) -> list[dict]:
    """Every translated item with its English original, the old machine translation and its notes."""
    rows = read_jsonl(main_file)
    english = {r["pair_id"]: r for r in rows if r["lang"] == "en"}
    old = {r["pair_id"]: r for r in rows if r["lang"] == "tr" and r.get("pair_id")}
    items = []
    for subset, path in files.items():
        for pid, translation in read_translations(path).items():
            items.append({"pair_id": pid, "subset": subset, "en": english[pid]["text"], "gold": english[pid]["gold"],
                          "tr": translation["text"], "notes": translation["notes"],
                          "old": old[pid]["text"] if pid in old else None})
    return items


def current_text(item: dict, reviews: dict[str, dict]) -> str:
    """The Turkish that goes into main_v2: the author's correction if there is one, else the translation."""
    review = reviews.get(item["pair_id"])
    return review["text"] if review and review["status"] == "duzeltme" else item["tr"]
