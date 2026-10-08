"""Read and write the question set (JSONL) and load it into the database."""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select

from .config import PILOT_FILE
from .db import Question, SessionLocal, init_db

FIELDS = ("id", "subset", "lang", "pair_id", "text", "gold", "answer_type", "topic", "source", "license",
          "note", "difficulty")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps({key: row.get(key) for key in FIELDS}, ensure_ascii=False) + "\n")


def import_questions(path: Path = PILOT_FILE) -> list[str]:
    """Insert or update every question in the file; returns their ids in file order."""
    init_db()
    rows = read_jsonl(path)
    with SessionLocal() as session:
        existing = {q.id: q for q in session.scalars(select(Question))}
        for row in rows:
            values = {key: row.get(key) for key in FIELDS}
            question = existing.get(row["id"])
            if question is None:
                session.add(Question(**values))
            else:
                for key, value in values.items():
                    setattr(question, key, value)
        session.commit()
    return [row["id"] for row in rows]
