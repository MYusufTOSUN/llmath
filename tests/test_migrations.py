import sqlite3

from sqlalchemy import create_engine, inspect

from llmath.db import init_db

# The schema the pilot database was created with, before migrations existed.
PILOT_SCHEMA = """
CREATE TABLE questions (id VARCHAR(64) PRIMARY KEY, subset VARCHAR(8), lang VARCHAR(2), pair_id VARCHAR(64),
    text TEXT, gold TEXT, answer_type VARCHAR(32), topic VARCHAR(64), source VARCHAR(128),
    license VARCHAR(64), note TEXT);
CREATE TABLE runs (id INTEGER PRIMARY KEY, name VARCHAR(128), created_at DATETIME, config JSON);
CREATE TABLE responses (id INTEGER PRIMARY KEY, run_id INTEGER, question_id VARCHAR(64), question_text TEXT,
    model_key VARCHAR(64), strategy VARCHAR(16), prompt_version VARCHAR(16), raw_output TEXT, reasoning TEXT,
    extracted_answer TEXT, extraction_method VARCHAR(32), code TEXT, exec_stdout TEXT, exec_error TEXT,
    input_tokens INTEGER, output_tokens INTEGER, cost_usd FLOAT, latency_ms INTEGER, error TEXT,
    created_at DATETIME);
"""


def _columns(url: str, table: str) -> set[str]:
    return {column["name"] for column in inspect(create_engine(url)).get_columns(table)}


def test_pilot_database_is_migrated_without_losing_rows(tmp_path):
    path = tmp_path / "pilot.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(PILOT_SCHEMA)
        connection.execute("INSERT INTO questions (id, subset, lang, text, gold, answer_type, source, license) "
                           "VALUES ('q1', 'P1', 'tr', '1+1', '2', 'integer', 'test', 'MIT')")
    url = f"sqlite:///{path.as_posix()}"

    init_db(url)
    init_db(url)   # a second run changes nothing

    assert "difficulty" in _columns(url, "questions")
    assert "finish_reason" in _columns(url, "responses")
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == "0002"
        assert connection.execute("SELECT gold FROM questions WHERE id = 'q1'").fetchone()[0] == "2"


def test_new_database_is_created_at_the_latest_revision(tmp_path):
    url = f"sqlite:///{(tmp_path / 'new.db').as_posix()}"
    init_db(url)
    assert {"difficulty", "note"} <= _columns(url, "questions")
    with sqlite3.connect(tmp_path / "new.db") as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == "0002"
