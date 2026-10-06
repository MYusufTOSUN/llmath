"""Database schema (SQLAlchemy). SQLite for local work, PostgreSQL via DATABASE_URL."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from .config import DATABASE_URL


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Question(Base):
    __tablename__ = "questions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    subset: Mapped[str] = mapped_column(String(8))           # P1 | P2 | P3
    lang: Mapped[str] = mapped_column(String(2))             # tr | en
    pair_id: Mapped[str | None] = mapped_column(String(64))  # links Turkish and English versions
    text: Mapped[str] = mapped_column(Text)
    gold: Mapped[str] = mapped_column(Text)
    answer_type: Mapped[str] = mapped_column(String(32))
    topic: Mapped[str | None] = mapped_column(String(64))
    source: Mapped[str] = mapped_column(String(128))
    license: Mapped[str] = mapped_column(String(64))
    note: Mapped[str | None] = mapped_column(Text)   # manual review notes (corrections, caveats)


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(default=_now)
    config: Mapped[dict] = mapped_column(JSON, default=dict)

    responses: Mapped[list[Response]] = relationship(back_populates="run")


class Response(Base):
    __tablename__ = "responses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)
    question_id: Mapped[str | None] = mapped_column(ForeignKey("questions.id"), index=True)
    question_text: Mapped[str] = mapped_column(Text)
    model_key: Mapped[str] = mapped_column(String(64))
    strategy: Mapped[str] = mapped_column(String(16))
    prompt_version: Mapped[str] = mapped_column(String(16))
    raw_output: Mapped[str | None] = mapped_column(Text)
    reasoning: Mapped[str | None] = mapped_column(Text)
    extracted_answer: Mapped[str | None] = mapped_column(Text)
    extraction_method: Mapped[str] = mapped_column(String(32))
    code: Mapped[str | None] = mapped_column(Text)
    exec_stdout: Mapped[str | None] = mapped_column(Text)
    exec_error: Mapped[str | None] = mapped_column(Text)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    cost_usd: Mapped[float | None] = mapped_column(Float)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=_now)

    run: Mapped[Run] = relationship(back_populates="responses")
    question: Mapped[Question | None] = relationship()
    verification: Mapped[Verification | None] = relationship(back_populates="response",
                                                            uselist=False)


class Verification(Base):
    __tablename__ = "verifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    response_id: Mapped[int] = mapped_column(ForeignKey("responses.id"), unique=True)
    status: Mapped[str] = mapped_column(String(16))      # correct | incorrect | unparsed
    layer: Mapped[str] = mapped_column(String(32))
    detail: Mapped[str] = mapped_column(Text, default="")
    ambiguous_format: Mapped[bool] = mapped_column(Boolean, default=False)
    normalized: Mapped[str | None] = mapped_column(Text)

    response: Mapped[Response] = relationship(back_populates="verification")


class Annotation(Base):
    """Error labels for wrong answers: Newman stage + error type (see the error taxonomy)."""

    __tablename__ = "annotations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    response_id: Mapped[int] = mapped_column(ForeignKey("responses.id"), index=True)
    annotator: Mapped[str] = mapped_column(String(64))   # "human" or an LLM name
    newman_stage: Mapped[str | None] = mapped_column(String(32))
    error_type: Mapped[str | None] = mapped_column(String(8))
    first_error_step: Mapped[int | None] = mapped_column(Integer)
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=_now)


_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)
