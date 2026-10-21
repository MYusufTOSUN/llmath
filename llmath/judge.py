"""LLM judge: a first label for every wrong answer, in the same form as a human label.

The judge reads the question, the answer key, the model's solution with numbered steps (CoT) or
numbered code lines (PoT), the program output and the verifier's verdict, together with the
codebook, and answers in JSON: error type (H1–H13, Y1, Y2), Newman stage, first wrong step and a
short reason. Its labels are stored as annotations by "llm:<judge>/<prompt version>". The labelling
page never shows them, so human labels stay blind; agreement with the human labels is measured
afterwards with Cohen's kappa (--agreement).

A judge from the same model family as the answering model is refused, because models tend to
rate their own family's output more favourably.

Örnek:
    python -m llmath.judge --run main-1 --judge gemini-flash --models qwen3-8b
    python -m llmath.judge --run main-1 --judge gemini-flash --agreement
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import MODELS, ModelSpec
from .db import Annotation, Question, Response, Run, SessionLocal, Verification
from .llm import LLMError, LLMResponse, chat
from .sandbox import TIMEOUT_MESSAGE, tidy_traceback
from .service import model_status, save_annotation
from .taxonomy import ERROR_TYPES, MEASUREMENT_CODES, NEWMAN_STAGES, is_truncated, solution_steps

# Bump on any change to the judge prompt or the codebook; it is part of the annotator name.
# j0.1 -> j0.2 (2026-10-21): H3 now sends misreadings of Turkish number writing to H10 (never used).
# j0.2 -> j0.3 (2026-10-21): the author's codebook review (H5/H6/H8/H10/H11/H12/H13 wording, H10 under okuma).
JUDGE_VERSION = "j0.3"

SYSTEM = ("Sen matematik çözümlerindeki hataları bir kod kitabına göre sınıflandıran titiz bir "
          "değerlendiricisin. Yalnızca istenen JSON nesnesini yazarsın.")
STATUS = {"incorrect": "yanlış", "unparsed": "doğrulanamadı (cevap çıkarılamadı ya da okunamadı)"}
REMINDER = ("Cevabın geçerli bir JSON nesnesi değildi ya da error_type kod kitabında yoktu. Yalnızca istenen "
            "biçimde tek bir JSON nesnesi yaz.")
_STAGE_ALIASES = {"dönüştürme": "donusturme", "işlem": "islem"}


def annotator_name(judge_key: str) -> str:
    return f"llm:{judge_key}/{JUDGE_VERSION}"


def codebook_text() -> str:
    stages = "\n".join(f"- {key}: {name}. {definition}" for key, (name, definition) in NEWMAN_STAGES.items())
    types = "\n".join(
        f"- {t.code} {t.name}" + (f" (olağan aşama: {t.stage})" if t.stage else " (model hatası değil, aşama yok)")
        + f": {t.definition} Örnek: {t.example}" for t in ERROR_TYPES.values())
    return f"Newman aşamaları:\n{stages}\n\nHata türleri:\n{types}"


def numbered_solution(response: Response) -> tuple[str, str, int]:
    """The solution as the judge sees it: a heading, the numbered text and the number of steps.

    Steps are numbered as on the labelling page: code lines for PoT, solution_steps() for CoT.
    """
    if response.code:
        lines = response.code.splitlines()
        text = "```python\n" + "\n".join(f"{i:>3}| {line}" for i, line in enumerate(lines, start=1)) + "\n```"
        text += f"\n\nProgram çıktısı:\n{response.exec_stdout or '(boş)'}"
        if response.exec_error:
            text += f"\n\nProgram hatası:\n{tidy_traceback(response.exec_error)}"
        return "Python programı (PoT); numaralar kod satırlarını gösterir", text, len(lines)
    steps = solution_steps(response.raw_output or "")
    text = "\n\n".join(f"[{i}] {step}" for i, step in enumerate(steps, start=1))
    return "adım adım çözüm (CoT); numaralar adımları gösterir", text, len(steps)


def judge_messages(question: Question, response: Response, verification: Verification) -> tuple[list[dict], int]:
    heading, solution, count = numbered_solution(response)
    unit = "kod satırı" if response.code else "adım"
    facts = [f"Modelin cevabı: {response.extracted_answer or '(cevap çıkarılamadı)'}",
             f"Doğrulayıcının kararı: {STATUS.get(verification.status, verification.status)}"
             + (f" ({verification.detail})" if verification.detail else "")]
    if response.finish_reason == "length":
        facts.append("Çıktı token sınırında kesildi.")
    if response.exec_error == TIMEOUT_MESSAGE:
        facts.append("Program 15 saniye içinde bitmedi (zaman aşımı).")
    user = "\n\n".join([
        "Bir dil modelinin matematik sorusuna verdiği cevap doğrulamadan geçmedi. Görevin, çözümdeki ilk hatayı "
        "bulmak ve aşağıdaki kod kitabına göre etiketlemek.",
        f"## Soru\n{question.text.strip()}",
        f"## Cevap anahtarı\n{question.gold}",
        f"## Modelin çözümü: {heading}\n{solution}",
        "## Doğrulama\n" + "\n".join(facts),
        f"## Kod kitabı\n{codebook_text()}",
        "## İstenen çıktı\nYalnızca bir JSON nesnesi yaz:\n"
        '{"error_type": "H1", "newman_stage": "islem", "first_error_step": 3, "explanation": "..."}\n'
        "- error_type: kod kitabındaki kodlardan biri; birden çok hata varsa ilkini etiketle.\n"
        "- newman_stage: okuma, anlama, donusturme, islem ya da kodlama; Y1 ve Y2 için null.\n"
        f"- first_error_step: ilk hatanın bulunduğu {unit} numarası; belirlenemiyorsa 0.\n"
        "- explanation: hatayı bir iki cümleyle Türkçe açıkla.\n"
        "Modelin cevabı aslında doğruysa (aynı değerin başka bir yazımı gibi) Y1, soru ya da cevap anahtarı "
        "hatalıysa Y2 seç.",
    ])
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], count


def parse_label(text: str, step_count: int) -> dict | None:
    """The judge's JSON checked against the codebook, as save_annotation() arguments; None if unusable."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    code = str(data.get("error_type") or "").strip().upper()
    if code not in ERROR_TYPES:
        return None
    stage = None
    if code not in MEASUREMENT_CODES:
        stage = str(data.get("newman_stage") or "").strip().lower()
        stage = _STAGE_ALIASES.get(stage, stage)
        if stage not in NEWMAN_STAGES:
            stage = ERROR_TYPES[code].stage   # fall back to the type's usual stage
    try:
        step = int(data.get("first_error_step") or 0)
    except (TypeError, ValueError):
        step = 0
    note = str(data.get("explanation") or "").strip()
    return {"error_type": code, "newman_stage": stage,
            "first_error_step": step if 0 < step <= step_count else None, "note": note or None}


def same_family(judge: ModelSpec, model_key: str) -> bool:
    """True when the judge may not label this model: same family or base model, or a model it cannot check.

    A distillation counts as both families, e.g. DeepSeek-R1 8B on Qwen3-8B is DeepSeek and Alibaba.
    """
    spec = MODELS.get(model_key)
    if spec is None:
        return True
    judges, answers = {judge.family, judge.base} - {None}, {spec.family, spec.base} - {None}
    return bool(judges & answers)


def to_judge(session: Session, run_id: int, annotator: str,
             models: list[str] | None = None) -> list[tuple[Response, Verification, Question]]:
    """Wrong or unverified answers this judge has not labelled, in the labelling page's order."""
    query = (select(Response, Verification, Question)
             .join(Verification, Verification.response_id == Response.id)
             .join(Question, Question.id == Response.question_id)
             .where(Response.run_id == run_id, Response.error.is_(None), Verification.status != "correct"))
    if models:
        query = query.where(Response.model_key.in_(models))
    labelled = set(session.scalars(select(Annotation.response_id).where(Annotation.annotator == annotator)))
    rows = [row for row in session.execute(query).all()
            if row[0].id not in labelled and (row[0].raw_output or row[0].code)]
    rows.sort(key=lambda row: hashlib.sha256(f"{run_id}:{row[0].id}".encode()).hexdigest())
    return rows


def cohen_kappa(a: list, b: list) -> float | None:
    n = len(a)
    if n == 0:
        return None
    observed = sum(x == y for x, y in zip(a, b)) / n
    count_a, count_b = Counter(a), Counter(b)
    expected = sum(count_a[k] * count_b[k] for k in count_a) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def agreement(session: Session, run_id: int, first: str, second: str) -> dict:
    """Agreement of two annotators on the answers both labelled: error type, Newman stage, first step."""
    labels = session.scalars(select(Annotation).join(Response, Response.id == Annotation.response_id)
                             .where(Response.run_id == run_id, Annotation.annotator.in_((first, second))))
    by_response: dict[int, dict[str, Annotation]] = defaultdict(dict)
    for label in labels:
        by_response[label.response_id][label.annotator] = label
    both = [(d[first], d[second]) for d in by_response.values() if first in d and second in d]
    types = [(x.error_type, y.error_type) for x, y in both]
    stages = [(x.newman_stage, y.newman_stage) for x, y in both if x.newman_stage and y.newman_stage]
    steps = [(x.first_error_step, y.first_error_step) for x, y in both if x.first_error_step and y.first_error_step]

    def share(pairs: list) -> float | None:
        return round(sum(x == y for x, y in pairs) / len(pairs), 3) if pairs else None

    kappa = cohen_kappa([x for x, _ in types], [y for _, y in types])
    stage_kappa = cohen_kappa([x for x, _ in stages], [y for _, y in stages])
    return {"first": first, "second": second, "n": len(both),
            "tür_uyumu": share(types), "tür_kappa": None if kappa is None else round(kappa, 3),
            "aşama_n": len(stages), "aşama_uyumu": share(stages),
            "aşama_kappa": None if stage_kappa is None else round(stage_kappa, 3),
            "adım_n": len(steps), "adım_uyumu": share(steps)}


def agreement_text(a: dict) -> str:
    if not a["n"]:
        return f"{a['first']} ile {a['second']} için ortak etiket yok."
    return (f"{a['first']} ve {a['second']}: {a['n']} ortak etiket. Hata türü uyumu {a['tür_uyumu']}, "
            f"kappa {a['tür_kappa']}; Newman aşaması ({a['aşama_n']} etiket) uyumu {a['aşama_uyumu']}, "
            f"kappa {a['aşama_kappa']}; ilk hatalı adım ({a['adım_n']} etiket) aynı: {a['adım_uyumu']}")


def label_table(session: Session, run_id: int, annotator: str) -> str:
    """How often the annotator chose each error type, per model and strategy.

    The last row counts cut-off outputs, which are H13 in addition to their labelled first error.
    """
    rows = session.execute(select(Annotation.error_type, Response.model_key, Response.strategy,
                                  Response.finish_reason, Response.exec_error)
                           .join(Response, Response.id == Annotation.response_id)
                           .where(Response.run_id == run_id, Annotation.annotator == annotator)).all()
    columns = sorted({(model, strategy) for _, model, strategy, _, _ in rows})
    counts = Counter((code, model, strategy) for code, model, strategy, _, _ in rows)
    cut = Counter((model, strategy) for _, model, strategy, finish, error in rows if is_truncated(finish, error))
    lines = ["| Tür | " + " | ".join(f"{m} {s}" for m, s in columns) + " |",
             "| --- |" + " --- |" * len(columns)]
    for code, t in ERROR_TYPES.items():
        if any(counts[(code, m, s)] for m, s in columns):
            lines.append(f"| {code} {t.name} | " + " | ".join(str(counts[(code, m, s)]) for m, s in columns) + " |")
    lines.append("| Yarıda kalan çıktı (H13, ilk hatadan ayrıca) | "
                 + " | ".join(str(cut[(m, s)]) for m, s in columns) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM-hakem: yanlış cevaplara hata türü ve Newman aşaması")
    parser.add_argument("--run", required=True, help="çalıştırma adı")
    parser.add_argument("--judge", required=True, choices=sorted(MODELS), help="hakem model")
    parser.add_argument("--models", nargs="+", help="yalnız bu modellerin cevapları")
    parser.add_argument("--limit", type=int, help="en fazla bu kadar cevap etiketlenir")
    parser.add_argument("--pause", type=float, default=4.0, help="bulut modellerinde çağrılar arası bekleme (sn)")
    parser.add_argument("--max-errors", type=int, default=5, help="art arda bu kadar API hatasında durulur")
    parser.add_argument("--agreement", action="store_true", help="etiketlemeden, insan etiketleriyle uyumu yazdır")
    parser.add_argument("--human", default="human", help="--agreement için insan etiketleyicinin adı")
    args = parser.parse_args(argv)
    judge = MODELS[args.judge]
    annotator = annotator_name(judge.key)

    with SessionLocal() as session:
        run = session.scalars(select(Run).where(Run.name == args.run)).first()
        if run is None:
            print(f"! '{args.run}' adlı çalıştırma yok", file=sys.stderr)
            return 1
        if args.agreement:
            print(agreement_text(agreement(session, run.id, args.human, annotator)))
            return 0
        ok, reason = model_status(judge)
        if not ok:
            print(f"! hakem kullanılamıyor: {reason}", file=sys.stderr)
            return 1
        rows = to_judge(session, run.id, annotator, args.models)
        for model, n in Counter(r[0].model_key for r in rows if same_family(judge, r[0].model_key)).items():
            print(f"! {model}: {n} cevap atlandı; hakem aynı aileden ({judge.family}) ya da model tanınmıyor",
                  file=sys.stderr)
        todo = [r for r in rows if not same_family(judge, r[0].model_key)][: args.limit]

        last = [float("-inf")]

        def ask(messages: list[dict]) -> LLMResponse:
            if judge.provider != "ollama":
                time.sleep(max(0.0, last[0] + args.pause - time.monotonic()))
            try:
                return chat(judge, messages)
            finally:
                last[0] = time.monotonic()

        errors = unusable = 0
        for i, (response, verification, question) in enumerate(todo, start=1):
            messages, count = judge_messages(question, response, verification)
            try:
                reply = ask(messages)
                label = parse_label(reply.text, count)
                if label is None:   # one reminder; an answer still unlabelled is retried on the next run
                    reply = ask(messages + [{"role": "assistant", "content": reply.text},
                                            {"role": "user", "content": REMINDER}])
                    label = parse_label(reply.text, count)
            except LLMError as exc:
                errors += 1
                print(f"[{i}/{len(todo)}] {question.id} API hatası: {exc}", flush=True)
                if errors >= args.max_errors:
                    print(f"! durduruldu: art arda {errors} API hatası (kota dolmuş olabilir); aynı komutla "
                          f"sonra devam edin", file=sys.stderr)
                    break
                continue
            errors = 0
            if label is None:
                unusable += 1
                print(f"[{i}/{len(todo)}] {question.id} kullanılamayan hakem cevabı: {reply.text[:120]!r}", flush=True)
                continue
            save_annotation(session, response.id, annotator, **label)
            session.commit()
            print(f"[{i}/{len(todo)}] {response.model_key} {response.strategy} {question.id} → {label['error_type']}"
                  f" / {label['newman_stage'] or '-'} / adım {label['first_error_step'] or '-'}", flush=True)
        if unusable:
            print(f"! {unusable} cevap etiketlenemedi; aynı komutla yeniden denenir", file=sys.stderr)
        print(f"\n{annotator} etiketleri ({run.name}):\n{label_table(session, run.id, annotator)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
