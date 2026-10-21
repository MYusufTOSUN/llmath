"""Refinement with feedback that never reveals the answer key (F1 and F2).

The starting point is a model's CoT and PoT answers from an earlier run. Two kinds of feedback are
tried:

F1  execution feedback: a PoT program that gave no answer (no code, refused by the sandbox, an
    error, a time-out or no output) goes back to the model with what went wrong, at most twice;
F2  consistency feedback: when the CoT and PoT answers still disagree, or one of them is missing,
    the model is shown both solutions and solves the problem once more.

The final answer is the CoT answer when the two agree (method "agree", or "pot_fix" when they
agree only after F1) and the F2 answer otherwise ("consistency"; "cot_fallback" when F2 gives no
answer). Every call is stored in the refinement run, as strategy "pot_fix" for F1 attempts and
"consistency" for F2, and the decision as strategy "refine", so the run can be compared with CoT,
PoT and the cascade on the same questions. A question's rows are written together once all its
calls are done, so a run stopped by a quota limit resumes cleanly.

Örnek:
    python -m llmath.refine --base main-1 --models gemini-flash-lite
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from .cascade import answers_agree, mcnemar_exact, wilson
from .config import MODELS, ModelSpec
from .db import Question, Response, Run, SessionLocal, Verification
from .extract import extract_answer, extract_code, remove_thinking
from .llm import LLMError, LLMResponse, chat
from .prompts import PROMPT_VERSION, REFINE_VERSION, build_messages, consistency_messages, fix_feedback
from .sandbox import TIMEOUT_MESSAGE, run_code, tidy_traceback
from .service import get_or_create_run, model_status, record
from .solver import SolveResult, program_answer
from .verify import Verdict, verify

Chat = Callable[[list[dict[str, str]]], LLMResponse]
Step = tuple[SolveResult, Verdict]


def pot_problem(method: str, answer: str | None, exec_error: str | None) -> tuple[str, str] | None:
    """Why a PoT attempt gave no answer, as (kind, detail) for the F1 message; None when it gave one."""
    if method == "no_code":
        return "no_code", ""
    if method == "blocked":
        return "blocked", exec_error or ""
    if method == "program_error":
        if exec_error == TIMEOUT_MESSAGE:
            return "timeout", ""
        return "error", tidy_traceback(exec_error or "")[-1500:]
    if answer is None:
        return "empty", ""
    return None


def refine_question(question: Question, cot: Response, pot: Response, call: Chat,
                    max_rounds: int = 2) -> list[Step]:
    """F1 and F2 for one question; the rows to store, with the final decision last.

    The verdicts are for the analysis only; nothing the model sees depends on the answer key.
    An LLMError from `call` propagates, so that nothing is stored for a half-done question.
    """
    def judged(result: SolveResult) -> Step:
        return result, verify(question.gold, result.answer, question.answer_type)

    lang, steps = question.lang, []
    pot_answer, code = pot.extracted_answer, pot.code
    problem = pot_problem(pot.extraction_method, pot.extracted_answer, pot.exec_error)
    fixed = False
    if problem is not None:
        messages = build_messages(question.text, "pot", lang) + [
            {"role": "assistant", "content": pot.raw_output or ""}]
        for _ in range(max_rounds):
            messages.append({"role": "user", "content": fix_feedback(*problem, lang)})
            response = call(messages)
            messages.append({"role": "assistant", "content": response.text})
            new_code = extract_code(response.text)
            if new_code is None:
                result = SolveResult("pot_fix", None, "no_code", response, prompt_version=REFINE_VERSION)
            else:
                run = run_code(new_code)
                result = SolveResult("pot_fix", *program_answer(run), response, code=new_code,
                                     exec_stdout=run.stdout, exec_error=run.stderr or None,
                                     prompt_version=REFINE_VERSION)
                code = new_code
            steps.append(judged(result))
            problem = pot_problem(result.extraction_method, result.answer, result.exec_error)
            if problem is None:
                pot_answer, fixed = result.answer, True
                break

    cot_answer = cot.extracted_answer
    if answers_agree(cot_answer, pot_answer):
        final, method = cot_answer, "pot_fix" if fixed else "agree"
    else:
        response = call(consistency_messages(question.text, lang, remove_thinking(cot.raw_output or ""),
                                             cot_answer, code, pot_answer))
        extraction = extract_answer(response.text)
        steps.append(judged(SolveResult("consistency", extraction.answer, extraction.method, response,
                                        prompt_version=REFINE_VERSION)))
        final, method = (extraction.answer, "consistency") if extraction.answer else (cot_answer, "cot_fallback")
    steps.append(judged(SolveResult("refine", final, method, prompt_version=REFINE_VERSION)))
    return steps


def paced(spec: ModelSpec, pause: float) -> Chat:
    """chat() for one model, keeping `pause` seconds between cloud calls (free-tier rate limits)."""
    last = [float("-inf")]

    def call(messages: list[dict[str, str]]) -> LLMResponse:
        if spec.provider != "ollama":
            time.sleep(max(0.0, last[0] + pause - time.monotonic()))
        try:
            return chat(spec, messages)
        finally:
            last[0] = time.monotonic()
    return call


def base_attempts(session: Session, base_run_id: int, model_key: str) -> dict[str, tuple[Response, Response]]:
    """(CoT, PoT) responses of the model in the base run, by question, for questions that have both."""
    rows = session.scalars(select(Response).where(
        Response.run_id == base_run_id, Response.model_key == model_key, Response.error.is_(None),
        Response.prompt_version == PROMPT_VERSION, Response.strategy.in_(("cot", "pot")),
        Response.question_id.is_not(None)))
    by_question: dict[str, dict[str, Response]] = defaultdict(dict)
    for row in rows:
        by_question[row.question_id][row.strategy] = row
    return {qid: (d["cot"], d["pot"]) for qid, d in by_question.items() if len(d) == 2}


def summary(session: Session, run_id: int, base_run_id: int, model_key: str) -> dict:
    """CoT, PoT, PoT after F1 and the refined answer on the same questions, with F1 and F2 outcomes."""
    def verdicts(rid: int) -> list[tuple[Response, Verification]]:
        return session.execute(select(Response, Verification)
                               .join(Verification, Verification.response_id == Response.id)
                               .where(Response.run_id == rid, Response.model_key == model_key,
                                      Response.error.is_(None))
                               .order_by(Response.id)).all()

    base = {(r.question_id, r.strategy): v.status == "correct" for r, v in verdicts(base_run_id)
            if r.prompt_version == PROMPT_VERSION}
    final, last_fix, rows = {}, {}, verdicts(run_id)
    for r, v in rows:
        if r.strategy == "refine":
            final[r.question_id] = (r.extraction_method, v.status == "correct")
        elif r.strategy == "pot_fix":
            last_fix[r.question_id] = (r.extracted_answer is not None and r.extraction_method == "program_output",
                                       v.status == "correct")   # later attempts overwrite earlier ones
    ids = [q for q in sorted(final) if (q, "cot") in base and (q, "pot") in base]
    cot = [base[(q, "cot")] for q in ids]
    pot = [base[(q, "pot")] for q in ids]
    pot_f1 = [last_fix[q][1] if q in last_fix else base[(q, "pot")] for q in ids]
    refined = [final[q][1] for q in ids]

    def rate(flags: list[bool]) -> dict:
        low, high = wilson(sum(flags), len(flags))
        return {"doğru": sum(flags), "n": len(flags), "güven_aralığı": [round(low, 3), round(high, 3)]}

    resolved = [(base[(q, "cot")], final[q][1]) for q in ids if final[q][0] in ("consistency", "cot_fallback")]
    extra = [r for r, _ in rows if r.strategy in ("pot_fix", "consistency")]
    return {
        "model": model_key, "n": len(ids),
        "CoT": rate(cot), "PoT": rate(pot), "PoT + F1": rate(pot_f1), "Düzeltilmiş": rate(refined),
        "mcnemar_cot_a_karşı": dict(zip(("yalnız_düzeltilmiş_doğru", "yalnız_cot_doğru", "p"),
                                        mcnemar_exact(refined, cot))),
        "F1": {"çalışmayan_program": len(last_fix), "düzelen": sum(ok for ok, _ in last_fix.values()),
               "düzelip_doğru": sum(c for _, c in last_fix.values())},
        "F2": {"soru": len(resolved),
               "doğru_kaldı": sum(a and b for a, b in resolved), "bozuldu": sum(a and not b for a, b in resolved),
               "düzeldi": sum(b and not a for a, b in resolved), "yanlış_kaldı": sum(not a and not b for a, b in resolved)},
        "ek_çağrı": len(extra),
        "ek_token": sum((r.input_tokens or 0) + (r.output_tokens or 0) for r in extra),
        "yöntem": {m: sum(1 for q in ids if final[q][0] == m)
                   for m in ("agree", "pot_fix", "consistency", "cot_fallback")},
    }


def summary_text(s: dict) -> str:
    def pct(r: dict) -> str:
        low, high = r["güven_aralığı"]
        return f"%{100 * r['doğru'] / r['n']:.1f} ({r['doğru']}/{r['n']}, GA %{100 * low:.0f}–%{100 * high:.0f})"

    if not s["n"]:
        return f"{s['model']}: düzeltilmiş cevap yok"
    m, f1, f2 = s["mcnemar_cot_a_karşı"], s["F1"], s["F2"]
    return "\n".join([
        f"{s['model']} (n = {s['n']})",
        f"  CoT {pct(s['CoT'])} | PoT {pct(s['PoT'])} | PoT + F1 {pct(s['PoT + F1'])} | düzeltilmiş {pct(s['Düzeltilmiş'])}",
        f"  McNemar (düzeltilmiş ve CoT): yalnız düzeltilmiş doğru {m['yalnız_düzeltilmiş_doğru']}, "
        f"yalnız CoT doğru {m['yalnız_cot_doğru']}, p = {m['p']:.3f}",
        f"  F1: {f1['çalışmayan_program']} çalışmayan programdan {f1['düzelen']} tanesi çalıştı, "
        f"{f1['düzelip_doğru']} tanesi doğru cevap verdi",
        f"  F2: {f2['soru']} soru; CoT doğruydu ve doğru kaldı {f2['doğru_kaldı']}, bozuldu {f2['bozuldu']}, "
        f"CoT yanlıştı ve düzeldi {f2['düzeldi']}, yanlış kaldı {f2['yanlış_kaldı']}",
        f"  Karar yolu: {s['yöntem']}; ek çağrı {s['ek_çağrı']}, ek token {s['ek_token']}",
    ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Geri bildirimle düzeltme (F1 çalıştırma, F2 tutarlılık)")
    parser.add_argument("--base", required=True, help="CoT ve PoT cevaplarının alınacağı çalıştırma")
    parser.add_argument("--name", help="düzeltme çalıştırmasının adı (varsayılan: <base>-refine)")
    parser.add_argument("--models", nargs="+", required=True, choices=sorted(MODELS))
    parser.add_argument("--max-rounds", type=int, default=2, help="F1'de en fazla düzeltme turu")
    parser.add_argument("--limit", type=int, help="model başına en fazla soru sayısı")
    parser.add_argument("--pause", type=float, default=4.0, help="bulut modellerinde çağrılar arası bekleme (sn)")
    parser.add_argument("--max-errors", type=int, default=5, help="art arda bu kadar API hatasında model durdurulur")
    args = parser.parse_args(argv)
    name = args.name or f"{args.base}-refine"

    with SessionLocal() as session:
        base = session.scalars(select(Run).where(Run.name == args.base)).first()
        if base is None:
            print(f"! '{args.base}' adlı çalıştırma yok", file=sys.stderr)
            return 1
        run = get_or_create_run(session, name, {"base_run": args.base, "models": args.models,
                                                "max_rounds": args.max_rounds, "prompt_version": REFINE_VERSION})
        session.commit()
        for model_key in args.models:
            spec = MODELS[model_key]
            ok, reason = model_status(spec)
            if not ok:
                print(f"! {model_key} atlandı: {reason}", file=sys.stderr)
                continue
            attempts = base_attempts(session, base.id, model_key)
            done = set(session.scalars(select(Response.question_id).where(
                Response.run_id == run.id, Response.model_key == model_key, Response.strategy == "refine")))
            questions = {q.id: q for q in session.scalars(select(Question).where(Question.id.in_(list(attempts))))}
            todo = [qid for qid in sorted(attempts) if qid not in done][: args.limit]
            call, errors = paced(spec, args.pause), 0
            for i, qid in enumerate(todo, start=1):
                question = questions[qid]
                try:
                    steps = refine_question(question, *attempts[qid], call, args.max_rounds)
                except LLMError as exc:
                    errors += 1
                    print(f"[{i}/{len(todo)}] {model_key} {qid} API hatası: {exc}", flush=True)
                    if errors >= args.max_errors:
                        print(f"! {model_key} durduruldu: art arda {errors} API hatası (kota dolmuş olabilir); "
                              f"aynı komutla sonra devam edin", file=sys.stderr)
                        break
                    continue
                errors = 0
                for result, verdict in steps:
                    record(session, run, question.text, qid, spec, result, verdict)
                session.commit()
                decision, verdict = steps[-1]
                if len(steps) > 1:
                    print(f"[{i}/{len(todo)}] {model_key} {qid} {decision.extraction_method} "
                          f"{'✓' if verdict.status == 'correct' else '✗'} cevap={decision.answer!r} "
                          f"({len(steps) - 1} çağrı)", flush=True)
            print(summary_text(summary(session, run.id, base.id, model_key)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
