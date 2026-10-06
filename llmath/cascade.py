"""Agreement-gated cascade (research question AS2).

The cheap model solves each question twice, with CoT and with PoT. When the two answers
agree, that answer is accepted; otherwise the question is escalated to a stronger model.
The cascade is simulated from responses already stored for a run, so evaluating it costs
nothing extra.

Örnek:
    python -m llmath.cascade --run pilot-1 --cheap qwen3-8b --strong gemini-flash-lite
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import Question, Response, Run, SessionLocal, Verification
from .verify import verify


@dataclass(frozen=True)
class Attempt:
    answer: str | None
    correct: bool
    cost: float        # USD at list price; 0 for local models
    latency_s: float


@dataclass(frozen=True)
class Decision:
    question_id: str
    subset: str
    lang: str
    agreed: bool
    escalated: bool
    answer: str | None
    correct: bool
    cost: float
    latency_s: float


Attempts = dict[tuple[str, str, str], Attempt]   # (model, strategy, question_id) -> attempt


def answers_agree(a: str | None, b: str | None) -> bool:
    """Equivalent answers, checked both ways (the verifier treats its first argument as gold)."""
    if a is None or b is None:
        return False
    return verify(a, b).status == "correct" or verify(b, a).status == "correct"


def collect(session: Session, run_id: int) -> tuple[Attempts, dict[str, Question]]:
    """Stored attempts of a run, leaving out calls that failed at the API."""
    rows = session.execute(
        select(Response, Verification)
        .join(Verification, Verification.response_id == Response.id)
        .where(Response.run_id == run_id, Response.error.is_(None), Response.question_id.is_not(None))
    ).all()
    attempts = {
        (r.model_key, r.strategy, r.question_id): Attempt(
            r.extracted_answer, v.status == "correct", r.cost_usd or 0.0, (r.latency_ms or 0) / 1000)
        for r, v in rows
    }
    questions = {q.id: q for q in session.scalars(select(Question))}
    return attempts, questions


def simulate(attempts: Attempts, questions: dict, cheap: str, strong: str,
             strong_strategy: str = "cot") -> tuple[list[Decision], list[str]]:
    """Cascade decisions over the questions that have every attempt the comparison needs."""
    decisions, missing = [], []
    for qid in sorted(questions):
        needed = [(cheap, "cot", qid), (cheap, "pot", qid), (strong, strong_strategy, qid)]
        if not all(key in attempts for key in needed):
            if any(key in attempts for key in needed):
                missing.append(qid)
            continue
        cot, pot, upper = (attempts[key] for key in needed)
        agreed = answers_agree(cot.answer, pot.answer)
        chosen = cot if agreed else upper
        q = questions[qid]
        decisions.append(Decision(
            question_id=qid, subset=q.subset, lang=q.lang, agreed=agreed, escalated=not agreed,
            answer=chosen.answer, correct=chosen.correct,
            cost=cot.cost + pot.cost + (0.0 if agreed else upper.cost),
            latency_s=cot.latency_s + pot.latency_s + (0.0 if agreed else upper.latency_s),
        ))
    return decisions, missing


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """Exact two-sided McNemar test: (only a correct, only b correct, p-value)."""
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2**n
    return only_a, only_b, min(1.0, 2 * tail)


def report(attempts: Attempts, decisions: list[Decision], cheap: str, strong: str,
           strong_strategy: str = "cot") -> dict:
    """Metrics for the cascade and its baselines on the same questions."""
    ids = [d.question_id for d in decisions]

    def row(name: str, correct: list[bool], cost: float, latency: float, strong_calls: int) -> dict:
        k, n = sum(correct), len(correct)
        low, high = wilson(k, n)
        return {"yöntem": name, "n": n, "doğru": k, "doğruluk": round(k / n, 3) if n else 0.0,
                "güven_aralığı": [round(low, 3), round(high, 3)], "maliyet_usd": round(cost, 5),
                "ort_süre_sn": round(latency / n, 1) if n else 0.0, "güçlü_model_çağrısı": strong_calls}

    def baseline(model: str, strategy: str, name: str) -> tuple[dict, list[bool]]:
        items = [attempts[(model, strategy, qid)] for qid in ids]
        correct = [a.correct for a in items]
        calls = len(items) if model == strong else 0
        return row(name, correct, sum(a.cost for a in items), sum(a.latency_s for a in items), calls), correct

    cheap_cot, cheap_cot_ok = baseline(cheap, "cot", f"{cheap} CoT")
    cheap_pot, _ = baseline(cheap, "pot", f"{cheap} PoT")
    strong_only, strong_ok = baseline(strong, strong_strategy, f"{strong} {strong_strategy.upper()}")
    cascade_ok = [d.correct for d in decisions]
    cascade = row("Kademeli (uyuşma kontrolü)", cascade_ok, sum(d.cost for d in decisions),
                  sum(d.latency_s for d in decisions), sum(d.escalated for d in decisions))

    accepted = [d for d in decisions if d.agreed]
    escalated = [d for d in decisions if d.escalated]
    return {
        "cheap": cheap, "strong": strong, "n": len(decisions),
        "rows": [cheap_cot, cheap_pot, strong_only, cascade],
        "agreement": {
            "kabul_edilen": len(accepted),
            "kabul_doğruluğu": round(sum(d.correct for d in accepted) / len(accepted), 3) if accepted else None,
            "yükseltilen": len(escalated),
            "yükseltilende_doğruluk": round(sum(d.correct for d in escalated) / len(escalated), 3) if escalated else None,
        },
        "mcnemar": {
            "güçlü_modele_karşı": dict(zip(("yalnız_kademeli_doğru", "yalnız_güçlü_doğru", "p"),
                                          mcnemar_exact(cascade_ok, strong_ok))),
            "ucuz_cot_ya_karşı": dict(zip(("yalnız_kademeli_doğru", "yalnız_ucuz_doğru", "p"),
                                         mcnemar_exact(cascade_ok, cheap_cot_ok))),
        },
    }


def run_cascade(session: Session, run_id: int, cheap: str, strong: str, strong_strategy: str = "cot") -> dict:
    attempts, questions = collect(session, run_id)
    decisions, missing = simulate(attempts, questions, cheap, strong, strong_strategy)
    result = report(attempts, decisions, cheap, strong, strong_strategy) if decisions else {
        "cheap": cheap, "strong": strong, "n": 0, "rows": [], "agreement": {}, "mcnemar": {}}
    result["decisions"] = [d.__dict__ for d in decisions]
    result["missing"] = missing
    return result


def to_markdown(result: dict) -> str:
    lines = [f"## Kademeli çözüm: {result['cheap']} → {result['strong']} (n = {result['n']})", "",
             "| Yöntem | Doğruluk | %95 GA | Maliyet (USD, liste) | Ort. süre (sn) | Güçlü model çağrısı |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in result["rows"]:
        low, high = r["güven_aralığı"]
        lines.append(f"| {r['yöntem']} | %{100 * r['doğruluk']:.0f} ({r['doğru']}/{r['n']}) | "
                     f"%{100 * low:.0f}–%{100 * high:.0f} | {r['maliyet_usd']:.4f} | {r['ort_süre_sn']} | "
                     f"{r['güçlü_model_çağrısı']} |")
    a = result["agreement"]
    if a:
        lines += ["", f"Uyuşma: {a['kabul_edilen']} soruda kabul edildi (doğruluk %{100 * a['kabul_doğruluğu']:.0f}), "
                      f"{a['yükseltilen']} soru güçlü modele gönderildi"
                      + (f" (orada doğruluk %{100 * a['yükseltilende_doğruluk']:.0f})." if a['yükseltilen'] else ".")]
        m = result["mcnemar"]["güçlü_modele_karşı"]
        lines.append(f"McNemar (kademeli ve güçlü model): yalnız kademeli doğru {m['yalnız_kademeli_doğru']}, "
                     f"yalnız güçlü doğru {m['yalnız_güçlü_doğru']}, p = {m['p']:.3f}")
        c = result["mcnemar"]["ucuz_cot_ya_karşı"]
        lines.append(f"McNemar (kademeli ve ucuz model CoT): yalnız kademeli doğru {c['yalnız_kademeli_doğru']}, "
                     f"yalnız ucuz doğru {c['yalnız_ucuz_doğru']}, p = {c['p']:.3f}")
    if result["missing"]:
        lines.append(f"\nEksik cevap yüzünden dışarıda kalan soru: {len(result['missing'])}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Uyuşma kontrollü kademeli çözüm analizi")
    parser.add_argument("--run", required=True, help="çalıştırma adı")
    parser.add_argument("--cheap", required=True, help="ucuz model anahtarı")
    parser.add_argument("--strong", required=True, help="güçlü model anahtarı")
    parser.add_argument("--strong-strategy", default="cot", choices=["cot", "pot"])
    args = parser.parse_args(argv)
    with SessionLocal() as session:
        run = session.scalars(select(Run).where(Run.name == args.run)).first()
        if run is None:
            parser.error(f"çalıştırma bulunamadı: {args.run}")
        print(to_markdown(run_cascade(session, run.id, args.cheap, args.strong, args.strong_strategy)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
