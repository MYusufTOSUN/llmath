from types import SimpleNamespace

import pytest

from llmath.cascade import Attempt, answers_agree, mcnemar_exact, report, simulate, wilson


def _question(qid: str):
    return SimpleNamespace(id=qid, subset="P1", lang="tr")


def _attempt(answer, correct, cost=0.0):
    return Attempt(answer, correct, cost, 1.0)


def test_agreement_is_symmetric_and_reads_turkish_formats():
    assert answers_agree("1250", "1.250")
    assert answers_agree("1.250", "1250")
    assert answers_agree("12/5", "2,4")
    assert not answers_agree("3", "4")
    assert not answers_agree(None, "4")


def test_cascade_accepts_agreement_and_escalates_disagreement():
    questions = {qid: _question(qid) for qid in ("a", "b", "c")}
    attempts = {
        ("cheap", "cot", "a"): _attempt("5", True), ("cheap", "pot", "a"): _attempt("5.0", True),
        ("strong", "cot", "a"): _attempt("5", True, 0.01),
        ("cheap", "cot", "b"): _attempt("7", False), ("cheap", "pot", "b"): _attempt("8", True),
        ("strong", "cot", "b"): _attempt("8", True, 0.01),
        # "c" has no strong-model answer, so it cannot enter the comparison
        ("cheap", "cot", "c"): _attempt("2", True), ("cheap", "pot", "c"): _attempt("2", True),
    }
    decisions, missing = simulate(attempts, questions, "cheap", "strong")
    assert missing == ["c"]
    assert [(d.question_id, d.agreed, d.correct) for d in decisions] == [("a", True, True), ("b", False, True)]
    assert decisions[1].cost == pytest.approx(0.01)

    result = report(attempts, decisions, "cheap", "strong")
    cheap_cot, _, strong_only, cascade = result["rows"]
    assert (cheap_cot["doğru"], strong_only["doğru"], cascade["doğru"]) == (1, 2, 2)
    assert cascade["güçlü_model_çağrısı"] == 1 and strong_only["güçlü_model_çağrısı"] == 2
    assert cascade["maliyet_usd"] < strong_only["maliyet_usd"]
    assert result["agreement"] == {"kabul_edilen": 1, "kabul_doğruluğu": 1.0,
                                   "yükseltilen": 1, "yükseltilende_doğruluk": 1.0}


def test_mcnemar_exact():
    assert mcnemar_exact([True] * 6, [False] * 6) == (6, 0, pytest.approx(2 / 64))
    assert mcnemar_exact([True, False], [True, False])[2] == 1.0


def test_wilson_interval():
    low, high = wilson(8, 10)
    assert low == pytest.approx(0.490, abs=0.001) and high == pytest.approx(0.943, abs=0.001)
    assert wilson(0, 0) == (0.0, 0.0)
