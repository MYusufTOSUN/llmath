"""Template answers are re-derived independently, by brute force, from the question text."""

import math
import re
from fractions import Fraction

import pytest

from llmath.templates import MAIN_TEMPLATES, TEMPLATES, generate, quadratic, tr_copula, tr_number
from llmath.verify import verify


def _numbers(text: str) -> list[int]:
    return [int(n) for n in re.findall(r"\d+", text)]


def _rows(name: str, n: int = 25) -> list[dict]:
    return generate([name], per_template=n, seed=11)


@pytest.mark.parametrize("value, text", [
    (12, "12"), (Fraction(25, 2), "12,5"), (Fraction(61, 4), "15,25"), (12500, "12.500"),
    (1280, "1280"), (Fraction(-3, 4), "-0,75"),
])
def test_tr_number(value, text):
    assert tr_number(value) == text


@pytest.mark.parametrize("n, suffix", [
    (3, "'tür"), (4, "'tür"), (5, "'tir"), (6, "'dır"), (9, "'dur"), (10, "'dur"), (12, "'dir"),
    (16, "'dır"), (40, "'tır"), (60, "'tır"), (100, "'dür"),
])
def test_tr_copula(n, suffix):
    assert tr_copula(n) == suffix


def test_quadratic_formatting():
    assert quadratic(-5, 6) == "x² - 5x + 6"
    assert quadratic(1, -6) == "x² + x - 6"
    assert quadratic(0, -4) == "x² - 4"


def test_every_template_yields_self_consistent_gold():
    rows = generate(list(TEMPLATES), per_template=4, seed=3)
    assert len({row["id"] for row in rows}) == len(rows)
    for row in rows:
        assert verify(row["gold"], row["gold"], row["answer_type"]).status == "correct", row


def test_main_templates_exist_and_exclusion_works():
    assert len(MAIN_TEMPLATES) == 20 and set(MAIN_TEMPLATES) <= set(TEMPLATES)
    first = generate(["zar"], per_template=3, seed=5)
    again = generate(["zar"], per_template=3, seed=5, exclude_texts={first[0]["text"]})
    assert first[0]["text"] not in {row["text"] for row in again}


def test_lcm_remainder_by_brute_force():
    for row in _rows("ekok_kalan"):
        a, left, b = _numbers(row["text"])[:3]
        answer = next(n for n in range(left + 1, 10_000) if n % a == left and n % b == left)
        assert str(answer) == row["gold"]


def test_digits_by_brute_force():
    for row in _rows("basamak"):
        total, gap = _numbers(row["text"])[:2]     # "İki basamaklı" is written in words
        matches = [n for n in range(10, 100)
                   if n // 10 + n % 10 == total and (n % 10) * 10 + n // 10 - n == gap]
        assert matches == [int(row["gold"])]


def test_logarithm_root_is_in_the_domain_and_unique():
    for row in _rows("logaritma"):
        a, b, c = _numbers(row["text"])[:3]          # the subscript ₂ is not a decimal digit
        roots = [x for x in range(-200, 200) if x - a > 0 and x + b > 0 and (x - a) * (x + b) == 2**c]
        assert roots == [int(row["gold"])]


def test_dice_probabilities_by_enumeration():
    outcomes = [(i, j) for i in range(1, 7) for j in range(1, 7)]
    for row in _rows("zar", 10):
        if "asal" in row["text"]:
            hits = sum(1 for i, j in outcomes if i + j in (2, 3, 5, 7, 11))
        elif "en az birinin" in row["text"]:
            hits = sum(1 for i, j in outcomes if 6 in (i, j))
        else:
            total = _numbers(row["text"])[-1]
            hits = sum(1 for i, j in outcomes if i + j == total)
        assert Fraction(row["gold"]) == Fraction(hits, 36)


def test_absolute_value_solutions():
    for row in _rows("mutlak_deger"):
        match = re.search(r"\|(\d*)x - (\d+)\| = (\d+)", row["text"])
        a, b, c = int(match[1] or 1), int(match[2]), int(match[3])
        solutions = [x for x in range(-100, 100) if abs(a * x - b) == c]
        assert row["gold"] == "\\{" + ", ".join(map(str, solutions)) + "\\}"


def test_inequality_interval_by_sampling():
    for row in _rows("esitsizlik"):
        low, high = map(int, re.match(r"\((-?\d+), (-?\d+)\)", row["gold"]).groups())
        b, c = -(low + high), low * high
        assert quadratic(b, c) in row["text"]
        negative = [x / 4 for x in range(-40, 50) if (x / 4) ** 2 + b * (x / 4) + c < 0]
        assert min(negative) > low and max(negative) < high


def test_round_table_and_committee_formulas():
    for row in _rows("yuvarlak_masa", 6):
        n = _numbers(row["text"])[0]
        expected = 2 * math.factorial(n - 2) if "yan yana" in row["text"] else math.factorial(n - 1)
        assert row["gold"] == str(expected)
    for row in _rows("komite", 10):
        girls, boys, size = _numbers(row["text"])[:3]
        expected = sum(math.comb(girls, g) * math.comb(boys, size - g) for g in range(1, size + 1))
        assert row["gold"] == str(expected)
