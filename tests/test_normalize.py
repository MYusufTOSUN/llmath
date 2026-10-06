import pytest

from llmath.normalize import normalize_answer, normalize_gold, number_readings, strip_wrappers


@pytest.mark.parametrize("token, readings, ambiguous", [
    ("3,5", ["3.5"], False),
    ("0,75", ["0.75"], False),
    ("1.250,75", ["1250.75"], False),
    ("1,250.75", ["1250.75"], False),
    ("1.250", ["1250", "1.250"], True),
    ("1,250", ["1.250", "1250"], True),
    ("1.250.000", ["1250000"], False),
    ("1,250,000", ["1250000"], False),
    ("-12,5", ["-12.5"], False),
    ("42", ["42"], False),
])
def test_number_readings(token, readings, ambiguous):
    assert number_readings(token) == (readings, ambiguous)


@pytest.mark.parametrize("raw, first", [
    ("25 TL", "25"),
    ("25 TL'dir", "25"),
    ("1.250'dir", "1250"),
    ("12 km/saat", "12"),
    ("45°", "45"),
    ("₺300", "300"),
    ("x = 4", "4"),
    ("$\\frac{1}{2}$", "\\frac{1}{2}"),
    ("\\boxed{\\dfrac{3}{4}}", "\\frac{3}{4}"),
    ("**18**", "18"),
    ("2,4x + 1", "2.4x + 1"),
])
def test_normalize_first_candidate(raw, first):
    assert normalize_answer(raw).candidates[0] == first


def test_percent_gives_both_readings():
    result = normalize_answer("%25")
    assert result.candidates == ["25", "25/100"]
    assert "yüzde" in result.notes


def test_multiple_answers_become_a_set():
    assert normalize_answer("x = 2 veya x = -3").candidates == ["{2, -3}"]


def test_choice_letters_only_for_choice_questions():
    assert normalize_answer("(C)", answer_type="choice").candidates == ["C"]
    assert normalize_answer("e").candidates == ["e"]


def test_gold_is_not_reinterpreted():
    assert normalize_gold("$y = 2x + 3$") == "2x + 3"
    assert normalize_gold("1.25") == "1.25"


def test_strip_wrappers_keeps_inner_braces():
    assert strip_wrappers("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert strip_wrappers("\\boxed{1}+\\boxed{2}") == "\\boxed{1}+\\boxed{2}"
