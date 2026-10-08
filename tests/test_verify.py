import pytest

from llmath.verify import to_fraction, verify


@pytest.mark.parametrize("gold, pred", [
    ("12/5", "2,4"),                       # Turkish decimal comma
    ("12/5", "\\frac{12}{5}"),
    ("12/5", "2.4"),
    ("1250", "1.250"),                     # Turkish thousands separator
    ("1.25", "1,25"),
    ("25", "%25"),
    ("1/4", "%25"),
    ("48", "48 km/saat"),
    ("300", "300 TL'dir"),
    ("1/3", "0,333"),                      # rounded to three places
    ("\\frac{\\sqrt{3}}{2}", "\\frac{\\sqrt3}{2}"),
    ("\\frac{\\sqrt{3}}{2}", "sqrt(3)/2"),
    ("\\frac{\\sqrt{3}}{2}", "0.8660"),
    ("2x+3", "3 + 2x"),
    ("x^2-1", "(x-1)(x+1)"),
    ("\\{2,-3\\}", "x = -3 veya x = 2"),
    ("(3, \\frac{\\pi}{2})", "(3,\\frac{\\pi}{2})"),
    ("11\\sqrt2", "11\\sqrt{2}"),                # LaTeX shorthand without braces
    ("\\frac9{19}", "9/19"),
    ("\\frac 34", "0,75"),
    ("(2, 3)", "2 < x < 3"),                   # interval and inequality, both directions
    ("2 < x < 3", "(2, 3)"),
    ("(3,4]", "3 < x \\le 4"),
    ("(3,4]", "3 < \\lambda \\le 4"),          # \lambda is LaTeX, not Python
    ("(3,4]", "\\lambda \\in (3, 4]"),
    ("221", "\\$221"),                         # dollar amounts in English answers
    ("221", "$221.00"),
    ("2100", "\\$2,100"),
    ("3140", "3.140 kilometre"),
    ("29", "29 sent"),
    ("864", "864 \\text{ in}^2"),
    ("\\frac{20000}{\\pi}", "\\frac{20000}{\\pi} \\text{ m}^2"),
    ("11\\sqrt2", "11√2"),
    ("12", "12^{\\text{th}}"),
    ("3\\pi", "3 pi"),
    ("2x", "2 x"),
])
def test_equivalent_answers_are_correct(gold, pred):
    verdict = verify(gold, pred)
    assert verdict.status == "correct", verdict


@pytest.mark.parametrize("gold, pred", [
    ("12/5", "2,5"),
    ("1/3", "0,33"),                       # two places is not accepted as the exact value
    ("48", "50"),
    ("\\frac{\\sqrt{3}}{2}", "\\frac{\\sqrt{2}}{2}"),
    ("2x+3", "2x-3"),
    ("221", "\\$212"),
])
def test_wrong_answers_are_incorrect(gold, pred):
    assert verify(gold, pred).status == "incorrect"


def test_missing_answer_is_unparsed():
    verdict = verify("5", None)
    assert (verdict.status, verdict.layer) == ("unparsed", "extraction")


def test_code_like_answer_is_refused():
    verdict = verify("5", "__import__('os').system('dir')")
    assert verdict.status == "unparsed"
    assert verify("5", "lambda: 5").status == "unparsed"


def test_ambiguity_is_reported():
    assert verify("1250", "1.250").ambiguous_format is True
    assert verify("3.5", "3,5").ambiguous_format is False


def test_to_fraction():
    assert to_fraction("12/5") == to_fraction("2.4")
    assert to_fraction("\\frac{-3}{4}") == to_fraction("-0.75")
    assert to_fraction("x+1") is None
