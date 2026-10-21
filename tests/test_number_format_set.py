import re
from collections import defaultdict

from sympy import Rational

from llmath.number_format_set import TEMPLATES, en_number, generate
from llmath.templates import tr_number
from llmath.verify import verify


def test_number_writing():
    assert (tr_number(Rational(84, 10)), en_number(Rational(84, 10))) == ("8,4", "8.4")
    assert (tr_number(12450), en_number(12450)) == ("12.450", "12,450")
    assert en_number(1250) == "1250" and en_number(Rational(-5, 4)) == "-1.25"


def test_pairs_share_numbers_and_answer_and_differ_in_format():
    rows = generate()
    assert len(rows) == 2 * 5 * len(TEMPLATES) and len({r["id"] for r in rows}) == len(rows)
    pairs = defaultdict(dict)
    for row in rows:
        pairs[row["pair_id"]][row["lang"]] = row
    for pair in pairs.values():
        tr, en = pair["tr"], pair["en"]
        assert tr["gold"] == en["gold"] and tr["subset"] == en["subset"]
        assert (Rational(tr["gold"]) * 100).q == 1                    # at most two decimals, nothing to round
        tr_digits = sorted(re.sub(r"[.,]", "", n) for n in re.findall(r"\d[\d.,]*\d|\d", tr["text"]))
        en_digits = sorted(re.sub(r"[.,]", "", n) for n in re.findall(r"\d[\d.,]*\d|\d", en["text"]))
        assert tr_digits == en_digits                                  # same numbers, different writing
        if tr["subset"] == "D1":
            assert re.search(r"\d,\d", tr["text"]) and not re.search(r"\d,\d", en["text"])
            assert re.search(r"\d\.\d", en["text"])
        else:
            assert re.search(r"\d\.\d{3}", tr["text"]) and re.search(r"\d,\d{3}", en["text"])
        # a correct answer is accepted however it is written
        value = Rational(tr["gold"])
        assert verify(tr["gold"], tr_number(value)).status == "correct"
        assert verify(tr["gold"], en_number(value)).status == "correct"


def test_generation_is_deterministic():
    assert generate() == generate()
