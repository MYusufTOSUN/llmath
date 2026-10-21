import re
from collections import defaultdict

from sympy import Rational

from llmath.number_format_set import TEMPLATES, en_number, generate, generate_fabric
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
    assert generate_fabric() == generate_fabric()


def test_fabric_follow_up_changes_one_factor_at_a_time():
    rows = {r["id"]: r for r in generate_fabric()}
    assert len(rows) == 50 and {r["subset"] for r in rows.values()} == {"K1", "K2", "K3", "K4"}
    for size, whole_digits in (("kisa", 1), ("uzun", 2)):
        for i in range(1, 11):
            comma, plain = rows[f"nf-kumas-{size}-{i}-virgul-tr"], rows[f"nf-kumas-{size}-{i}-ve-tr"]
            assert plain["text"] == comma["text"].replace(" metre, Elif ", " metre ve Elif ")
            assert comma["gold"] == plain["gold"] and (Rational(comma["gold"]) * 100).q == 1
            numbers = re.findall(r"(\d+),\d+", comma["text"])
            assert len(numbers) == 3 and all(len(n) == whole_digits for n in numbers)
            if size == "kisa":
                assert rows[f"nf-kumas-kisa-{i}-en"]["pair_id"] == comma["pair_id"]
