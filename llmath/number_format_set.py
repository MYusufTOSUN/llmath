"""Turkish–English parallel problems that hinge on how numbers are written.

D1 problems contain decimals (Turkish 8,4 and English 8.4); D2 problems contain large numbers with
thousands separators (Turkish 12.450 and English 12,450). Each problem is drawn once and written in
both languages from the same numbers, and its answer is computed with SymPy, so a pair differs only
in language and number format. The set tests the PoT error seen in main-1, where the Turkish "8,4"
became Rational(8, 4). Answers have at most two decimals, so a model that rounds money to kuruş
is not marked wrong for it.
"""

from __future__ import annotations

import random
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal

from sympy import Rational

from .templates import gold_string, tr_copula, tr_number

LICENSE = "CC BY 4.0"
SOURCE = "Bu çalışma (sayı yazımı alt seti)"


def en_number(value) -> str:
    """English style: decimal point, and commas as thousands separators from 10,000 up."""
    value = Rational(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value.q == 1:
        digits = str(value.p)
        return sign + (f"{value.p:,}" if len(digits) >= 5 else digits)
    whole, _, fraction = format((Decimal(value.p) / Decimal(value.q)).normalize(), "f").partition(".")
    return f"{sign}{whole}.{fraction}"


def _decimal(rng: random.Random, low: int, high: int, places: int = 1) -> Rational:
    """A decimal with exactly `places` digits after the point, so it is never a whole number."""
    scale = 10 ** places
    while True:
        n = rng.randint(low * scale, high * scale)
        if n % 10:
            return Rational(n, scale)


@dataclass(frozen=True)
class Problem:
    tr: str
    en: str
    answer: Rational


@dataclass(frozen=True)
class FormatTemplate:
    name: str
    subset: str      # D1: decimals, D2: thousands separators
    topic: str
    make: Callable[[random.Random], Problem]


def _grocer(rng):
    # A price with two decimals is only bought in half kilograms, so the bill stays in whole kuruş
    # and no one has to round it.
    p = Rational(10 * rng.randint(100, 399) + rng.choice([2, 4, 6, 8]), 100)
    q, a, b = _decimal(rng, 10, 40), Rational(rng.choice([1, 3, 5, 7]), 2), _decimal(rng, 1, 4)
    tr = (f"Bir manavda domatesin kilogramı {tr_number(p)} TL, biberin kilogramı {tr_number(q)} TL'dir. "
          f"{tr_number(a)} kg domates ile {tr_number(b)} kg biber alan biri kaç TL öder?")
    en = (f"At a grocer, tomatoes cost {en_number(p)} lira per kilogram and peppers cost {en_number(q)} lira "
          f"per kilogram. How much does someone pay for {en_number(a)} kg of tomatoes and {en_number(b)} kg of "
          f"peppers, in lira?")
    return Problem(tr, en, p * a + q * b)


def _fuel(rng):
    c, d = _decimal(rng, 4, 9), 10 * rng.randint(12, 48)
    tr = f"Bir araba 100 km'de {tr_number(c)} litre benzin yakıyor. Bu araba {d} km'lik bir yolda kaç litre benzin yakar?"
    en = f"A car uses {en_number(c)} liters of fuel per 100 km. How many liters of fuel does it use on a {d} km trip?"
    return Problem(tr, en, c * d / 100)


def _average(rng):
    # One score has two decimals and an even last digit, so the average still has two decimals.
    x, y = Rational(10 * rng.randint(400, 989) + rng.choice([2, 4, 6, 8]), 100), _decimal(rng, 40, 99)
    tr = f"Bir öğrenci iki sınavdan {tr_number(x)} ve {tr_number(y)} puan almıştır. Bu iki puanın ortalaması kaçtır?"
    en = f"A student scored {en_number(x)} and {en_number(y)} points on two exams. What is the average of the two scores?"
    return Problem(tr, en, (x + y) / 2)


def _bottles(rng):
    k = rng.choice([Rational(3, 4), Rational(3, 2), Rational(5, 2), Rational(5, 4), Rational(7, 4)])
    n = rng.randint(8, 40)
    t = k * n
    if t.q == 1:   # the total must be a decimal too, or the problem would not test the format
        t, n = k * (n + 1), n + 1
    tr = (f"{tr_number(t)} litre zeytinyağı, her biri {tr_number(k)} litre alan şişelere doldurulacaktır. "
          f"Bu iş için kaç şişe gerekir?")
    en = (f"{en_number(t)} liters of olive oil will be poured into bottles that hold {en_number(k)} liters each. "
          f"How many bottles are needed?")
    return Problem(tr, en, Rational(n))


def _discount(rng):
    p, r = 10 * rng.randint(8, 40), _decimal(rng, 5, 30)
    tr = f"Fiyatı {p} TL olan bir ürüne %{tr_number(r)} indirim yapılıyor. Ürünün indirimli fiyatı kaç TL'dir?"
    en = f"A product priced at {p} lira is discounted by {en_number(r)}%. What is the discounted price, in lira?"
    return Problem(tr, en, p * (1 - r / 100))


def _garden(rng):
    a, b = _decimal(rng, 3, 25), _decimal(rng, 3, 25)
    tr = f"Kenarları {tr_number(a)} m ve {tr_number(b)} m olan dikdörtgen biçimindeki bir bahçenin alanı kaç metrekaredir?"
    en = f"A rectangular garden measures {en_number(a)} m by {en_number(b)} m. What is its area in square meters?"
    return Problem(tr, en, a * b)


def _income(rng):
    def amount():   # five digits with a non-zero last digit, so reading the dot as a decimal changes the value
        while True:
            value = rng.randint(10_000, 99_999)
            if value % 10:
                return value
    x, y = amount(), amount()
    tr = (f"Bir kafe ocak ayında {tr_number(x)} TL, şubat ayında {tr_number(y)} TL gelir elde ediyor. "
          f"İki ayın toplam geliri kaç TL'dir?")
    en = (f"A café earns {en_number(x)} lira in January and {en_number(y)} lira in February. "
          f"What is its total income for the two months, in lira?")
    return Problem(tr, en, Rational(x + y))


def _population(rng):
    n, r = 100 * rng.randint(120, 990), rng.choice([12, 15, 18, 24, 25, 35, 40, 45])
    tr = (f"Bir ilçenin nüfusu {tr_number(n)} kişidir. Öğrencilerin nüfusa oranı %{r}{tr_copula(r)}. "
          f"Bu ilçede kaç öğrenci yaşamaktadır?")
    en = (f"A district has a population of {en_number(n)}. Students make up {r}% of the population. "
          f"How many students live in the district?")
    return Problem(tr, en, Rational(n * r, 100))


TEMPLATES: dict[str, FormatTemplate] = {t.name: t for t in [
    FormatTemplate("manav", "D1", "ondalık sayılar", _grocer),
    FormatTemplate("yakit", "D1", "ondalık sayılar", _fuel),
    FormatTemplate("ortalama", "D1", "ondalık sayılar", _average),
    FormatTemplate("sise", "D1", "ondalık sayılar", _bottles),
    FormatTemplate("indirim", "D1", "ondalık sayılar", _discount),
    FormatTemplate("bahce", "D1", "ondalık sayılar", _garden),
    FormatTemplate("gelir", "D2", "binlik ayırıcı", _income),
    FormatTemplate("nufus", "D2", "binlik ayırıcı", _population),
]}


def generate(per_template: int = 5, seed: int = 2026_10_21) -> list[dict]:
    """Paired dataset rows (Turkish first, then English); the same seed always gives the same set."""
    rows = []
    for name, template in TEMPLATES.items():
        rng = random.Random(seed + zlib.crc32(name.encode()))
        seen: set[str] = set()
        while len(seen) < per_template:
            problem = template.make(rng)
            if problem.tr in seen:
                continue
            seen.add(problem.tr)
            pair = f"nf-{name}-{len(seen)}"
            base = {"subset": template.subset, "pair_id": pair, "gold": gold_string(problem.answer),
                    "answer_type": "integer" if problem.answer.q == 1 else "fraction",
                    "topic": template.topic, "source": SOURCE, "license": LICENSE}
            rows.append({**base, "id": f"{pair}-tr", "lang": "tr", "text": problem.tr})
            rows.append({**base, "id": f"{pair}-en", "lang": "en", "text": problem.en})
    return rows
