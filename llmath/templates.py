"""Fresh Turkish questions (subset P3) generated from templates.

Each template draws its numbers from a seeded RNG and computes the answer with
SymPy, so the answer key is correct by construction and the questions cannot
appear in any model's training data.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass
from typing import Callable

from sympy import Integer, Rational


@dataclass(frozen=True)
class Template:
    name: str
    topic: str
    make: Callable[[random.Random], tuple[str, Rational]]


def _percent_chain(rng: random.Random) -> tuple[str, Rational]:
    while True:
        p, q = rng.choice([10, 20, 25, 30, 40, 50]), rng.choice([10, 20, 25, 40])
        first = Integer(40 * rng.randint(5, 50))
        final = first * Rational(100 + p, 100) * Rational(100 - q, 100)
        if final.is_integer and final != first:
            text = (f"Bir ürünün fiyatı önce %{p} artırılıyor, ardından yeni fiyat üzerinden %{q} "
                    f"indirim yapılıyor. Ürünün son fiyatı {final} TL olduğuna göre ilk fiyatı kaç TL'dir?")
            return text, first


def _shared_work(rng: random.Random) -> tuple[str, Rational]:
    a = rng.randint(3, 12)
    b = rng.randint(a + 1, 16)
    text = (f"Ali bir işi tek başına {a} günde, Ayşe aynı işi tek başına {b} günde bitirebiliyor. "
            f"İkisi birlikte çalışırsa bu işi kaç günde bitirirler?")
    return text, Rational(a * b, a + b)


def _two_draws(rng: random.Random) -> tuple[str, Rational]:
    red, blue = rng.randint(2, 7), rng.randint(2, 8)
    total = red + blue
    text = (f"Bir torbada {red} kırmızı ve {blue} mavi top vardır. Torbadan geri konulmadan art arda "
            f"iki top çekiliyor. Çekilen iki topun da kırmızı olma olasılığı kaçtır?")
    return text, Rational(red * (red - 1), total * (total - 1))


def _average_speed(rng: random.Random) -> tuple[str, Rational]:
    v1, v2 = rng.choice([(40, 60), (60, 90), (30, 60), (50, 75), (40, 120), (80, 120), (60, 120), (70, 105)])
    distance = rng.choice([120, 240, 360, 480])
    text = (f"İki şehir arasındaki uzaklık {distance} km'dir. Bir araç gidişte saatte {v1} km, dönüşte "
            f"saatte {v2} km sabit hızla gidiyor. Aracın gidiş-dönüş boyunca ortalama hızı saatte kaç km'dir?")
    return text, Rational(2 * v1 * v2, v1 + v2)


def _ages(rng: random.Random) -> tuple[str, Rational]:
    while True:
        k = rng.randint(3, 6)
        m = rng.randint(2, k - 1)
        years = rng.randint(2, 15)
        daughter = Rational(years * (m - 1), k - m)
        if daughter.is_integer and 3 <= daughter <= 20:
            text = (f"Bir annenin bugünkü yaşı kızının yaşının {k} katıdır. {years} yıl sonra annenin "
                    f"yaşı kızının yaşının {m} katı olacaktır. Kızın bugünkü yaşı kaçtır?")
            return text, daughter


def _root_squares(rng: random.Random) -> tuple[str, Rational]:
    while True:
        s, p = rng.randint(2, 12), rng.randint(1, 20)
        if s * s - 4 * p > 0:
            text = (f"x² - {s}x + {p} = 0 denkleminin kökleri x₁ ve x₂ olduğuna göre "
                    f"x₁² + x₂² toplamı kaçtır?")
            return text, Integer(s * s - 2 * p)


def _arithmetic_sum(rng: random.Random) -> tuple[str, Rational]:
    a1, d, n = rng.randint(-5, 15), rng.randint(2, 9), rng.randint(8, 25)
    text = (f"İlk terimi {a1} ve ortak farkı {d} olan bir aritmetik dizinin ilk {n} teriminin "
            f"toplamı kaçtır?")
    return text, Rational(n * (2 * a1 + (n - 1) * d), 2)


TEMPLATES: dict[str, Template] = {t.name: t for t in [
    Template("yuzde_zincir", "yüzde", _percent_chain),
    Template("ortak_is", "iş problemi", _shared_work),
    Template("iki_cekim", "olasılık", _two_draws),
    Template("ortalama_hiz", "hız problemi", _average_speed),
    Template("yas", "yaş problemi", _ages),
    Template("kok_kareleri", "ikinci derece denklem", _root_squares),
    Template("aritmetik_toplam", "diziler", _arithmetic_sum),
]}

PILOT_TEMPLATES = ["yuzde_zincir", "ortak_is", "iki_cekim", "ortalama_hiz", "yas"]


def gold_string(value: Rational) -> str:
    return str(value.p) if value.q == 1 else f"{value.p}/{value.q}"


def generate(names: list[str], per_template: int, seed: int = 2026) -> list[dict]:
    """Questions as dataset rows; the same seed always yields the same questions."""
    rows = []
    for name in names:
        template = TEMPLATES[name]
        rng = random.Random(seed + zlib.crc32(name.encode()))
        seen: set[str] = set()
        while len(seen) < per_template:
            text, answer = template.make(rng)
            if text in seen:
                continue
            seen.add(text)
            rows.append({
                "id": f"p3-{name}-{len(seen)}",
                "subset": "P3",
                "lang": "tr",
                "pair_id": None,
                "text": text,
                "gold": gold_string(answer),
                "answer_type": "integer" if answer.q == 1 else "fraction",
                "topic": template.topic,
                "source": "LLMath şablonları",
                "license": "CC BY 4.0",
            })
    return rows
