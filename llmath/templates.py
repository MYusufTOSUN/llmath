"""Fresh Turkish questions (subset P3) generated from templates.

Each template draws its numbers from a seeded RNG and computes the answer with
SymPy, so the answer key is correct by construction and the questions cannot
appear in any model's training data.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass
from decimal import Decimal
from math import comb, factorial, lcm
from typing import Callable

from sympy import Integer, Rational


@dataclass(frozen=True)
class SetAnswer:
    values: tuple[Rational, ...]


@dataclass(frozen=True)
class IntervalAnswer:          # open interval (low, high)
    low: Rational
    high: Rational


Answer = Rational | SetAnswer | IntervalAnswer


@dataclass(frozen=True)
class Template:
    name: str
    topic: str
    make: Callable[[random.Random], tuple[str, Answer]]


# --- Turkish formatting --------------------------------------------------------------------

_UNITS = {1: "bir", 2: "iki", 3: "üç", 4: "dört", 5: "beş", 6: "altı", 7: "yedi", 8: "sekiz", 9: "dokuz"}
_TENS = {1: "on", 2: "yirmi", 3: "otuz", 4: "kırk", 5: "elli", 6: "altmış", 7: "yetmiş", 8: "seksen",
         9: "doksan"}


def tr_number(value) -> str:
    """Turkish style: decimal comma, and dots as thousands separators from 10 000 up."""
    value = Rational(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value.q == 1:
        digits = str(value.p)
        return sign + (f"{value.p:,}".replace(",", ".") if len(digits) >= 5 else digits)
    whole, _, fraction = format((Decimal(value.p) / Decimal(value.q)).normalize(), "f").partition(".")
    return f"{sign}{whole},{fraction}"


def tr_copula(n: int) -> str:
    """The '-dir' suffix after a written number: 3'tür, 6'dır, 12'dir, 40'tır."""
    n = abs(n)
    if n == 0:
        word = "sıfır"
    elif n % 10:
        word = _UNITS[n % 10]
    elif n % 100:
        word = _TENS[(n % 100) // 10]
    else:
        word = "yüz" if n % 1000 else "bin"
    vowel = next(ch for ch in reversed(word) if ch in "aeıioöuü")
    harmony = {"a": "ı", "ı": "ı", "e": "i", "i": "i", "o": "u", "u": "u", "ö": "ü", "ü": "ü"}[vowel]
    consonant = "t" if word[-1] in "çfhkpsşt" else "d"
    return f"'{consonant}{harmony}r"


def _coefficient(c: int, variable: str = "x") -> str:
    return variable if c == 1 else f"{c}{variable}"


def _signed_term(c: int, variable: str = "") -> str:
    if c == 0:
        return ""
    body = (variable if abs(c) == 1 and variable else f"{abs(c)}{variable}")
    return f" {'+' if c > 0 else '-'} {body}"


def quadratic(b: int, c: int) -> str:
    """x² + bx + c written the way a textbook would: x² - 5x + 6."""
    return "x²" + _signed_term(b, "x") + _signed_term(c)


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


def _decimal_fabric(rng: random.Random) -> tuple[str, Rational]:
    price = rng.choice([Rational(25, 2), Rational(15, 2), Rational(61, 4), Rational(42, 5), Rational(27, 4)])
    a, b = rng.sample([Rational(12, 5), Rational(18, 5), Rational(9, 5), Rational(21, 5), Rational(5, 2),
                       Rational(13, 10)], 2)
    text = (f"Metresi {tr_number(price)} TL olan bir kumaştan Ayşe {tr_number(a)} metre, Elif {tr_number(b)} "
            f"metre alıyor. İkisinin ödediği toplam tutar kaç TL'dir?")
    return text, price * (a + b)


def _inverse_proportion(rng: random.Random) -> tuple[str, Rational]:
    while True:
        workers, days, others = rng.randint(3, 12), rng.randint(4, 30), rng.randint(2, 20)
        if others != workers and (workers * days) % others == 0:
            text = (f"{workers} işçi bir duvarı {days} günde örüyor. Aynı hızda çalışan {others} işçi bu "
                    f"duvarı kaç günde örer?")
            return text, Rational(workers * days, others)


def _mixture(rng: random.Random) -> tuple[str, Rational]:
    while True:
        p, q = rng.choice([20, 25, 30, 40, 45, 50]), rng.choice([5, 10, 15])
        if p - q < 10:
            continue
        r = 5 * rng.randint(q // 5 + 1, p // 5 - 1)
        litres = rng.choice([20, 30, 40, 50, 60, 80])
        added = Rational(litres * (p - r), r - q)
        if added.is_integer and added > 0:
            text = (f"Tuz oranı %{p} olan {litres} litre tuzlu suya, tuz oranı %{q} olan kaç litre tuzlu su "
                    f"eklenirse karışımın tuz oranı %{r} olur?")
            return text, added


def _meeting(rng: random.Random) -> tuple[str, Rational]:
    while True:
        distance = rng.choice([180, 240, 300, 360, 450, 540, 600])
        v1, v2 = rng.choice([40, 50, 60, 70, 80]), rng.choice([50, 60, 70, 80, 90, 100])
        hours = Rational(distance, v1 + v2)
        if hours.q in (1, 2, 4, 5):
            text = (f"Aralarında {distance} km bulunan iki şehirden aynı anda birbirine doğru yola çıkan iki "
                    f"araçtan birinin saatteki hızı {v1} km, diğerininki {v2} km'dir. Araçlar kaç saat sonra "
                    f"karşılaşır?")
            return text, hours


def _digits(rng: random.Random) -> tuple[str, Rational]:
    a = rng.randint(1, 8)
    b = rng.randint(a + 1, 9)
    total, gap = a + b, 9 * (b - a)
    text = (f"İki basamaklı bir sayının rakamları toplamı {total}{tr_copula(total)}. Bu sayının rakamlarının "
            f"yerleri değiştirildiğinde elde edilen sayı, ilk sayıdan {gap} fazladır. İlk sayı kaçtır?")
    return text, Integer(10 * a + b)


def _lcm_remainder(rng: random.Random) -> tuple[str, Rational]:
    a, b = rng.sample([4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24], 2)
    left = rng.randint(1, min(a, b) - 1)
    text = (f"Bir sınıftaki öğrenciler {a} kişilik gruplara ayrıldığında {left} öğrenci, {b} kişilik gruplara "
            f"ayrıldığında yine {left} öğrenci açıkta kalıyor. Sınıfta {left} kişiden fazla öğrenci "
            f"olduğuna göre öğrenci sayısı en az kaçtır?")
    return text, Integer(lcm(a, b) + left)


def _round_table(rng: random.Random) -> tuple[str, Rational]:
    n = rng.randint(5, 8)
    if rng.random() < 0.5:
        return f"{n} kişi yuvarlak bir masa etrafına kaç farklı şekilde oturabilir?", Integer(factorial(n - 1))
    text = (f"Aralarında Ali ile Veli'nin de bulunduğu {n} kişi, Ali ile Veli yan yana olmak koşuluyla "
            f"yuvarlak bir masa etrafına kaç farklı şekilde oturabilir?")
    return text, Integer(2 * factorial(n - 2))


def _committee(rng: random.Random) -> tuple[str, Rational]:
    girls, boys = rng.randint(2, 6), rng.randint(3, 7)
    size = rng.randint(3, min(5, girls + boys - 1))
    text = (f"{girls} kız ve {boys} erkek öğrenci arasından {size} kişilik bir temsilci grubu seçilecektir. "
            f"Grupta en az bir kız öğrenci bulunması koşuluyla kaç farklı grup oluşturulabilir?")
    return text, Integer(comb(girls + boys, size) - comb(boys, size))


def _dice(rng: random.Random) -> tuple[str, Rational]:
    pairs = [(i, j) for i in range(1, 7) for j in range(1, 7)]
    kind = rng.random()
    if kind < 0.7:
        total = rng.randint(3, 11)
        favourable = sum(1 for i, j in pairs if i + j == total)
        condition = f"toplamının {total} olma"
    elif kind < 0.85:
        favourable = sum(1 for i, j in pairs if i + j in (2, 3, 5, 7, 11))
        condition = "toplamının asal sayı olma"
    else:
        favourable = sum(1 for i, j in pairs if 6 in (i, j))
        condition = "en az birinin 6 olma"
    text = (f"İki hilesiz zar birlikte atılıyor. Üst yüze gelen sayıların {condition} olasılığı kaçtır?")
    return text, Rational(favourable, 36)


def _absolute_value(rng: random.Random) -> tuple[str, SetAnswer]:
    while True:
        a, b, c = rng.choice([1, 2, 3]), rng.randint(1, 15), rng.randint(1, 15)
        x1, x2 = Rational(b - c, a), Rational(b + c, a)
        if x1.is_integer and x2.is_integer:
            return f"|{_coefficient(a)} - {b}| = {c} denkleminin çözüm kümesi nedir?", SetAnswer((x1, x2))


def _quadratic_inequality(rng: random.Random) -> tuple[str, IntervalAnswer]:
    r1, r2 = sorted(rng.sample(range(-6, 10), 2))
    text = f"{quadratic(-(r1 + r2), r1 * r2)} < 0 eşitsizliğinin çözüm kümesi nedir?"
    return text, IntervalAnswer(Integer(r1), Integer(r2))


def _inverse_function(rng: random.Random) -> tuple[str, Rational]:
    while True:
        a, b, c, d, t = (rng.randint(1, 5), rng.randint(1, 9), rng.randint(1, 4), rng.randint(1, 9),
                         rng.randint(1, 6))
        if c * t != a:
            text = (f"f(x) = ({_coefficient(a)} + {b}) / ({_coefficient(c)} - {d}) olduğuna göre "
                    f"f⁻¹({t}) kaçtır?")
            return text, Rational(b + d * t, c * t - a)


def _logarithm(rng: random.Random) -> tuple[str, Rational]:
    # (x - a)(x + b) = 2^(m+n) with x - a = 2^m and x + b = 2^n; the other root is outside the domain.
    while True:
        m = rng.randint(1, 3)
        n = rng.randint(m + 1, 5)
        if 2**n - 2**m < 3:
            continue
        x = rng.randint(2**m + 1, 2**n - 1)
        a, b = x - 2**m, 2**n - x
        text = f"log₂(x - {a}) + log₂(x + {b}) = {m + n} denklemini sağlayan x değeri kaçtır?"
        return text, Integer(x)


def _profit_loss(rng: random.Random) -> tuple[str, Rational]:
    while True:
        p, q = rng.choice([10, 20, 25, 30, 40, 50]), rng.choice([10, 20, 25, 30, 40])
        cost = Integer(20 * rng.randint(5, 60))
        sale, loss_price = cost * Rational(100 + p, 100), cost * Rational(100 - q, 100)
        if sale.is_integer and loss_price.is_integer:
            text = (f"Bir satıcı bir malı %{p} kârla {tr_number(sale)} TL'ye satıyor. Aynı malı %{q} zararla "
                    f"satsaydı kaç TL'ye satmış olurdu?")
            return text, loss_price


def _fill_and_drain(rng: random.Random) -> tuple[str, Rational]:
    while True:
        fill, drain = rng.randint(2, 10), rng.randint(3, 20)
        if drain <= fill:
            continue
        hours = Rational(fill * drain, drain - fill)
        if hours.q in (1, 2, 3, 4, 5):
            text = (f"Boş bir havuzu A musluğu tek başına {fill} saatte doldurmakta, B musluğu ise dolu havuzu "
                    f"{drain} saatte boşaltmaktadır. Havuz boşken iki musluk birlikte açılırsa havuz kaç "
                    f"saatte dolar?")
            return text, hours


def _geometric_sum(rng: random.Random) -> tuple[str, Rational]:
    a1, r = rng.randint(1, 5), rng.randint(2, 4)
    text = (f"Terimleri pozitif olan bir geometrik dizinin ikinci terimi {a1 * r}, beşinci terimi "
            f"{tr_number(a1 * r**4)} olduğuna göre bu dizinin ilk altı teriminin toplamı kaçtır?")
    return text, Integer(a1 * (r**6 - 1) // (r - 1))


TEMPLATES: dict[str, Template] = {t.name: t for t in [
    Template("yuzde_zincir", "yüzde", _percent_chain),
    Template("ortak_is", "iş problemi", _shared_work),
    Template("iki_cekim", "olasılık", _two_draws),
    Template("ortalama_hiz", "hız problemi", _average_speed),
    Template("yas", "yaş problemi", _ages),
    Template("kok_kareleri", "ikinci derece denklem", _root_squares),
    Template("aritmetik_toplam", "diziler", _arithmetic_sum),
    Template("ondalik_kumas", "ondalık sayılar", _decimal_fabric),
    Template("ters_oranti", "oran-orantı", _inverse_proportion),
    Template("karisim", "karışım problemi", _mixture),
    Template("karsilasma", "hız problemi", _meeting),
    Template("basamak", "sayı problemi", _digits),
    Template("ekok_kalan", "EBOB-EKOK", _lcm_remainder),
    Template("yuvarlak_masa", "permütasyon", _round_table),
    Template("komite", "kombinasyon", _committee),
    Template("zar", "olasılık", _dice),
    Template("mutlak_deger", "mutlak değer", _absolute_value),
    Template("esitsizlik", "eşitsizlik", _quadratic_inequality),
    Template("ters_fonksiyon", "fonksiyonlar", _inverse_function),
    Template("logaritma", "logaritma", _logarithm),
    Template("kar_zarar", "yüzde", _profit_loss),
    Template("musluk", "havuz problemi", _fill_and_drain),
    Template("geometrik_dizi", "diziler", _geometric_sum),
]}

PILOT_TEMPLATES = ["yuzde_zincir", "ortak_is", "iki_cekim", "ortalama_hiz", "yas"]
MAIN_TEMPLATES = ["yuzde_zincir", "ortak_is", "ortalama_hiz", "kok_kareleri", "ondalik_kumas", "ters_oranti",
                  "karisim", "karsilasma", "basamak", "ekok_kalan", "yuvarlak_masa", "komite", "zar",
                  "mutlak_deger", "esitsizlik", "ters_fonksiyon", "logaritma", "kar_zarar", "musluk",
                  "geometrik_dizi"]


def gold_string(value: Rational) -> str:
    return str(value.p) if value.q == 1 else f"{value.p}/{value.q}"


def _latex_number(value: Rational) -> str:
    if value.q == 1:
        return str(value.p)
    sign = "-" if value < 0 else ""
    return f"{sign}\\frac{{{abs(value.p)}}}{{{value.q}}}"


def gold_and_type(answer: Answer) -> tuple[str, str]:
    if isinstance(answer, SetAnswer):
        return "\\{" + ", ".join(_latex_number(v) for v in sorted(answer.values)) + "\\}", "set"
    if isinstance(answer, IntervalAnswer):
        return f"({_latex_number(answer.low)}, {_latex_number(answer.high)})", "interval"
    return gold_string(answer), "integer" if answer.q == 1 else "fraction"


def generate(names: list[str], per_template: int, seed: int = 2026, id_prefix: str = "p3",
             exclude_texts: set[str] | frozenset[str] = frozenset()) -> list[dict]:
    """Questions as dataset rows; the same seed always yields the same questions."""
    rows = []
    for name in names:
        template = TEMPLATES[name]
        rng = random.Random(seed + zlib.crc32(name.encode()))
        seen: set[str] = set()
        attempts = 0
        while len(seen) < per_template:
            attempts += 1
            if attempts > 1000:
                raise RuntimeError(f"{name}: yeterince farklı soru üretilemedi")
            text, answer = template.make(rng)
            if text in seen or text in exclude_texts:
                continue
            seen.add(text)
            gold, answer_type = gold_and_type(answer)
            rows.append({
                "id": f"{id_prefix}-{name}-{len(seen)}",
                "subset": "P3",
                "lang": "tr",
                "pair_id": None,
                "text": text,
                "gold": gold,
                "answer_type": answer_type,
                "topic": template.topic,
                "source": "LLMath şablonları",
                "license": "CC BY 4.0",
            })
    return rows
