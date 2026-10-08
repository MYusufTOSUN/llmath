"""Layered answer verification: exact text -> numeric -> Math-Verify -> SymPy.

Every verdict records the layer that decided it, so that the reliability of
each layer can later be measured against human labels (research question AS4).
"""

from __future__ import annotations

import inspect
import logging
import random
import re
from dataclasses import dataclass
from fractions import Fraction

import sympy
from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application,
                                        parse_expr, standard_transformations)

from .normalize import normalize_answer, normalize_gold


@dataclass
class Verdict:
    status: str                   # correct | incorrect | unparsed
    layer: str                    # extraction | safety | exact | numeric | math_verify | sympy | parse
    detail: str = ""
    ambiguous_format: bool = False
    normalized: str | None = None


def verify(gold: str, pred: str | None, answer_type: str | None = None) -> Verdict:
    if pred is None or not str(pred).strip():
        return Verdict("unparsed", "extraction", "cevap çıkarılamadı")
    expected = normalize_gold(gold)
    norm = normalize_answer(str(pred), answer_type)
    amb = norm.ambiguous
    candidates = [c for c in norm.candidates if is_safe(c)]
    if not candidates:
        return Verdict("unparsed", "safety", "güvenle ayrıştırılamayan ifade", amb)

    squashed = _squash(expected)
    for cand in candidates:
        if _squash(cand) == squashed:
            return Verdict("correct", "exact", "metin olarak eşit", amb, cand)

    gold_number = to_fraction(expected)
    if gold_number is not None:
        values = [(cand, to_fraction(cand)) for cand in candidates]
        for cand, value in values:
            if value is not None:
                match, why = numbers_match(gold_number, value, cand)
                if match:
                    return Verdict("correct", "numeric", why, amb, cand)
        if all(value is not None for _, value in values):
            return Verdict("incorrect", "numeric", f"beklenen {expected}, bulunan {candidates[0]}",
                           amb, candidates[0])

    by_math_verify = _math_verify_equal(expected, candidates)
    if by_math_verify:
        return Verdict("correct", "math_verify", "Math-Verify eşdeğer buldu", amb, candidates[0])
    by_sympy = _sympy_equal(expected, candidates)
    if by_sympy:
        return Verdict("correct", "sympy", "SymPy eşdeğer buldu", amb, candidates[0])
    if by_math_verify is None and by_sympy is None:
        return Verdict("unparsed", "parse", "ifade ayrıştırılamadı", amb, candidates[0])
    return Verdict("incorrect", "sympy" if by_sympy is not None else "math_verify",
                   f"beklenen {expected}, bulunan {candidates[0]}", amb, candidates[0])


# --- safety and numbers ------------------------------------------------------------------

_UNSAFE = re.compile(r"__|[`'\";@$]|\b(?:import|lambda|exec|eval|open|os|sys|subprocess|globals)\b")
_DECIMAL = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_LATEX_FRACTION = re.compile(r"^(-?)\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}$")
_RELATIVE_TOLERANCE = Fraction(1, 10**9)


def is_safe(text: str) -> bool:
    """Answers go to parsers that may evaluate Python; refuse anything code-like."""
    return len(text) <= 300 and not _UNSAFE.search(text)


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def to_fraction(text: str) -> Fraction | None:
    s = text.replace(" ", "")
    latex = _LATEX_FRACTION.match(s)
    if latex:
        num, den = to_fraction(latex[2]), to_fraction(latex[3])
        if num is None or not den:
            return None
        return -(num / den) if latex[1] else num / den
    if s.count("/") == 1:
        num, den = (to_fraction(part) for part in s.split("/"))
        return num / den if num is not None and den else None
    return Fraction(s) if _DECIMAL.match(s) else None


def numbers_match(expected: Fraction, value: Fraction, text: str) -> tuple[bool, str]:
    """Exact match, or a decimal correctly rounded to at least three places."""
    if expected == value:
        return True, "sayıca eşit"
    diff = abs(expected - value)
    if diff <= _RELATIVE_TOLERANCE * max(Fraction(1), abs(expected)):
        return True, "sayıca eşit (kayan nokta payı)"
    places = _decimal_places(text)
    if places >= 3 and diff <= Fraction(1, 2 * 10**places):
        return True, f"{places} basamağa doğru yuvarlanmış"
    return False, ""


def _decimal_places(text: str) -> int:
    match = re.fullmatch(r"[+-]?\d*\.(\d+)", text.strip())
    return len(match[1]) if match else 0


# --- Math-Verify -------------------------------------------------------------------------

def _math_verify_equal(expected: str, candidates: list[str]) -> bool | None:
    try:
        from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse
        from math_verify import verify as mv_verify
    except ImportError:
        return None
    configs = [LatexExtractionConfig(), ExprExtractionConfig()]

    def parsed(text: str) -> list:
        for form in (f"${text}$", text):
            try:
                result = parse(form, extraction_config=configs, **_kwargs(parse, "parsing_timeout"))
            except Exception:
                result = []
            if result:
                return result
        return []

    gold = parsed(expected)
    if not gold:
        return None
    seen = False
    for cand in candidates:
        answer = parsed(cand)
        if not answer:
            continue
        seen = True
        options = _kwargs(mv_verify, "timeout_seconds")
        if "allow_set_relation_comp" in inspect.signature(mv_verify).parameters:
            options["allow_set_relation_comp"] = True   # "(2, 3)" and "2 < x < 3" match both ways
        try:
            if mv_verify(gold, answer, **options):
                return True
        except Exception:
            continue
    return False if seen else None


def _kwargs(func, timeout_name: str) -> dict:
    """Math-Verify's timeouts rely on SIGALRM, which Windows and worker threads lack."""
    return {timeout_name: None} if timeout_name in inspect.signature(func).parameters else {}


# With timeouts disabled on purpose (inputs are length-limited instead), Math-Verify warns on
# every call; keep its log quiet unless something actually fails.
logging.getLogger("math_verify").setLevel(logging.ERROR)


# --- SymPy -------------------------------------------------------------------------------

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
_GLOBALS: dict = {"__builtins__": {}}
_GLOBALS.update({name: getattr(sympy, name) for name in (
    "Integer", "Rational", "Float", "Symbol", "Function", "sqrt", "root", "pi", "E", "I", "oo",
    "sin", "cos", "tan", "cot", "sec", "csc", "asin", "acos", "atan", "log", "ln", "exp", "Abs",
    "factorial", "binomial", "floor", "ceiling", "Interval", "FiniteSet", "Union",
)})
_LOCALS = {"e": sympy.E, "pi": sympy.pi}


def to_sympy(text: str):
    """Parse plain-text or LaTeX math into a SymPy object; None when it cannot."""
    if not is_safe(text):
        return None
    if "\\" in text:
        try:
            from latex2sympy2_extended import latex2sympy
            return latex2sympy(text)
        except Exception:
            return None
    s = text.replace("√", "sqrt").replace("π", "pi").replace("²", "**2").replace("³", "**3")
    try:
        return parse_expr(s, local_dict=dict(_LOCALS), global_dict=dict(_GLOBALS),
                          transformations=_TRANSFORMS)
    except Exception:
        return None


def _sympy_equal(expected: str, candidates: list[str]) -> bool | None:
    gold = to_sympy(expected)
    if gold is None:
        return None
    seen = False
    for cand in candidates:
        value = to_sympy(cand)
        if value is None:
            continue
        seen = True
        if _same(gold, value, cand):
            return True
    return False if seen else None


def _same(a, b, text: str = "") -> bool:
    items_a, items_b = _items(a), _items(b)
    if items_a is not None or items_b is not None:
        if items_a is None or items_b is None or len(items_a) != len(items_b):
            return False
        if _is_set(a) or _is_set(b):
            items_a, items_b = sorted(items_a, key=_sort_key), sorted(items_b, key=_sort_key)
        return all(_same(x, y) for x, y in zip(items_a, items_b))
    try:
        a, b = sympy.sympify(a), sympy.sympify(b)
        if a == b:
            return True
        symbols = a.free_symbols | b.free_symbols
        if not symbols:
            return _close(complex(a.evalf(30)), complex(b.evalf(30)), text)
        return _random_points_agree(a, b, sorted(symbols, key=str))
    except Exception:
        return False


def _items(obj):
    if isinstance(obj, (set, frozenset, tuple, list)):
        return list(obj)
    if isinstance(obj, (sympy.FiniteSet, sympy.Tuple)):
        return list(obj.args)
    return None


def _is_set(obj) -> bool:
    return isinstance(obj, (set, frozenset, sympy.FiniteSet))


def _sort_key(x):
    try:
        value = complex(sympy.N(x))
        return (0, value.real, value.imag)
    except Exception:
        return (1, str(x), 0.0)


def _close(x: complex, y: complex, text: str) -> bool:
    if abs(x - y) <= 1e-9 * max(1.0, abs(x)):
        return True
    places = _decimal_places(text)
    return places >= 3 and abs(x - y) <= 0.5 * 10**-places


def _random_points_agree(a, b, symbols, points: int = 5) -> bool:
    rng = random.Random(1234)
    checked = 0
    for _ in range(points * 3):
        subs = {s: sympy.Rational(rng.randint(1, 97), rng.randint(1, 13)) for s in symbols}
        try:
            x, y = complex(a.subs(subs).evalf(30)), complex(b.subs(subs).evalf(30))
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if not (abs(x) < float("inf") and abs(y) < float("inf")):
            continue
        if abs(x - y) > 1e-8 * max(1.0, abs(y)):
            return False
        checked += 1
        if checked == points:
            return True
    return False
