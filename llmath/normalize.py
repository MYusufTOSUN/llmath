"""Normalize answer strings, with Turkish number formats in mind.

Turkish writes 3,5 for 3.5 and 1.250 for 1250. A model may follow either
convention, so an ambiguous string yields several candidate readings and the
verifier accepts a match on any of them. Gold answers are stored in a single
canonical form (dot decimals, no thousands separators, LaTeX for expressions)
and only have their wrappers removed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_NUMBER = r"[+-]?\d[\d.,]*"
# A number followed by a unit or a counted noun: "25 TL'dir", "12 km/saat", "15 kasa", "864 in^2".
_WORD = r"[^\W\d_]+(?:[/'’.][^\W\d_]*)*(?:\^\{?[23]\}?|[²³])?"
_NUMBER_WITH_UNIT = re.compile(rf"^(?P<num>{_NUMBER})\s*(?P<unit>°|{_WORD}(?:\s+{_WORD}){{0,3}})$")
_UNIT_LETTERS = {"m", "g", "l", "L", "s", "h"}       # one-letter words that are units, not variables
_MATH_WORDS = {"pi", "sqrt", "log", "ln", "sin", "cos", "tan", "cot", "sec", "csc", "exp", "mod"}
_CURRENCY = re.compile(r"^(?:\\?\$|₺|€|£)\s*|\s*(?:\\?\$|₺|€|£)$")
_TEXT_UNIT = re.compile(r"\s*\\(?:text|mathrm|mbox)\{[^{}]*\}(?:\^\{?[23]\}?|[²³])?$")
_TURKISH_SUFFIX = re.compile(r"^(?P<num>" + _NUMBER + r")['’][a-zçğıöşü]+$", re.IGNORECASE)
_PERCENT = re.compile(r"^(?:%\s*(?P<a>" + _NUMBER + r")|(?P<b>" + _NUMBER + r")\s*%)$")
_ASSIGNMENT = re.compile(r"^(?:[a-zA-Z](?:_\{?\w+\}?)?|Ç\.?\s*K\.?)\s*=\s*(?P<rhs>[^=]+)$")   # Ç.K. = {...}
_CHOICE = re.compile(r"^\(?(?P<letter>[A-Ea-e])\)?[.)]?$")
_OR_WORDS = re.compile(r"\s+(?:veya|ya da|yahut|ve|or|and)\s+", re.IGNORECASE)   # "4 ve 20" lists solutions


@dataclass
class Normalized:
    candidates: list[str]
    ambiguous: bool = False          # Turkish and English readings disagree
    notes: list[str] = field(default_factory=list)


def strip_wrappers(text: str) -> str:
    """Remove math delimiters, \\boxed{}, Markdown emphasis and trailing punctuation."""
    s = text.strip()
    s = re.sub(r"^\*\*|\*\*$", "", s).strip()
    for left, right in (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
        if s.startswith(left) and s.endswith(right) and len(s) > len(left) + len(right):
            s = s[len(left): len(s) - len(right)].strip()
    boxed = _unwrap_command(s, r"\boxed") or _unwrap_command(s, r"\fbox")
    if boxed is not None:
        s = boxed.strip()
    s = s.rstrip(".。").strip()
    return s


def _unwrap_command(s: str, command: str) -> str | None:
    """Return the argument of `command{...}` when it spans the whole string."""
    if not s.startswith(command + "{") or not s.endswith("}"):
        return None
    depth = 0
    for i, ch in enumerate(s[len(command):], start=len(command)):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[len(command) + 1: i] if i == len(s) - 1 else None
    return None


def _brace_frac_arguments(s: str) -> str:
    """\\frac34, \\frac 34 and \\frac9{19} -> \\frac{3}{4}, \\frac{9}{19} (LaTeX takes one token)."""
    out, i = [], 0
    while (j := s.find(r"\frac", i)) != -1:
        out.append(s[i:j + 5])
        k = j + 5
        args = []
        for _ in range(2):
            while k < len(s) and s[k] == " ":
                k += 1
            if k < len(s) and s[k] == "{":
                depth, start = 0, k
                while k < len(s):
                    depth += {"{": 1, "}": -1}.get(s[k], 0)
                    k += 1
                    if depth == 0:
                        break
                args.append(s[start:k])
            elif k < len(s):
                args.append("{" + s[k] + "}")
                k += 1
        out.append("".join(args))
        i = k
    out.append(s[i:])
    return "".join(out)


def clean_latex(s: str) -> str:
    s = re.sub(r"\\[dt]frac", r"\\frac", s)
    s = _brace_frac_arguments(s)
    s = re.sub(r"√\s*(\d+|[a-zA-Z])", r"\\sqrt{\1}", s)      # 11√2 -> 11\sqrt{2}
    s = re.sub(r"\\sqrt\s*([0-9a-zA-Z])", r"\\sqrt{\1}", s)   # \sqrt2 -> \sqrt{2}
    s = re.sub(r"\\(?:left|right|displaystyle|!|,|;|:|\s)", "", s)
    s = re.sub(r"\^\{?\\circ\}?", "", s)
    s = re.sub(r"\^\{?\\(?:text|mathrm)\{(?:st|nd|rd|th)\}\}?", "", s)   # 12^{\text{th}} -> 12
    s = re.sub(r"\\(?:text|mathrm|textbf|mbox)\{([^{}]*)\}", r"\1", s)
    s = s.replace(r"\%", "%").replace("−", "-").replace("·", "*").replace("×", "*")
    return re.sub(r"\s+", " ", s).strip()


def number_readings(token: str) -> tuple[list[str], bool]:
    """Canonical decimal readings of a number token and whether they disagree.

    >>> number_readings("3,5")
    (['3.5'], False)
    >>> number_readings("1.250")
    (['1250', '1.250'], True)
    """
    sign = "-" if token.startswith("-") else ""
    body = token.lstrip("+-")
    has_dot, has_comma = "." in body, "," in body
    if has_dot and has_comma:
        if body.rfind(",") > body.rfind("."):          # 1.250,75 (Turkish)
            return [sign + body.replace(".", "").replace(",", ".")], False
        return [sign + body.replace(",", "")], False   # 1,250.75 (English)
    if has_comma:
        if re.fullmatch(r"\d{1,3}(,\d{3})+", body):
            if body.count(",") > 1:                    # 1,250,000 can only be English
                return [sign + body.replace(",", "")], False
            return [sign + body.replace(",", "."), sign + body.replace(",", "")], True
        return [sign + body.replace(",", ".")], False  # 3,5 -> 3.5
    if has_dot and re.fullmatch(r"\d{1,3}(\.\d{3})+", body):
        if body.count(".") > 1:                        # 1.250.000 can only be Turkish
            return [sign + body.replace(".", "")], False
        return [sign + body.replace(".", ""), sign + body], True
    return [sign + body], False


def normalize_gold(text: str) -> str:
    """Gold answers are already canonical; only wrappers and `x =` are removed."""
    s = clean_latex(strip_wrappers(text))
    assignment = _ASSIGNMENT.match(s)
    return assignment["rhs"].strip() if assignment else s


def _is_unit(words: str) -> bool:
    """Words after a number are a unit unless they could be math: a variable or a function."""
    for word in re.split(r"[\s/]+", words):
        bare = re.sub(r"['’.].*|\^.*|[²³°]", "", word)
        if bare and ((len(bare) == 1 and bare not in _UNIT_LETTERS) or bare.lower() in _MATH_WORDS):
            return False
    return True


def normalize_answer(text: str, answer_type: str | None = None) -> Normalized:
    """Turn a model's final answer into one or more canonical candidates."""
    notes: list[str] = []
    s = strip_wrappers(text)
    unit = _TEXT_UNIT.search(s)
    if unit and re.search(r"\d", s[:unit.start()]):     # "\frac{20000}{\pi} \text{ m}^2"
        s = s[:unit.start()]
        notes.append("birim kaldırıldı")
    s = clean_latex(s)
    without_currency = _CURRENCY.sub("", s)
    if without_currency != s:
        s = without_currency
        notes.append("birim kaldırıldı")

    if answer_type == "choice":
        choice = _CHOICE.match(s)
        if choice:
            return Normalized([choice["letter"].upper()], notes=["şık"])

    assignment = _ASSIGNMENT.match(s)
    if assignment and not _OR_WORDS.search(s):
        s = assignment["rhs"].strip()
        notes.append("atama kaldırıldı")

    if _OR_WORDS.search(s):
        parts = [normalize_answer(p).candidates[0] for p in _OR_WORDS.split(s) if p.strip()]
        return Normalized(["{" + ", ".join(parts) + "}"], notes=notes + ["çoklu cevap"])

    percent = _PERCENT.match(s)
    if percent:
        readings, ambiguous = number_readings(percent["a"] or percent["b"])
        value = readings[0]
        return Normalized([value, f"{value}/100"], ambiguous=True, notes=notes + ["yüzde"])

    for pattern, note in ((_NUMBER_WITH_UNIT, "birim kaldırıldı"), (_TURKISH_SUFFIX, "ek kaldırıldı")):
        match = pattern.match(s)
        if match and (pattern is not _NUMBER_WITH_UNIT or _is_unit(match["unit"])):
            s = match["num"]
            notes.append(note)
            break

    if re.fullmatch(_NUMBER, s):
        readings, ambiguous = number_readings(s)
        if ambiguous:
            notes.append("sayı biçimi belirsiz")
        return Normalized(readings, ambiguous=ambiguous, notes=notes)

    # Expressions: a comma followed by a digit inside a number is a Turkish decimal
    # only when the expression has no other commas (tuples, sets, intervals use them).
    if "," in s and not re.search(r"[(\[{]", s) and re.fullmatch(r"[\d\s,+\-*/^.√πa-zA-Z\\{}]+", s):
        readings = [re.sub(r"(\d),(\d)", r"\1.\2", s)]
        if readings[0] != s:
            notes.append("ondalık virgül")
        return Normalized(readings, notes=notes)
    return Normalized([s], notes=notes)
