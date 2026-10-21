"""Prompt templates. Any wording change must bump PROMPT_VERSION.

PoT also has two experimental variants for the Turkish number-format study; the base variant is the
v0.1 prompt word for word. A variant's answers are stored with the version "v0.1-<variant>".
"""

from __future__ import annotations

PROMPT_VERSION = "v0.1"
VARIANTS = ("base", "norational", "trnote")
POT_RATIONAL = {
    "tr": " Kesirli sonuçlarda sympy.Rational kullan, ondalığa çevirme.",
    "en": " Use sympy.Rational for fractional results instead of converting to decimals.",
}
# Added to Turkish questions only, to see whether naming the convention prevents "8,4" -> Rational(8, 4).
TR_NUMBER_NOTE = ("Sorudaki sayılar Türkçe yazımla verilmiştir: ondalık ayırıcı virgül (2,5 = 2.5), "
                  "binlik ayırıcı noktadır (1.250 = 1250). ")

SYSTEM = {
    "cot": {
        "tr": "Sen dikkatli bir matematik öğretmenisin. Soruları Türkçe, adım adım ve açık biçimde çözersin.",
        "en": "You are a careful mathematics teacher. You solve problems step by step and explain each step clearly.",
    },
    "pot": {
        "tr": "Sen matematik sorularını Python ve SymPy koduyla çözen bir asistansın.",
        "en": "You are an assistant that solves mathematics problems by writing Python and SymPy code.",
    },
}

USER = {
    "cot": {
        "tr": (
            "{question}\n\n"
            "Soruyu adım adım çöz ve her adımı kısaca açıkla. En sonda nihai cevabı yalnızca bir kez "
            "\\boxed{{...}} içinde yaz (ör. \\boxed{{42}}). Kesirli cevapları sadeleştirilmiş kesir "
            "olarak ver ve kutunun içine birim yazma."
        ),
        "en": (
            "{question}\n\n"
            "Solve the problem step by step and briefly explain each step. At the end, write the final "
            "answer exactly once inside \\boxed{{...}} (e.g. \\boxed{{42}}). Give fractional answers as "
            "simplified fractions and do not put units inside the box."
        ),
    },
    "pot": {
        "tr": (
            "{question}\n\n{note}"
            "Bu soruyu çözen kısa bir Python programı yaz. Yalnızca math, fractions ve sympy "
            "kullanabilirsin. Kodu tek bir ```python bloğu içinde ver. Program nihai cevabı en sonda "
            "tek satır olarak print() ile yazdırsın.{rational} Kod dışında açıklama yazma."
        ),
        "en": (
            "{question}\n\n{note}"
            "Write a short Python program that solves this problem. You may only use math, fractions "
            "and sympy. Put the code in a single ```python block. The program must print the final "
            "answer as a single line at the end.{rational} Do not write any explanation outside the code."
        ),
    },
}

STRATEGIES = {"cot": "Adım adım çözüm (CoT)", "pot": "SymPy koduyla çözüm (PoT)"}


def prompt_version_for(variant: str = "base") -> str:
    return PROMPT_VERSION if variant == "base" else f"{PROMPT_VERSION}-{variant}"


def build_messages(question: str, strategy: str, lang: str = "tr", variant: str = "base") -> list[dict[str, str]]:
    if variant not in VARIANTS:
        raise ValueError(f"bilinmeyen istem varyantı: {variant}")
    lang = lang if lang in ("tr", "en") else "tr"
    user = USER[strategy][lang].format(
        question=question.strip(),
        note=TR_NUMBER_NOTE if variant == "trnote" and lang == "tr" else "",
        rational="" if variant == "norational" else POT_RATIONAL[lang],
    )
    return [{"role": "system", "content": SYSTEM[strategy][lang]}, {"role": "user", "content": user}]
