"""Prompt templates. Any wording change must bump PROMPT_VERSION."""

from __future__ import annotations

PROMPT_VERSION = "v0.1"

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
            "{question}\n\n"
            "Bu soruyu çözen kısa bir Python programı yaz. Yalnızca math, fractions ve sympy "
            "kullanabilirsin. Kodu tek bir ```python bloğu içinde ver. Program nihai cevabı en sonda "
            "tek satır olarak print() ile yazdırsın. Kesirli sonuçlarda sympy.Rational kullan, "
            "ondalığa çevirme. Kod dışında açıklama yazma."
        ),
        "en": (
            "{question}\n\n"
            "Write a short Python program that solves this problem. You may only use math, fractions "
            "and sympy. Put the code in a single ```python block. The program must print the final "
            "answer as a single line at the end. Use sympy.Rational for fractional results instead of "
            "converting to decimals. Do not write any explanation outside the code."
        ),
    },
}

STRATEGIES = {"cot": "Adım adım çözüm (CoT)", "pot": "SymPy koduyla çözüm (PoT)"}


def build_messages(question: str, strategy: str, lang: str = "tr") -> list[dict[str, str]]:
    lang = lang if lang in ("tr", "en") else "tr"
    return [
        {"role": "system", "content": SYSTEM[strategy][lang]},
        {"role": "user", "content": USER[strategy][lang].format(question=question.strip())},
    ]
