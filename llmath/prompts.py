"""Prompt templates. Any wording change must bump PROMPT_VERSION.

PoT also has two experimental variants for the Turkish number-format study; the base variant is the
v0.1 prompt word for word. A variant's answers are stored with the version "v0.1-<variant>".
The feedback prompts of llmath.refine are stored as "v0.1-refine"; they never contain the answer key.
"""

from __future__ import annotations

PROMPT_VERSION = "v0.1"
VARIANTS = ("base", "norational", "trnote")
REFINE_VERSION = f"{PROMPT_VERSION}-refine"
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

# F1: what went wrong with a PoT program, then the request to fix it.
FIX_FEEDBACK = {
    "tr": {
        "no_code": "Cevabında ```python bloğu bulunamadı.",
        "blocked": "Program güvenlik denetiminden geçmedi ({detail}). Yalnızca math, fractions ve sympy kullanabilirsin.",
        "timeout": "Program 15 saniye içinde bitmedi.",
        "empty": "Program hiçbir şey yazdırmadı.",
        "error": "Program çalışırken hata verdi:\n```\n{detail}\n```",
        "ask": ("Hatayı düzelt ve programın tamamını yine tek bir ```python bloğu içinde ver. Program nihai "
                "cevabı en sonda tek satır olarak print() ile yazdırsın. Kod dışında açıklama yazma."),
    },
    "en": {
        "no_code": "Your reply did not contain a ```python block.",
        "blocked": "The program did not pass the safety check ({detail}). You may only use math, fractions and sympy.",
        "timeout": "The program did not finish within 15 seconds.",
        "empty": "The program printed nothing.",
        "error": "The program raised an error:\n```\n{detail}\n```",
        "ask": ("Fix the problem and give the whole program again in a single ```python block. The program must "
                "print the final answer as a single line at the end. Do not write any explanation outside the code."),
    },
}

# F2: both solutions go back to the model when their answers disagree; neither is marked as right.
CONSISTENCY = {
    "tr": {
        "intro": "Bu soru iki yöntemle çözüldü ve iki çözüm aynı sonuca varmadı.",
        "cot": "1. Adım adım çözüm:\n{solution}\n\nBu çözümün cevabı: {answer}",
        "pot": "2. Python programı:\n```python\n{code}\n```\nProgramın çıktısı: {answer}",
        "no_answer": "cevap bulunamadı",
        "no_code": "2. Python programı: program yazılamadı.",
        "ask": ("Çözümlerden biri ya da ikisi de yanlış olabilir. İkisini de kontrol et, hatayı bul ve soruyu adım "
                "adım yeniden çöz. En sonda nihai cevabı yalnızca bir kez \\boxed{...} içinde yaz. Kesirli cevapları "
                "sadeleştirilmiş kesir olarak ver ve kutunun içine birim yazma."),
    },
    "en": {
        "intro": "This problem was solved in two ways, and the two solutions reached different results.",
        "cot": "1. Step-by-step solution:\n{solution}\n\nAnswer of this solution: {answer}",
        "pot": "2. Python program:\n```python\n{code}\n```\nProgram output: {answer}",
        "no_answer": "no answer found",
        "no_code": "2. Python program: no program was written.",
        "ask": ("One or both solutions may be wrong. Check both, find the mistake and solve the problem again step by "
                "step. At the end, write the final answer exactly once inside \\boxed{...}. Give fractional answers "
                "as simplified fractions and do not put units inside the box."),
    },
}


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


def fix_feedback(kind: str, detail: str = "", lang: str = "tr") -> str:
    """F1 message: what went wrong (no_code, blocked, timeout, empty, error) and the request to fix it."""
    text = FIX_FEEDBACK["en" if lang == "en" else "tr"]
    return f"{text[kind].format(detail=detail.strip())}\n\n{text['ask']}"


def _clip(text: str, limit: int) -> str:
    """Long texts keep their start and their end, where a solution's final answer is."""
    text = text.strip()
    return text if len(text) <= limit else f"{text[: limit // 2]}\n[...]\n{text[-(limit // 2):]}"


def consistency_messages(question: str, lang: str, solution: str, cot_answer: str | None, code: str | None,
                         pot_output: str | None) -> list[dict[str, str]]:
    """F2 conversation: the question, the CoT solution and the PoT program with their answers."""
    lang = "en" if lang == "en" else "tr"
    text = CONSISTENCY[lang]
    pot = (text["pot"].format(code=_clip(code, 3000), answer=pot_output or text["no_answer"]) if code
           else text["no_code"])
    user = "\n\n".join([question.strip(), text["intro"],
                        text["cot"].format(solution=_clip(solution, 4000), answer=cot_answer or text["no_answer"]),
                        pot, text["ask"]])
    return [{"role": "system", "content": SYSTEM["cot"][lang]}, {"role": "user", "content": user}]
