import pytest

from llmath.prompts import PROMPT_VERSION, build_messages, prompt_version_for

# The v0.1 PoT prompts exactly as they were when main-1 ran; the base variant must not drift.
V01_POT = {
    "tr": ("Bu soruyu çözen kısa bir Python programı yaz. Yalnızca math, fractions ve sympy kullanabilirsin. "
           "Kodu tek bir ```python bloğu içinde ver. Program nihai cevabı en sonda tek satır olarak print() ile "
           "yazdırsın. Kesirli sonuçlarda sympy.Rational kullan, ondalığa çevirme. Kod dışında açıklama yazma."),
    "en": ("Write a short Python program that solves this problem. You may only use math, fractions and sympy. "
           "Put the code in a single ```python block. The program must print the final answer as a single line at "
           "the end. Use sympy.Rational for fractional results instead of converting to decimals. Do not write any "
           "explanation outside the code."),
}


@pytest.mark.parametrize("lang", ["tr", "en"])
def test_base_pot_prompt_is_unchanged(lang):
    user = build_messages("Soru {x}?", "pot", lang)[1]["content"]
    assert user == "Soru {x}?\n\n" + V01_POT[lang]
    assert prompt_version_for("base") == PROMPT_VERSION


def test_variants_change_only_their_part():
    base = build_messages("S", "pot", "tr")[1]["content"]
    no_hint = build_messages("S", "pot", "tr", "norational")[1]["content"]
    note = build_messages("S", "pot", "tr", "trnote")[1]["content"]
    assert "Rational" in base and "Rational" not in no_hint
    assert no_hint == base.replace(" Kesirli sonuçlarda sympy.Rational kullan, ondalığa çevirme.", "")
    assert note.startswith("S\n\nSorudaki sayılar Türkçe yazımla verilmiştir") and note.endswith(base[3:])
    # the note is about Turkish numbers; English questions keep the base prompt
    assert build_messages("S", "pot", "en", "trnote") == build_messages("S", "pot", "en")
    assert prompt_version_for("trnote") == f"{PROMPT_VERSION}-trnote"
    with pytest.raises(ValueError):
        build_messages("S", "pot", "tr", "baska")
