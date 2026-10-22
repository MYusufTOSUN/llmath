import sys
from pathlib import Path

import pytest

from llmath import translation_v2 as tv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_main_v2  # noqa: E402


def test_translation_file_blocks_notes_and_duplicates(tmp_path):
    path = tmp_path / "p.md"
    path.write_text("# başlık\n\n### m-p1-0001\nBirinci satır.\nİkinci satır.\n> Not: bir not\n\n### m-p1-0002\nX\n",
                    encoding="utf-8")
    blocks = tv2.read_translations(path)
    assert blocks["m-p1-0001"] == {"text": "Birinci satır.\nİkinci satır.", "notes": ["Not: bir not"]}
    path.write_text("### a\nx\n### a\ny\n", encoding="utf-8")
    with pytest.raises(ValueError):
        tv2.read_translations(path)


def test_figures_are_copied_from_the_english():
    english = "See [asy]draw((0,0)--(1,1));[/asy] and [asy]dot((0,0));[/asy]."
    assert tv2.with_figures("Bak [ŞEKİL] ve [ŞEKİL].", english) == \
        "Bak [asy]draw((0,0)--(1,1));[/asy] ve [asy]dot((0,0));[/asy]."
    assert tv2.without_figures(english) == "See [ŞEKİL] and [ŞEKİL]."
    with pytest.raises(ValueError):
        tv2.with_figures("Şekil yok.", english)


def test_latest_review_counts_and_a_correction_replaces_the_text(tmp_path):
    path = tmp_path / "inceleme.jsonl"
    tv2.save_review("m-p1-0001", "duzeltme", text="Düzeltilmiş metin.", comment="ek yanlış", path=path)
    tv2.save_review("m-p1-0002", "onay", path=path)
    tv2.save_review("m-p1-0002", "duzeltme", text="Sonradan düzeltildi.", path=path)
    reviews = tv2.load_reviews(path)
    assert reviews["m-p1-0001"]["comment"] == "ek yanlış" and reviews["m-p1-0002"]["status"] == "duzeltme"
    item = {"pair_id": "m-p1-0002", "tr": "Claude'un çevirisi."}
    assert tv2.current_text(item, reviews) == "Sonradan düzeltildi."
    assert tv2.current_text({"pair_id": "m-p1-0003", "tr": "Aynen."}, reviews) == "Aynen."
    with pytest.raises(ValueError):
        tv2.save_review("m-p1-0001", "duzeltme", text="  ", path=path)
    with pytest.raises(ValueError):
        tv2.save_review("m-p1-0001", "belki", path=path)


def test_all_200_translations_keep_the_numbers_and_formulas_of_the_english():
    items = tv2.load_items()
    assert len(items) == 200 and {i["subset"] for i in items} == {"P1", "P2"}
    texts, report = build_main_v2.final_texts(items, {})
    assert report == [] and len(texts) == 200


def test_checks_catch_a_changed_number_or_formula():
    assert build_main_v2.problems("P1", "He has $2.74 and 88,000 pens.", "2,74 doları ve 88.000 kalemi var.") == []
    assert build_main_v2.problems("P1", "He has 3 friends.", "Üç arkadaşı var.")       # number written out
    assert build_main_v2.problems("P2", "Find $w.$", "$w$ sayısını bulun.") == []      # punctuation may move
    assert build_main_v2.problems("P2", "Find $x^2$.", "$x^3$ değerini bulun.")         # formula changed
