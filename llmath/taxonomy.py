"""Error taxonomy and annotation codebook: Newman stage + error type for every wrong answer.

H1–H9 come from Li et al. (2024); H10–H13 are new in this project (H10 is specific to Turkish).
Y1 and Y2 are not model errors: they record verifier mistakes (research question AS4) and faulty
questions, so that neither is counted against a model. Every type has a definition and a Turkish
example so that people and the LLM judge label alike. The author reviewed the codebook on 2026-10-21
(judge version j0.3): the H3/H10 boundary, a narrower H11, H5 against H12, unused data under H8,
misunderstanding the question under H6, and H13 recorded for every cut-off output.

Kod kitabını yazdırmak için: python -m llmath.taxonomy
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .sandbox import TIMEOUT_MESSAGE

# Newman (1977): the stage of the solution in which the error first appears.
NEWMAN_STAGES: dict[str, tuple[str, str]] = {
    "okuma": ("Okuma", "Sorudaki sözcük, sayı ya da sembolleri yanlış okumak."),
    "anlama": ("Anlama", "Sorunun ne istediğini ya da veriler arasındaki ilişkiyi yanlış anlamak."),
    "donusturme": ("Dönüştürme", "Doğru anlaşılan problemi uygun işlemlere ya da matematiksel modele çevirememek."),
    "islem": ("İşlem", "Seçilen işlemleri yürütürken hata yapmak."),
    "kodlama": ("Kodlama", "Doğru bulunan sonucu istenen nicelik ve biçimde ifade edememek."),
}


@dataclass(frozen=True)
class ErrorType:
    code: str
    name: str
    stage: str | None   # the usual Newman stage; the annotator may choose another
    source: str         # "Li vd. 2024", "yeni" or "ölçüm"
    definition: str
    example: str


ERROR_TYPES: dict[str, ErrorType] = {t.code: t for t in [
    ErrorType("H1", "Hesaplama hatası", "islem", "Li vd. 2024",
              "Doğru kurulmuş bir işlemin sonucunu yanlış hesaplamak.",
              "48 × 7 = 326 yazmak (doğrusu 336)."),
    ErrorType("H2", "Sayma hatası", "islem", "Li vd. 2024",
              "Nesneleri, durumları ya da olasılıkları eksik veya fazla saymak.",
              "İki zarda toplamı 7 olan durumları 5 tane saymak (doğrusu 6)."),
    ErrorType("H3", "Bağlam değeri hatası", "okuma", "Li vd. 2024",
              "Sorudaki bir sayıyı yanlış aktarmak ya da başka bir nicelikle karıştırmak. Sayının Türkçe "
              "yazımından (ondalık virgül, binlik nokta) doğan yanlış okumalar H3 değil H10'dur.",
              "Soruda 35 TL olan fiyatı çözümde 53 TL almak; öğrenci sayısı yerine sınıf sayısını kullanmak."),
    ErrorType("H4", "Halüsinasyon", "anlama", "Li vd. 2024",
              "Soruda olmayan bir bilgiyi, koşulu ya da sayıyı çözüme katmak.",
              "Soruda indirim yokken fiyata %10 indirim uygulamak."),
    ErrorType("H5", "Birim dönüşümü hatası", "donusturme", "Li vd. 2024",
              "Çözüm sırasında birimler arasında yanlış dönüşüm yapmak ya da gereken dönüşümü atlamak. Son "
              "cevabın istenen birimde verilmemesi H12'dir.",
              "90 dakikayı 1,3 saat almak; metre ile santimetreyi çevirmeden toplamak."),
    ErrorType("H6", "İşlem (operatör) hatası", "donusturme", "Li vd. 2024",
              "Gereken işlem yerine başka bir işlem seçmek. Sorulanı yanlış anlayıp başka bir nicelik "
              "hesaplamak da (dilden bağımsız yanlış anlama) H6'dır.",
              "\"3 katından 5 eksik\" ifadesinde 5 eklemek; toplam sorulurken fark hesaplamak."),
    ErrorType("H7", "Formül karışıklığı", "donusturme", "Li vd. 2024",
              "Yanlış formül ya da kural kullanmak.",
              "Dairenin alanı için 2πr kullanmak; sıralı seçimde kombinasyon kullanmak."),
    ErrorType("H8", "Eksik adım", "donusturme", "Li vd. 2024",
              "Çözüm için gerekli bir adımı ya da koşulu atlamak. Sorudaki bir veriyi hiç kullanmamak da H8'dir.",
              "Logaritmalı denklemde tanım kümesini kontrol etmeyip geçersiz kökü de cevaba katmak; soruda "
              "verilen \"6 öğrenci tenis oynuyor\" bilgisini hiç kullanmamak."),
    ErrorType("H9", "Çelişkili adım", "islem", "Li vd. 2024",
              "Önceki adımlarla ya da kendi içinde çelişen bir adım atmak.",
              "Bir adımda x = 4 bulup sonraki adımda x = 6 ile devam etmek."),
    ErrorType("H10", "Türkçe ifadeyi yanlış yorumlama", "okuma", "yeni",
              "Türkçe bir sözcüğün, ekin, kalıbın ya da Türkçe sayı yazımının (ondalık virgül, binlik nokta) "
              "anlamını yanlış okumak. Sayı yazımı hataları okuma, sözcük ve kalıp hataları çoğunlukla anlama "
              "aşamasındadır.",
              "\"6,5\" ondalık sayısını 6 ve 5 diye iki sayı okumak (PoT'de Rational(6, 5) yazmak); \"en az\" ile "
              "\"en çok\"u karıştırmak."),
    ErrorType("H11", "Kod/araç hatası", "islem", "yeni",
              "PoT kodunun çalışmaması (hata vermesi, cevabı yazdırmaması) ya da kurulan planın koda yanlış "
              "aktarılması (değişken karışıklığı, yanlış fonksiyon kullanımı). Planın kendisi yanlışsa tür H6 "
              "ya da H7'dir.",
              "Tanımlanmamış bir değişken kullanıp NameError almak; sonucu yazdırmayı unutmak; solve() "
              "sonucunu sözlük sanıp indekslemek."),
    ErrorType("H12", "Biçim/kodlama hatası", "kodlama", "yeni",
              "Doğru hesaplanmış bir sonucu son adımda istenen nicelik ya da biçim dışında vermek.",
              "Soru dakikayı sorarken cevabı saat olarak vermek; kesir istenirken yuvarlanmış ondalık vermek."),
    ErrorType("H13", "Yarım kalan çıktı", "kodlama", "yeni",
              "Çıktının token sınırı ya da zaman aşımı yüzünden yarıda kalması. Yarıda kalmadan önce görünür bir "
              "hata varsa tür olarak o hata seçilir, yoksa H13. Yarıda kalma her durumda ayrıca H13 olarak "
              "kaydedilir; bunu sistem kendisi yapar.",
              "Çözüm adımların ortasında kesilmiş (finish_reason = \"length\"); program 15 saniyede bitmemiş."),
    ErrorType("Y1", "Cevap aslında doğru (doğrulayıcı hatası)", None, "ölçüm",
              "Modelin cevabı doğru, doğrulayıcı reddetmiş. Model hatası sayılmaz; doğrulayıcının "
              "yanlış negatif oranını ölçer.",
              "Beklenen 1/2, modelin cevabı \"yarım\"."),
    ErrorType("Y2", "Soru ya da cevap anahtarı hatalı", None, "ölçüm",
              "Soru metni belirsiz ya da hatalı, veya cevap anahtarı yanlış. Model hatası sayılmaz.",
              "Cevap anahtarı 600, doğru cevap 550."),
]}

MEASUREMENT_CODES = frozenset(code for code, t in ERROR_TYPES.items() if t.source == "ölçüm")


def is_truncated(finish_reason: str | None, exec_error: str | None) -> bool:
    """Cut-off output: counted as H13 in addition to whatever its first error is labelled as."""
    return finish_reason == "length" or exec_error == TIMEOUT_MESSAGE


def solution_steps(solution: str) -> list[str]:
    """The numbered steps a CoT label refers to: paragraphs, or lines when there is a single paragraph.

    The labelling page and the LLM judge both number steps with this, so their step numbers agree.
    """
    parts = [p.strip() for p in re.split(r"\n\s*\n", solution) if p.strip()]
    return parts if len(parts) > 1 else [line for line in solution.splitlines() if line.strip()]


def codebook_markdown() -> str:
    lines = ["# Hata etiketleme kod kitabı", "", "## Newman aşamaları", "", "| Aşama | Tanım |", "| --- | --- |"]
    lines += [f"| {name} | {definition} |" for name, definition in NEWMAN_STAGES.values()]
    lines += ["", "## Hata türleri", "", "| Kod | Tür | Olağan aşama | Kaynak | Tanım | Örnek |",
              "| --- | --- | --- | --- | --- | --- |"]
    for t in ERROR_TYPES.values():
        stage = NEWMAN_STAGES[t.stage][0] if t.stage else "—"
        lines.append(f"| {t.code} | {t.name} | {stage} | {t.source} | {t.definition} | {t.example} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(codebook_markdown(), end="")
