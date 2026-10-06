# LLMath

Türkçe matematik sorularını farklı büyük dil modelleriyle (LLM) çözen, cevapları cevap anahtarı ve SymPy ile katmanlı olarak doğrulayan ve modelleri doğruluk ile hata türleri açısından karşılaştıran sistem. Bitirme projesi, **pilot sürüm (v0.1)**.

Akış: soru → LLM çözümü (adım adım ya da SymPy kodu) → cevap çıkarımı → Türkçe biçim normalleştirme → doğrulama → veritabanı → pano.

## Hızlı başlangıç (Windows, PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .

# Yerel model (Ollama kurulu olmalı: https://ollama.com)
ollama pull qwen3:8b

# İsteğe bağlı: Gemini için ücretsiz anahtar (https://aistudio.google.com/apikey)
Copy-Item .env.example .env   # sonra .env içine GEMINI_API_KEY yazın
```

Aşağıdaki komutlar proje klasöründe, sanal ortamın Python'u ile çalıştırılır (`.\.venv\Scripts\python.exe` ya da ortam etkinse `python`).

| İş | Komut |
| --- | --- |
| Toplu deney (pilot) | `python -m llmath.runner --name pilot-1 --models qwen3-8b --strategies cot pot` |
| Yarıda kalan deneye devam | aynı komutu aynı `--name` ile tekrar çalıştırın |
| API + arayüz | `.\baslat.ps1` → arayüz http://localhost:8501, API belgeleri http://127.0.0.1:8000/docs |
| Bir çalıştırmanın Markdown raporu | `python scripts/pilot_report.py --name pilot-1 > outputs/pilot-1.md` |
| Testler | `python -m pytest` |
| Pilot veri setini yeniden üret | `python scripts/build_pilot_set.py` |

Kullanılabilir model anahtarları `llmath/config.py` içindeki `MODELS` sözlüğündedir: `qwen3-8b`, `qwen3-8b-think`, `llama3.1-8b`, `gemini-flash-lite`, `gemini-flash`. Ollama modelleri indirilmişse, bulut modelleri ise `.env` içinde anahtarı varsa kullanılabilir.

## Pilot veri seti (`data/pilot.jsonl`, 40 öğe)

| Alt küme | İçerik | Kaynak ve lisans |
| --- | --- | --- |
| P1 | 10 ilköğretim sözel problemi (Türkçe) | [GSM8K-TR](https://huggingface.co/datasets/bezir/gsm8k-tr), MIT |
| P2 | 10 MATH-500 problemi, Türkçe + İngilizce paralel | [MCLM MT-MATH100](https://huggingface.co/datasets/amphora/MCLM), MIT |
| P3 | 10 taze Türkçe şablon sorusu (LGS/YKS tarzı); cevaplar SymPy ile hesaplanır | Bu proje, CC BY 4.0 |

Makine çevirileri elle gözden geçirildi; düzeltmeler gerekçeleriyle birlikte `scripts/build_pilot_set.py` içinde (`CORRECTIONS`, `EXCLUDE`) ve her sorunun `note` alanında kayıtlıdır. Örnek: GSM8K-TR'deki bir sorunun cevap anahtarı hatalıydı (600 yerine 550).

## Doğrulama katmanları (`llmath/verify.py`)

1. **Çıkarım:** son `\boxed{}`, "Cevap:" satırı ya da son sayı (PoT'de programın son çıktı satırı).
2. **Türkçe normalleştirme** (`llmath/normalize.py`): ondalık virgül (`3,5`), binlik nokta (`1.250`), `%25`, birimler ve ekler (`25 TL'dir`), `x = 4`, `x = 2 veya x = -3`. Belirsiz yazımlar (`1.250` → 1250 mi, 1,25 mi?) her iki okumayla denenir ve işaretlenir.
3. **Metin eşitliği → sayısal eşitlik → Math-Verify → SymPy** (sabitlerde sayısal, değişkenli ifadelerde rastgele nokta testi).

Her kararın hangi katmanda verildiği veritabanına yazılır; böylece katmanların güvenilirliği insan etiketine göre ölçülebilir.

## Klasör yapısı

```
llmath/     çekirdek: ayarlar, LLM erişimi, istemler, çözücü, çıkarım, normalleştirme, doğrulama, sandbox, veritabanı
api/        FastAPI servisi
ui/         Streamlit arayüzü
scripts/    veri seti üretimi
data/       soru setleri (JSONL)
tests/      birim testleri
```

## Güvenlik notu

PoT stratejisinde modelin yazdığı kod ayrı bir Python sürecinde çalışır: yalnız `math`, `fractions` ve `sympy` gibi modüllere izin verilir, süre sınırı vardır ve süreç API anahtarlarını göremez. Bu pilot düzeyinde bir korumadır; tam yalıtım (ağsız Docker konteyneri) sonraki aşamada eklenecek.

## Lisans

Kod MIT lisanslıdır (`LICENSE`). Veri setlerinin lisansları yukarıdaki tabloda belirtilmiştir.
