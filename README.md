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
| Kademeli çözüm analizi (AS2) | `python -m llmath.cascade --run pilot-1 --cheap qwen3-8b --strong gemini-flash-lite` |
| Testler | `python -m pytest` |
| Pilot veri setini yeniden üret | `python scripts/build_pilot_set.py` |
| Ana veri setini yeniden üret | `python scripts/build_main_set.py` (inceleme listesi: `outputs/main_v1_review.md`) |

Kullanılabilir model anahtarları `llmath/config.py` içindeki `MODELS` sözlüğündedir: `qwen3-8b`, `qwen3-8b-think`, `llama3.1-8b`, `gemini-flash-lite`, `gemini-flash`. Ollama modelleri indirilmişse, bulut modelleri ise `.env` içinde anahtarı varsa kullanılabilir.

## Pilot veri seti (`data/pilot.jsonl`, 40 öğe)

| Alt küme | İçerik | Kaynak ve lisans |
| --- | --- | --- |
| P1 | 10 ilköğretim sözel problemi (Türkçe) | [GSM8K-TR](https://huggingface.co/datasets/bezir/gsm8k-tr), MIT |
| P2 | 10 MATH-500 problemi, Türkçe + İngilizce paralel | [MCLM MT-MATH100](https://huggingface.co/datasets/amphora/MCLM), MIT |
| P3 | 10 taze Türkçe şablon sorusu (LGS/YKS tarzı); cevaplar SymPy ile hesaplanır | Bu proje, CC BY 4.0 |

Makine çevirileri elle gözden geçirildi; düzeltmeler gerekçeleriyle birlikte `scripts/build_pilot_set.py` içinde (`CORRECTIONS`, `EXCLUDE`) ve her sorunun `note` alanında kayıtlıdır. Örnek: GSM8K-TR'deki bir sorunun cevap anahtarı hatalıydı (600 yerine 550).

## Ana veri seti (`data/main_v1.jsonl`, 500 öğe)

| Alt küme | İçerik | Kaynak ve lisans |
| --- | --- | --- |
| P1 | 120 GSM8K test problemi, Türkçe + İngilizce paralel (240 öğe) | [GSM8K](https://huggingface.co/datasets/openai/gsm8k) ve GSM8K-TR, MIT |
| P2 | 80 MATH-500 problemi, Türkçe + İngilizce paralel (160 öğe); konu ve zorluk düzeyi MATH-500'den | MCLM MT-MATH100, MIT |
| P3 | 100 taze Türkçe şablon sorusu (20 şablon × 5) | Bu proje, CC BY 4.0 |

Pilotta kullanılan sorular ana sete alınmadı. GSM8K-TR yerelleştirilmiş bir çeviridir: bazı sorularda sayılar değişmiş, bazılarında cevap anahtarı hatalı. Bu yüzden P1'e yalnızca Türkçe çevirisi dosyada beklenen sırada duran ve sayıları ile cevabı İngilizce aslıyla birebir eşleşen sorular alındı (1319 test sorusundan 622'si); cevap anahtarı İngilizce asıldan gelir. Sıra koşulu gereklidir, çünkü az sayılı sorularda başka bir problem rastlantıyla aynı sayıları ve aynı cevabı taşıyabiliyor (19 böyle eşleşme vardı). P1'in Türkçe metinlerinde ondalık ayırıcı virgül, binlik ayırıcı noktadır (2,74; 1.250). P2 metinleri LaTeX olduğu için formüllerdeki sayılar kaynaktaki gibi bırakıldı.

Otomatik kontrollerin işaretlediği 24 öğe incelendi, ardından 200 Türkçe metnin tamamı İngilizce aslıyla yan yana okundu. 14 Türkçe metin düzeltildi. Bunlardan anlamı değiştirenler şunlardı: "segments" kelimesinin "doğrular" diye çevrilmesi, "per meter" ifadesinin "metrekare" diye çevrilmesi, çeviride düşen "en az" ve "her biri", ayrıca üç soruda kaybolan şekil. İngilizce aslı belirsiz olan bir soru çıkarıldı. Kararlar `scripts/build_main_set.py` içinde (`CORRECTIONS`, `CHECKED`, `EXCLUDE`), özetleri `outputs/main_v1_review.md` dosyasındadır.

## Doğrulama katmanları (`llmath/verify.py`)

1. **Çıkarım:** son `\boxed{}`, "Cevap:" satırı ya da son sayı (PoT'de programın son çıktı satırı).
2. **Türkçe normalleştirme** (`llmath/normalize.py`): ondalık virgül (`3,5`), binlik nokta (`1.250`), `%25`, para birimleri (`\$221`, `₺300`), birimler ve ekler (`25 TL'dir`, `15 kasa`, `864 \text{ in}^2`), `x = 4`, `x = 2 veya x = -3`. Belirsiz yazımlar (`1.250` → 1250 mi, 1,25 mi?) her iki okumayla denenir ve işaretlenir. Sayıdan sonra gelen kelime değişken ya da fonksiyon olabiliyorsa (`2 x`, `3 pi`) silinmez.
3. **Metin eşitliği → sayısal eşitlik → Math-Verify → SymPy** (sabitlerde sayısal, değişkenli ifadelerde rastgele nokta testi).

Her kararın hangi katmanda verildiği veritabanına yazılır; böylece katmanların güvenilirliği insan etiketine göre ölçülebilir.

## Kademeli çözüm (`llmath/cascade.py`)

Ucuz model soruyu hem CoT hem PoT ile çözer. İki cevap uyuşursa kabul edilir, uyuşmazsa soru güçlü modele gönderilir. Analiz kayıtlı cevaplar üzerinde yapılır, ek API çağrısı gerektirmez; doğruluk (Wilson %95 güven aralığı), güçlü model çağrısı sayısı, maliyet ve süre raporlanır, güçlü modelle fark McNemar testiyle sınanır. Pilotta Qwen3 8B → Gemini Flash-Lite kademesi 40 sorunun 37'sini doğru çözdü (güçlü model tek başına 38/40) ve güçlü modeli 40 yerine 15 kez çağırdı.

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
