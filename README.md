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
| Ana deney (500 soru) | `python -m llmath.runner --name main-1 --data data/main_v1.jsonl --models qwen3-8b` (modeller ayrı süreçlerde aynı anda çalışabilir) |
| Yarıda kalan deneye devam | aynı komutu aynı `--name` ile tekrar çalıştırın; bir model art arda 5 API hatası alırsa (ör. günlük kota) durur |
| API + arayüz | `baslat.bat` dosyasına çift tıklayın (ya da `.\baslat.ps1`) → arayüz http://localhost:8501, API belgeleri http://127.0.0.1:8000/docs. `baslat.bat` gerekirse Ollama'yı da açar, ikisi zaten çalışıyorsa yalnız tarayıcıyı açar |
| Bir çalıştırmanın Markdown raporu | `python scripts/report.py --name main-1 --cascade qwen3-8b gemini-flash-lite --out outputs/main-1.md` |
| Kademeli çözüm analizi (AS2) | `python -m llmath.cascade --run pilot-1 --cheap qwen3-8b --strong gemini-flash-lite` |
| Testler | `python -m pytest` |
| Hata etiketleme kod kitabı | `python -m llmath.taxonomy` |
| Bir çalıştırmayı güncel sandbox ve doğrulayıcıyla yeniden değerlendir | `python scripts/reverify.py --name main-1` (model yeniden çağrılmaz; değişenler `outputs/reverify-main-1-<zaman>.csv`) |
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

## İlk ana deney (main-1)

Ana setin 500 sorusu Qwen3 8B (yerel) ve Gemini 3.5 Flash-Lite (ücretsiz API) ile adım adım (CoT) ve SymPy koduyla (PoT) çözüldü: 2.000 cevap, sıcaklık 0, istem sürümü v0.1.

| Model | CoT | PoT |
| --- | --- | --- |
| Gemini 3.5 Flash-Lite | %97,8 (489/500) | %90,8 (454/500) |
| Qwen3 8B | %89,4 (447/500) | %78,8 (394/500) |

- **Türkçe–İngilizce:** Aynı sorular karşılaştırıldığında Qwen3 8B Türkçede anlamlı biçimde geride kalıyor. CoT'ta fark GSM8K'da −9,2, MATH'te −11,3 puan (McNemar p = 0,003 ve 0,049). Gemini'de CoT için anlamlı fark yok; PoT'ta MATH sorularında fark −11,3 puan (p = 0,022).
- **CoT ve PoT:** CoT, iki modelde de iki dilde de PoT'den anlamlı ölçüde iyi.
- **Kademeli çözüm:** Qwen'in CoT ve PoT cevapları uyuşursa cevap kabul ediliyor, uyuşmazsa soru Gemini'ye gidiyor. Doğruluk %97,4; Gemini tek başına %97,8 (p = 0,754). Gemini 500 yerine 126 kez çağrılıyor.
- **Ondalık virgül:** İki model de PoT'ta Türkçe ondalık virgülü yanlış okuyor ("8,4" → `Rational(8, 4)`). Bu, pilotta görülen hatanın aynısı.

Ayrıntılı tablolar `python scripts/report.py --name main-1 --cascade qwen3-8b gemini-flash-lite --out outputs/main-1.md` ile üretilir.

## Hata etiketleme (`llmath/taxonomy.py`)

Her yanlış cevapta ilk hatalı adım bulunur. Ardından iki etiket verilir: hatanın doğduğu Newman aşaması (okuma, anlama, dönüştürme, işlem, kodlama) ve 13 hata türünden biri. H1–H9 Li vd. (2024)'ten alındı; H10–H13 bu projede eklendi, H10 Türkçeye özgüdür (Türkçe ifadeyi ya da sayı yazımını yanlış okuma). İki ölçüm etiketi model hatası sayılmaz: Y1 "cevap aslında doğru" doğrulayıcının yanlış negatiflerini, Y2 "soru ya da cevap anahtarı hatalı" veri hatalarını kaydeder. Her türün tanımı ve Türkçe örneği kod kitabındadır (`python -m llmath.taxonomy`).

Etiketleme arayüzün **Hata etiketleme** sayfasından yapılır. Yanlış cevaplar sabit bir rastgele sırayla gelir, böylece ilk N etiket rastgele bir örneklem olur. Başka etiketleyicilerin ve LLM-hakemin etiketleri gösterilmez. Plan, LLM-hakemin bütün yanlışlara ön etiket vermesi ve en az 150 insan etiketiyle uyumun kappa ile ölçülmesidir; LLM-hakem henüz eklenmedi.

## Klasör yapısı

```
llmath/     çekirdek: ayarlar, LLM erişimi, istemler, çözücü, çıkarım, normalleştirme, doğrulama, sandbox,
            veritabanı, hata taksonomisi
api/        FastAPI servisi
ui/         Streamlit arayüzü (soru çözme, sonuçlar, hata etiketleme)
scripts/    veri seti üretimi, çalıştırma raporu
data/       soru setleri (JSONL)
tests/      birim testleri
```

## Güvenlik notu

PoT stratejisinde modelin yazdığı kod ayrı bir Python sürecinde çalışır. Yalnız dosya, ağ ya da süreç erişimi olmayan hesaplama modüllerine izin verilir: `math`, `fractions`, `decimal`, `sympy`, `itertools`, `collections`, `datetime` vb. Ayrıca süre sınırı vardır ve süreç API anahtarlarını göremez. Bu pilot düzeyinde bir korumadır; tam yalıtım (ağsız Docker konteyneri) sonraki aşamada eklenecek.

## Lisans

Kod MIT lisanslıdır (`LICENSE`). Veri setlerinin lisansları yukarıdaki tabloda belirtilmiştir.
