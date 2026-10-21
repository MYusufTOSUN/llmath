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
| Geri bildirimle düzeltme | `python -m llmath.refine --base main-1 --models gemini-flash-lite` (sonuçlar `main-1-refine` adıyla saklanır) |
| LLM-hakem ön etiketleri | `python -m llmath.judge --run main-1 --judge gemini-flash --models qwen3-8b`; insan etiketleriyle uyum için `--agreement` |
| Testler | `python -m pytest` |
| Hata etiketleme kod kitabı | `python -m llmath.taxonomy` |
| Bir çalıştırmayı güncel sandbox ve doğrulayıcıyla yeniden değerlendir | `python scripts/reverify.py --name main-1` (model yeniden çağrılmaz; değişenler `outputs/reverify-main-1-<zaman>.csv`) |
| Sayı yazımı deneyi | `python -m llmath.runner --name numfmt-1 --data data/number_format_v1.jsonl --models gemini-flash-lite`; PoT istem varyantları ayrı bir `--name` ile: `--strategies pot --prompt-variant norational` ya da `--strategies pot --langs tr --prompt-variant trnote` |
| Pilot veri setini yeniden üret | `python scripts/build_pilot_set.py` |
| Ana veri setini yeniden üret | `python scripts/build_main_set.py` (inceleme listesi: `outputs/main_v1_review.md`) |
| Sayı yazımı setini yeniden üret | `python scripts/build_number_format_set.py` |

Kullanılabilir model anahtarları `llmath/config.py` içindeki `MODELS` sözlüğündedir: `qwen3-8b`, `qwen3-8b-think`, `llama3.1-8b`, `deepseek-r1-8b`, `phi4-mini-reasoning`, `gemini-flash-lite`, `gemini-flash`. Ollama modelleri indirilmişse, bulut modelleri ise `.env` içinde anahtarı varsa kullanılabilir.

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

## Sayı yazımı deneyi (`data/number_format_v1.jsonl`, 80 öğe)

main-1'de iki model de PoT'ta Türkçe ondalık virgülü yanlış okudu. Bu deney hatanın ne kadar yaygın olduğunu ve istemle ilgisini ölçer. Set 40 Türkçe–İngilizce soru çiftinden oluşur (8 şablon × 5). Bir çiftin iki sorusunda sayılar ve cevap aynıdır, yalnız dil ve sayı yazımı değişir. D1'deki 30 çiftte ondalık sayılar (Türkçe 8,4; İngilizce 8.4), D2'deki 10 çiftte binlik ayırıcılı büyük sayılar (Türkçe 12.450; İngilizce 12,450) vardır. Cevaplar SymPy ile hesaplanır ve en fazla iki ondalık basamaklıdır; böylece kuruşa yuvarlayan bir model yanlış sayılmaz.

PoT istemi üç biçimde denenir:

| Varyant | İstemdeki fark | Sınanan soru |
| --- | --- | --- |
| `base` | Yok (main-1'deki v0.1 istemi) | Hata bu sette de çıkıyor mu? |
| `norational` | "Kesirli sonuçlarda sympy.Rational kullan" cümlesi yok | Hatayı bu ipucu mu tetikliyor? |
| `trnote` | Türkçe sorulara "ondalık ayırıcı virgül, binlik ayırıcı noktadır" notu eklenir | Açık bir uyarı hatayı önlüyor mu? |

`trnote` yalnız Türkçe istemi değiştirdiği için yalnız Türkçe sorularda çalıştırılır. Her varyant ayrı bir çalıştırma adıyla kaydedilir, istem sürümü (`v0.1`, `v0.1-norational`, `v0.1-trnote`) de her cevabın yanında saklanır.

**Gemini 3.5 Flash-Lite (21 Ekim 2026).** Yeni sette 160 cevabın hepsi doğru (CoT 80/80, PoT 80/80); hiçbir programda sayı yazımı yanlış okunmadı. Türkçe not eklenince model sayıları açıkça dönüştürdü (`"15,8".replace(",", ".")`) ve 40 sorunun hepsini doğru çözdü. Rational ipucu kaldırılınca doğruluk 78/80'e indi: model kayan noktaya geçti ve bir soruda `int(43800 * 0.35)` 15330 yerine 15329 verdi.

main-1'deki dört bölünme hatasının dördü de kumaş şablonundan çıktığı için bu kalıp ayrıca sınandı (`data/number_format_kumas_v1.jsonl`, 55 soru). Setin içeriği:
- K0: main-1'in beş kumaş sorusu, aynen.
- K1–K4: aynı kalıpta, kısa (4,2) ya da iki basamaklı (14,2) sayılar; sayılar virgülle ya da "ve" ile sıralanıyor.
- K1'in İngilizcesi kontrol grubu olarak.

45 Türkçe sorunun 3'ünde yanlış okuma görüldü. "7,5" → `Rational(7, 5)` yazıldı; yanlış cevap main-1'dekiyle aynı. "14,8" → `Rational(14, 8)` yazıldı. Bir programda da sayı dönüştürülmeden `Rational('7,5')` diye verildi. İngilizce kontrol grubu 10/10 doğru. Hücreler arasındaki farklar anlamlı değil.

Gemini'de görülen bölünmelerin hepsi Rational ipucunun olduğu istemde çıktı; ipucusuz istemle yazılan 105 programda bölünme yok ama iki kayan nokta hatası var (Fisher p = 0,26). Qwen3 8B'nin sonuçları bu farkın ipucundan kaynaklanmadığını gösterdi.

**Qwen3 8B (yerel, 21 Ekim 2026).**

| Koşul | Türkçe | İngilizce | Türkçe programlarda bölünme |
| --- | --- | --- | --- |
| CoT | 38/40 | 39/40 | — |
| PoT | 25/40 | 33/40 | 5/40 |
| PoT, Rational ipucu yok | 22/40 | 39/40 | 12/40 |
| PoT, Türkçe sayı notu | 15/40 | — | 19/40 |

- **CoT:** Sayı yazımından etkilenmiyor.
- **PoT:** Türkçe soruların 40'ından 25'i, İngilizce karşılıklarının 33'ü doğru (McNemar p = 0,039). İngilizce programların hiçbirinde bölünme yok.
- **Rational ipucu kaldırılınca:** İngilizce 39/40'a çıkıyor, Türkçe 22/40'a iniyor (Türkçe–İngilizce p < 0,001).
- **Türkçe not eklenince:** Durum düzelmiyor, kötüleşiyor (25/40'tan 15/40'a, p = 0,013). Not modeli virgüle odaklıyor ve model `benzin = Rational(8, 3)  # 8,3 litre = 83/10 litre` gibi programlar yazıyor: değeri yorum satırında doğru biliyor ama kodda virgülü argüman ayıracı olarak kopyalıyor.

Kumaş setinde Qwen'in 45 Türkçe programının 20'sinde bölünme var; bu oran yeni setteki 5/40'tan anlamlı biçimde yüksek (Fisher p = 0,002). İngilizce kontrol grubunda bölünme yok. Virgül ile "ve" arasında ve sayı uzunluğunda fark yok. Rational ipucu kaldırılınca Türkçe doğruluk 15/25'ten 4/25'e düşüyor (McNemar p < 0,001), yani Qwen'de ipucu bölünmeyi artırmıyor, azaltıyor.

Özetle:
- **Hata türü:** Türkçe ondalık virgülün PoT'de yanlış okunması (H10) küçük modelde sık ve gerçek bir hata. Temel istemle, kalıba göre Türkçe programların %12'sinde ile %44'ünde görülüyor.
- **Model farkı:** Güçlü modelde seyrek. Temel istemle Türkçe programlarda Gemini 3/85, Qwen 25/85 (Fisher p < 0,001).
- **İstemin etkisi:** Açık bir Türkçe sayı notu küçük modelde işe yaramıyor, ters tepiyor.

Tablolar: `python scripts/number_format_report.py --model qwen3-8b` (Gemini için `--model gemini-flash-lite`).

## Hata etiketleme (`llmath/taxonomy.py`)

Her yanlış cevapta ilk hatalı adım bulunur. Ardından iki etiket verilir: hatanın doğduğu Newman aşaması (okuma, anlama, dönüştürme, işlem, kodlama) ve 13 hata türünden biri. H1–H9 Li vd. (2024)'ten alındı; H10–H13 bu projede eklendi, H10 Türkçeye özgüdür (Türkçe ifadeyi ya da sayı yazımını yanlış okuma). İki ölçüm etiketi model hatası sayılmaz: Y1 "cevap aslında doğru" doğrulayıcının yanlış negatiflerini, Y2 "soru ya da cevap anahtarı hatalı" veri hatalarını kaydeder. Her türün tanımı ve Türkçe örneği kod kitabındadır (`python -m llmath.taxonomy`). Yarıda kalan bir çıktıda tür olarak görünen ilk hata seçilir; yarıda kalma ayrıca H13 olarak sayılır ve bunu sistem token sınırı ya da zaman aşımı kaydından kendisi çıkarır.

Etiketleme arayüzün **Hata etiketleme** sayfasından yapılır. Yanlış cevaplar sabit bir rastgele sırayla gelir, böylece ilk N etiket rastgele bir örneklem olur. Başka etiketleyicilerin ve LLM-hakemin etiketleri gösterilmez. Plan, LLM-hakemin bütün yanlışlara ön etiket vermesi ve en az 150 insan etiketiyle uyumun kappa ile ölçülmesidir.

**LLM-hakem** (`llmath/judge.py`) soruyu, cevap anahtarını, numaralı çözüm adımlarını (PoT'de kod satırlarını), program çıktısını ve doğrulayıcının kararını kod kitabıyla birlikte görür. JSON olarak hata türünü, Newman aşamasını, ilk hatalı adımı ve kısa bir gerekçe verir. Adımlar etiketleme sayfasındakiyle aynı biçimde numaralanır, böylece insan ve hakem aynı adım numarasını kullanır. Etiketler `llm:<hakem>/<sürüm>` adıyla saklanır (şu an `j0.3`); istem ya da kod kitabı değişince sürüm artırılır, eski etiketler karşılaştırma için kalır. Hakem, cevabı veren modelle aynı aileden olamaz: modeller kendi ailelerinin çıktısına daha hoşgörülü davranabildiği için Gemini, Gemini'nin cevaplarını etiketlemez. Damıtılmış modeller temel modelin ailesinden de sayılır; örneğin Ollama'daki DeepSeek-R1 8B, Qwen3-8B üzerine kurulu olduğu için Qwen'le aynı aileden kabul edilir. İnsan etiketleriyle uyum `--agreement` ile hesaplanır (hata türü ve aşama için Cohen kappa, ilk hatalı adımda birebir uyum).

İlk denemede (j0.1) hakem, "8,4 → `Rational(8, 4)`" gibi ondalık virgül hatalarını H10 yerine H3'e koydu; H3'ün örneğinde de ondalıklı bir sayı vardı. Bunun üzerine H3'ün tanımına "Türkçe sayı yazımından doğan yanlış okumalar H10'dur" kuralı eklendi ve örneği tam sayıyla değiştirildi. Kod kitabı 21 Ekim 2026'da baştan sona gözden geçirildi (j0.3). Bu gözden geçirmede şunlar değişti:
- H10'un olağan aşaması "okuma" oldu.
- H11 yalnız çalışmayan kodu ya da koda yanlış aktarılan planı kapsıyor.
- Çözüm sırasındaki birim hatası H5, son cevabın yanlış birimde verilmesi H12 sayılıyor.
- Sorudaki bir veriyi hiç kullanmamak H8'e girdi.
- Sorulanı yanlış anlayıp başka bir nicelik hesaplamak H6'ya girdi.

Qwen'in hatalarını Gemini 3.5 Flash-Lite, Gemini'nin hatalarını yerel Qwen3 8B (düşünme açık) etiketliyor. Gemini Flash daha güçlü olsa da ücretsiz katmanda günde yalnız 20 istek verdiği için kullanılmadı. Flash ile Flash-Lite'ın ortak 18 cevaptaki uyumu orta düzeyde (kappa 0,53), yani hakem seçimi etiketleri etkiliyor.

## Geri bildirimle düzeltme (`llmath/refine.py`)

Başlangıç noktası, modelin önceki bir çalıştırmadaki CoT ve PoT cevaplarıdır. Cevap anahtarı modele hiçbir aşamada gösterilmez.

- **F1, çalıştırma geri bildirimi:** Cevap üretmeyen PoT programı (kod yok, sandbox reddi, çalışma hatası, zaman aşımı, boş çıktı) hata mesajıyla birlikte modele geri gönderilir; en fazla iki tur.
- **F2, tutarlılık geri bildirimi:** CoT ve PoT cevapları hâlâ uyuşmuyorsa ya da biri eksikse model iki çözümü birlikte görür ve soruyu yeniden çözer.

İki cevap uyuşursa CoT cevabı kabul edilir, uyuşmazsa F2'nin cevabı. Her model çağrısı ayrı satır olarak saklanır (`pot_fix`, `consistency`), nihai karar da `refine` stratejisiyle. Böylece düzeltme; CoT, PoT ve kademeli çözümle aynı sorularda karşılaştırılabilir. Kademeli çözümden farkı, uyuşmazlığı güçlü bir modele göndermek yerine modelin kendisine çözdürmesidir. Hata mesajlarındaki yerel dosya yolları modele gönderilmeden önce silinir.

**İlk sonuç (Gemini 3.5 Flash-Lite, main-1 cevapları).**
- **F1:** Çalışmayan 27 programın hepsi düzeltme turundan sonra çalıştı, 18'i doğru cevap verdi. PoT doğruluğu %90,8'den %94,4'e çıktı.
- **F2:** Etkisi sınırlı kaldı. CoT ile PoT'nin uyuşmadığı 28 soruda model 3 yanlış cevabı düzeltti, 5 doğru cevabı bozdu.
- **Toplam:** Düzeltilmiş cevapların doğruluğu %97,4 (487/500). Tek başına CoT %97,8 (p = 0,727). Ek maliyet 59 çağrıdır.

Gerçek bir hata mesajı (F1) işe yarıyor; Gemini'de modelin iki çözümünü karşılaştırıp karar vermesi (F2) CoT'tan iyi değil. Bu, dış bir geri bildirim olmadan öz-düzeltmenin akıl yürütmeyi güvenilir biçimde iyileştirmediği bulgusuyla (Huang vd., 2024) uyumlu.

**Qwen3 8B (yerel, main-1 cevapları).** Sonuç Gemini'nin tersi:
- **F1:** Çalışmayan 45 programın 30'u çalıştı, 16'sı doğru cevap verdi. PoT %78,8'den %82,0'ye çıktı.
- **F2:** Uyuşmayan 110 soruda model 15 yanlışı düzeltti, hiçbir doğruyu bozmadı.
- **Toplam:** Düzeltilmiş cevaplar %92,4 (462/500). Tek başına CoT %89,4 (McNemar 15'e 0, p < 0,001). Ek maliyet 176 yerel çağrıdır.
- **Kademeli çözümle karşılaştırma:** Uyuşmazlığı güçlü modele gönderen kademeli çözüm aynı sorularda hâlâ açıkça daha iyi: %97,4, 126 Gemini çağrısıyla (29'a 4, p < 0,001).

Yani zayıf model için öz-düzeltme ücretsiz ve anlamlı bir kazanç sağlıyor, güçlü modelde ise kazanç yok.

## Klasör yapısı

```
llmath/     çekirdek: ayarlar, LLM erişimi, istemler, çözücü, çıkarım, normalleştirme, doğrulama, sandbox,
            veritabanı, hata taksonomisi, kademeli çözüm, geri bildirimle düzeltme, LLM-hakem
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
