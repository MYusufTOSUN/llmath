"""LLMath arayüzü (Streamlit).

Çalıştırma: önce API (python -m uvicorn api.main:app), sonra streamlit run ui/app.py
"""

from __future__ import annotations

import os
import re

import altair as alt
import httpx
import pandas as pd
import streamlit as st

from llmath.taxonomy import is_truncated, solution_steps

API_URL = os.getenv("LLMATH_API_URL", "http://127.0.0.1:8000")
STRATEGIES = {"cot": "Adım adım (CoT)", "pot": "SymPy kodu (PoT)"}
# Strategies that only appear in stored runs (llmath.refine), never in the solve form.
LABELS = {**STRATEGIES, "pot_fix": "PoT, hata geri bildirimiyle", "consistency": "Tutarlılık geri bildirimi",
          "refine": "Geri bildirimle düzeltilmiş"}
STATUS = {"correct": ("✅", "Doğru"), "incorrect": ("❌", "Yanlış"), "unparsed": ("⚠️", "Doğrulanamadı")}
INTERACTIVE_RUN = "etkileşimli"
# Human labels for the agreement (kappa) check with the LLM judge: all GSM8K and template errors of
# main-1 (91 answers), MATH items only when the annotator is sure (author's decision, 2026-10-21).
ANNOTATION_TARGET = 90
ANNOTATION_RUN = "main-1"   # the run the LLM judge labels, so human labels are collected there
ANNOTATION_SCOPE = "P1,P3"
SUBSET_LABELS = {"(hepsi)": "Hepsi", ANNOTATION_SCOPE: "P1 + P3 · insan etiketlemesinin kapsamı",
                 "P1": "P1 · GSM8K, ilköğretim sözel problemleri", "P2": "P2 · MATH, yarışma problemleri (zor)",
                 "P3": "P3 · Türkçe şablon soruları"}
# Categorical slots 1-2 of the reference palette, validated for light and dark surfaces.
SERIES = {"light": ("#2a78d6", "#eb6834"), "dark": ("#3987e5", "#d95926")}

st.set_page_config(page_title="LLMath", page_icon="∑", layout="wide")


def call(method: str, path: str, **kwargs):
    try:
        response = httpx.request(method, API_URL + path, timeout=900, **kwargs)
    except httpx.HTTPError as exc:
        st.error(f"API'ye ulaşılamadı ({API_URL}). Önce `python -m uvicorn api.main:app` ile başlatın.\n\n{exc}")
        st.stop()
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        st.error(f"API hatası ({response.status_code}): {detail}")
        st.stop()
    return response.json()


_MATH_SPAN = re.compile(r"\$\$.*?\$\$|\$[^$\n]+\$", re.DOTALL)
_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def as_markdown(text: str) -> str:
    """Streamlit renders $...$ and $$...$$, but models often write \\( \\) and \\[ \\], and leave the
    final \\boxed{...} outside any math, where it would show as raw LaTeX."""
    text = re.sub(r"\\\[(.+?)\\\]", lambda m: "$$" + m.group(1) + "$$", text, flags=re.DOTALL)
    text = re.sub(r"\\\((.+?)\\\)", lambda m: "$" + m.group(1) + "$", text, flags=re.DOTALL)
    pieces, last = [], 0
    for match in _MATH_SPAN.finditer(text):
        pieces += [_BOXED.sub(lambda m: f"${m.group(0)}$", text[last:match.start()]), match.group(0)]
        last = match.end()
    pieces.append(_BOXED.sub(lambda m: f"${m.group(0)}$", text[last:]))
    return "".join(pieces)


def wilson_low(correct: int, n: int, z: float = 1.96) -> float:
    """Lower end of the 95% Wilson interval: ranks 38/40 above 9/9, unlike raw accuracy."""
    if n == 0:
        return 0.0
    p = correct / n
    return ((p + z * z / (2 * n)) - z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / (1 + z * z / n)


def theme_mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except AttributeError:
        return "light"


# --- Soru çöz --------------------------------------------------------------------------------

def page_solve() -> None:
    st.header("Soru çöz")
    models = call("GET", "/models")
    available = {m["key"]: m["label"] for m in models if m["available"]}
    with st.sidebar:
        st.subheader("Modeller")
        for m in models:
            note = "" if m["available"] else f" — {m['reason']}"
            st.caption(("● " if m["available"] else "○ ") + m["label"] + note)
    if not available:
        st.warning("Kullanılabilir model yok. Ollama'yı başlatın ya da .env dosyasına API anahtarı ekleyin.")
        return

    mode = st.radio("Soru kaynağı", ["Pilot setinden seç", "Kendi sorunu yaz"], horizontal=True)
    if mode == "Pilot setinden seç":
        questions = call("GET", "/questions")
        left, right = st.columns([1, 4])
        subset = left.selectbox("Alt küme", sorted({q["subset"] for q in questions}))
        pool = [q for q in questions if q["subset"] == subset]
        question = right.selectbox("Soru", pool, format_func=lambda q: f"{q['id']} · {q['text'][:80]}")
        st.markdown(as_markdown(question["text"]))
        st.caption(f"Kaynak: {question['source']}")
        payload: dict = {"question_id": question["id"]}
        ready = True
    else:
        text = st.text_area("Soru metni", height=140)
        left, right = st.columns(2)
        gold = left.text_input("Beklenen cevap (isteğe bağlı; girilirse doğrulanır)")
        lang = right.selectbox("Dil", ["tr", "en"])
        payload = {"text": text, "gold": gold or None, "lang": lang}
        ready = bool(text.strip())

    left, right = st.columns(2)
    model_key = left.selectbox("Model", list(available), format_func=available.get)
    strategy = right.radio("Strateji", list(STRATEGIES), format_func=STRATEGIES.get, horizontal=True)
    if st.button("Çöz", type="primary", disabled=not ready):
        with st.spinner("Model çözüyor; yerel modelde ilk çağrı birkaç dakika sürebilir..."):
            st.session_state.result = call("POST", "/solve",
                                           json={**payload, "model_key": model_key, "strategy": strategy})
            st.session_state.step = 1
    if st.session_state.get("result"):
        show_result(st.session_state.result)


def show_result(result: dict) -> None:
    st.divider()
    if result.get("error"):
        st.error(result["error"])
        return
    verdict = result.get("verdict")
    icon, label = STATUS.get(verdict["status"], ("•", "?")) if verdict else ("•", "Beklenen cevap yok")
    cols = st.columns(4)
    cols[0].metric("Sonuç", f"{icon} {label}")
    cols[1].metric("Model cevabı", result.get("answer") or "—")
    cols[2].metric("Beklenen", result.get("gold") or "—")
    cols[3].metric("Süre", f"{result['latency_ms'] / 1000:.1f} sn" if result.get("latency_ms") else "—")
    if verdict:
        extra = " · sayı biçimi belirsizdi (Türkçe/İngilizce yazım)" if verdict.get("ambiguous_format") else ""
        st.caption(f"Doğrulama katmanı: {verdict['layer']} · {verdict['detail']}{extra}")

    solution = result.get("solution") or ""
    if result.get("code"):
        st.subheader("Modelin yazdığı kod")
        st.code(result["code"], language="python")
        st.text("Program çıktısı:\n" + (result.get("exec_stdout") or "(boş)"))
        if result.get("exec_error"):
            st.warning(result["exec_error"])
    else:
        st.subheader("Çözüm")
        steps = solution_steps(solution)
        if st.toggle("İpucu modu: çözümü adım adım aç") and steps:
            shown = min(st.session_state.get("step", 1), len(steps))
            for part in steps[:shown]:
                st.markdown(as_markdown(part))
            if shown < len(steps) and st.button(f"Sonraki adım ({shown}/{len(steps)})"):
                st.session_state.step = shown + 1
                st.rerun()
        else:
            st.markdown(as_markdown(solution))
    with st.expander("Ham model çıktısı"):
        st.text(solution)
    if result.get("reasoning"):
        with st.expander("Modelin düşünme metni"):
            st.text(result["reasoning"])


# --- Sonuçlar --------------------------------------------------------------------------------

def accuracy_chart(overall: pd.DataFrame) -> alt.Chart:
    names = list(LABELS.values())
    data = overall.assign(
        strateji=overall["strategy"].map(lambda k: LABELS.get(k, k)),
        dogruluk=(overall["accuracy"] * 100).round(1),
        etiket=(overall["accuracy"] * 100).round(0).astype(int).astype(str) + "% (" +
        overall["correct"].astype(str) + "/" + overall["n"].astype(str) + ")",
    )
    base = alt.Chart(data).encode(
        y=alt.Y("model:N", title=None),
        yOffset=alt.YOffset("strateji:N", sort=names, scale=alt.Scale(paddingInner=0.15)),
        x=alt.X("dogruluk:Q", title="Doğruluk (%)", scale=alt.Scale(domain=[0, 100])),
        tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("strateji:N", title="Strateji"),
                 alt.Tooltip("dogruluk:Q", title="Doğruluk (%)"), alt.Tooltip("correct:Q", title="Doğru"),
                 alt.Tooltip("n:Q", title="Soru"), alt.Tooltip("unparsed:Q", title="Doğrulanamayan")],
    )
    bars = base.mark_bar(cornerRadiusEnd=4).encode(
        color=alt.Color("strateji:N", scale=alt.Scale(domain=names, range=list(SERIES[theme_mode()])),
                        legend=alt.Legend(title=None, orient="top")))
    labels = base.mark_text(align="left", dx=6).encode(text="etiket:N")
    return (bars + labels).properties(height=70 * overall["model"].nunique() + 40)


def show_cascade(run_id: int, overall: pd.DataFrame) -> None:
    """AS2: accept the cheap model's answer when its CoT and PoT agree, otherwise escalate."""
    complete = overall.groupby("model")["strategy"].nunique()
    cheap_options = sorted(complete[complete == 2].index)          # needs both CoT and PoT
    totals = overall.groupby("model")[["correct", "n"]].sum()
    by_accuracy = totals.apply(lambda r: wilson_low(r["correct"], r["n"]), axis=1).sort_values()
    if not cheap_options or len(by_accuracy) < 2:
        return
    st.subheader("Kademeli çözüm: CoT ve PoT uyuşmazsa güçlü modele geç")
    left, right = st.columns(2)
    weakest = next(m for m in by_accuracy.index if m in cheap_options)
    cheap = left.selectbox("Ucuz model", cheap_options, index=cheap_options.index(weakest))
    strong_options = [m for m in reversed(by_accuracy.index) if m != cheap]
    strong = right.selectbox("Güçlü model", strong_options)
    result = call("GET", f"/runs/{run_id}/cascade", params={"cheap": cheap, "strong": strong})
    if not result["n"]:
        st.info("Bu iki modelin ortak, eksiksiz cevabı olan soru yok.")
        return
    table = pd.DataFrame(result["rows"])
    table["doğruluk"] = (table["doğruluk"] * 100).round(1)
    table["güven_aralığı"] = table["güven_aralığı"].map(lambda ci: f"%{100 * ci[0]:.0f}–%{100 * ci[1]:.0f}")
    st.dataframe(table.rename(columns={"yöntem": "Yöntem", "n": "Soru", "doğru": "Doğru", "doğruluk": "Doğruluk (%)",
                                       "güven_aralığı": "%95 GA", "maliyet_usd": "Maliyet (USD, liste)",
                                       "ort_süre_sn": "Ort. süre (sn)", "güçlü_model_çağrısı": "Güçlü model çağrısı"}),
                 hide_index=True)
    agreement, test = result["agreement"], result["mcnemar"]["güçlü_modele_karşı"]
    st.caption(f"{agreement['kabul_edilen']} soruda iki strateji uyuştu ve cevap kabul edildi "
               f"(doğruluk %{100 * agreement['kabul_doğruluğu']:.0f}); {agreement['yükseltilen']} soru güçlü modele "
               f"gönderildi. Güçlü modele karşı McNemar p = {test['p']:.3f}."
               + (f" Eksik cevap yüzünden {len(result['missing'])} soru dışarıda kaldı." if result["missing"] else ""))
    with st.expander("Soru bazında kararlar"):
        st.dataframe(pd.DataFrame(result["decisions"]), hide_index=True)


def page_results() -> None:
    st.header("Deney sonuçları")
    runs = [r for r in call("GET", "/runs") if r["responses"]]
    runs.sort(key=lambda r: (r["name"] == INTERACTIVE_RUN, -r["id"]))   # batch experiments first
    if not runs:
        st.info("Henüz sonuç yok. Toplu deney için: `python -m llmath.runner --name pilot-1`")
        return
    run = st.selectbox("Çalıştırma", runs, format_func=lambda r: f"#{r['id']} · {r['name']} · {r['responses']} yanıt")
    summary = pd.DataFrame(call("GET", f"/runs/{run['id']}/summary"))
    if summary.empty:
        st.info("Bu çalıştırmada soru setine bağlı, doğrulanmış yanıt yok.")
        return

    overall = summary.groupby(["model", "strategy"], as_index=False)[["n", "correct", "unparsed"]].sum()
    overall["accuracy"] = overall["correct"] / overall["n"]
    best = overall.loc[overall.apply(lambda r: wilson_low(r["correct"], r["n"]), axis=1).idxmax()]
    st.subheader(f"En yüksek doğruluk: {best['model']}, {LABELS.get(best['strategy'], best['strategy'])} — "
                 f"%{100 * best['accuracy']:.0f} ({best['correct']}/{best['n']})")
    st.caption("Sıralama, az soruyla elde edilen yüksek oranları öne çıkarmamak için %95 güven aralığının "
               "alt sınırına göre yapılır.")
    st.altair_chart(accuracy_chart(overall), width="stretch")
    st.dataframe(overall.assign(accuracy=(overall["accuracy"] * 100).round(1))
                 .rename(columns={"model": "Model", "strategy": "Strateji", "n": "Soru", "correct": "Doğru",
                                  "unparsed": "Doğrulanamayan", "accuracy": "Doğruluk (%)"}),
                 hide_index=True)

    st.subheader("Alt küme ve dile göre")
    st.dataframe(summary.assign(accuracy=(summary["accuracy"] * 100).round(1)), hide_index=True)

    bilingual = [subset for subset, langs in summary.groupby("subset")["lang"].agg(set).items()
                 if {"tr", "en"} <= langs]
    if bilingual:
        st.subheader("Aynı sorular Türkçe ve İngilizce")
        paired = summary[summary["subset"].isin(bilingual)]
        table = paired.pivot_table(index=["model", "strategy", "subset"], columns="lang",
                                   values=["accuracy", "n"]).reset_index()
        table.columns = [f"{a}_{b}" if b else a for a, b in table.columns]
        table["fark (puan)"] = ((table["accuracy_tr"] - table["accuracy_en"]) * 100).round(1)
        table[["accuracy_tr", "accuracy_en"]] = (table[["accuracy_tr", "accuracy_en"]] * 100).round(1)
        st.dataframe(table.rename(columns={"accuracy_tr": "Türkçe (%)", "accuracy_en": "İngilizce (%)",
                                           "n_tr": "Türkçe soru", "n_en": "İngilizce soru"}), hide_index=True)
        st.caption("Eşleştirilmiş karşılaştırma ve McNemar testi `scripts/report.py` raporundadır; "
                   "30'dan az soruda farklar anlamlı sayılmamalı.")

    show_cascade(run["id"], overall)

    st.subheader("Yanlış ve doğrulanamayan cevaplar")
    responses = pd.DataFrame(call("GET", f"/runs/{run['id']}/responses"))
    wrong = responses[responses["status"] != "correct"]
    if wrong.empty:
        st.success("Bu çalıştırmada yanlış cevap yok.")
    else:
        st.dataframe(wrong[["question_id", "model", "strategy", "gold", "answer", "status", "layer", "detail"]],
                     hide_index=True)
        if len(wrong) > 50:
            st.caption(f"Ayrıntısı ilk 50 cevabın gösteriliyor; {len(wrong)} cevabın tamamı tabloda ve CSV'de.")
        for _, row in wrong.head(50).iterrows():
            with st.expander(f"{row['question_id']} · {row['model']} · {row['strategy']} · {row['status']}"):
                st.markdown(as_markdown(row["question"] or ""))
                st.markdown(f"**Beklenen:** {row['gold']} · **Model:** {row['answer']} · {row['detail']}")
                if row.get("code"):
                    st.code(row["code"], language="python")
                if row.get("exec_error"):
                    st.warning(row["exec_error"])
                st.text(row["solution"] or "")
    st.download_button("Tüm yanıtları CSV olarak indir", responses.to_csv(index=False).encode("utf-8"),
                       file_name=f"llmath_calistirma_{run['id']}.csv", mime="text/csv")


# --- Hata etiketleme -------------------------------------------------------------------------

def page_annotate() -> None:
    st.header("Hata etiketleme")
    taxonomy = call("GET", "/taxonomy")
    types = {t["code"]: t for t in taxonomy["types"]}
    stages = {s["key"]: s for s in taxonomy["stages"]}
    with st.expander("Kod kitabı"):
        st.dataframe(pd.DataFrame([{"Kod": t["code"], "Tür": t["name"],
                                    "Olağan aşama": stages[t["stage"]]["name"] if t["stage"] else "—",
                                    "Tanım": t["definition"], "Örnek": t["example"]} for t in types.values()]),
                     hide_index=True)
        for s in stages.values():
            st.markdown(f"**{s['name']}:** {s['definition']}")
    st.caption("Her yanlış cevapta ilk hatalı adımı bulun, hatanın türünü ve doğduğu Newman aşamasını seçin. "
               "Cevaplar sabit bir rastgele sırayla gelir; başka etiketleyicilerin ve LLM-hakemin etiketleri "
               "gösterilmez.")

    runs = [r for r in call("GET", "/runs") if r["responses"] and r["name"] != INTERACTIVE_RUN]
    if not runs:
        st.info("Etiketlenecek toplu deney yok.")
        return
    left, middle, right = st.columns([2, 1, 1])
    names = [r["name"] for r in runs]
    run = left.selectbox("Çalıştırma", runs, index=names.index(ANNOTATION_RUN) if ANNOTATION_RUN in names else 0,
                         format_func=lambda r: f"#{r['id']} · {r['name']}",
                         help=f"İnsan etiketleri {ANNOTATION_RUN} için toplanıyor; LLM-hakem de bu çalıştırmayı etiketliyor.")
    annotator = middle.text_input("Etiketleyen", value="human",
                                  help="İkinci bir etiketleyici kendi adını yazar; uyum etiketleyenler "
                                       "arasında ölçülür.")
    summary = call("GET", f"/runs/{run['id']}/summary")
    models = sorted({row["model"] for row in summary})
    model = right.selectbox("Model", ["(hepsi)"] + models)
    sub_col, strategy_col = st.columns(2)
    subsets = sorted({row["subset"] for row in summary})
    options = ["(hepsi)"] + ([ANNOTATION_SCOPE] if {"P1", "P3"} <= set(subsets) else []) + subsets
    subset = sub_col.selectbox("Alt küme", options, index=options.index(ANNOTATION_SCOPE) if ANNOTATION_SCOPE in options
                               else 0, format_func=lambda s: SUBSET_LABELS.get(s, s),
                               help="P2 (MATH) sorularını yalnız eminseniz etiketleyin; emin değilseniz 'Atla'.")
    strategy = strategy_col.selectbox("Strateji", ["(hepsi)", "cot", "pot"], format_func=lambda s: LABELS.get(s, s))
    skipped: list[int] = st.session_state.setdefault("skipped", [])
    params = {"annotator": annotator, "skip": ",".join(map(str, skipped))}
    for key, value in (("model", model), ("subset", subset), ("strategy", strategy)):
        if value != "(hepsi)":
            params[key] = value
    data = call("GET", f"/runs/{run['id']}/annotation/next", params=params)
    st.progress(min(data["done_in_run"] / ANNOTATION_TARGET, 1.0),
                text=f"Bu çalıştırmada {data['done_in_run']} cevap etiketlediniz (hedef en az {ANNOTATION_TARGET}) "
                     f"· bu seçimde {data['remaining']} cevap kaldı")
    item = data["item"]
    if item is None:
        st.success("Bu seçimde etiketlenecek yanlış cevap kalmadı.")
        if skipped and st.button(f"Atlanan {len(skipped)} cevabı geri getir"):
            skipped.clear()
            st.rerun()
        return

    rid = item["response_id"]
    st.subheader(f"{item['question_id']} · {item['model']} · {LABELS.get(item['strategy'], item['strategy'])}")
    st.markdown(as_markdown(item["question"]))
    cols = st.columns(3)
    cols[0].metric("Beklenen", item["gold"])
    cols[1].metric("Model cevabı", item["answer"] or "—")
    cols[2].metric("Doğrulayıcı", STATUS[item["status"]][1], item["layer"], delta_color="off")
    if is_truncated(item["finish_reason"], item["exec_error"]):
        st.warning("Çıktı yarıda kalmış (token sınırı ya da zaman aşımı). Bu cevap ayrıca H13 sayılır. Tür olarak "
                   "görünen ilk hatayı seçin; görünür bir hata yoksa H13 seçin.")
    if item["code"]:
        st.code(item["code"], language="python", line_numbers=True)
        st.text("Program çıktısı:\n" + (item["exec_stdout"] or "(boş)"))
        if item["exec_error"]:
            st.warning(item["exec_error"])
        step_count, step_label = item["code"].count("\n") + 1, "İlk hatalı kod satırı"
    else:
        steps = solution_steps(item["solution"] or "")
        for number, part in enumerate(steps, start=1):
            st.markdown(f"**{number}.** " + as_markdown(part))
        step_count, step_label = len(steps), "İlk hatalı adım"

    st.divider()
    code = st.selectbox("Hata türü", list(types), index=None, placeholder="Hata türünü seçin", key=f"type-{rid}",
                        format_func=lambda c: f"{c} · {types[c]['name']}")
    stage = None
    if code:
        st.caption(f"{types[code]['definition']} Örnek: {types[code]['example']}")
        if types[code]["stage"]:
            keys = list(stages)
            stage = st.radio("Newman aşaması (hatanın doğduğu yer)", keys, index=keys.index(types[code]["stage"]),
                             format_func=lambda k: stages[k]["name"], horizontal=True, key=f"stage-{rid}-{code}")
    step = st.number_input(f"{step_label} (0 = belirlenemedi)", min_value=0, max_value=max(step_count, 1),
                           value=0, key=f"step-{rid}")
    note = st.text_input("Not (isteğe bağlı)", key=f"note-{rid}")
    left, right = st.columns([1, 6])
    if left.button("Kaydet", type="primary", disabled=code is None):
        call("POST", "/annotations", json={"response_id": rid, "annotator": annotator, "error_type": code,
                                           "newman_stage": stage, "first_error_step": step or None,
                                           "note": note or None})
        st.rerun()
    if right.button("Atla"):
        skipped.append(rid)
        st.rerun()


page = st.sidebar.radio("Sayfa", ["Soru çöz", "Deney sonuçları", "Hata etiketleme"])
st.sidebar.caption(f"API: {API_URL}")
if page == "Soru çöz":
    page_solve()
elif page == "Deney sonuçları":
    page_results()
else:
    page_annotate()
