"""Витрина поверх API: клиент пишет жалобу — тренер видит ограничения.

Сам продукт — это API (src/serve.py), витрина только для демострации, позже допилить!!!

  streamlit run demo.py    # API на localhost:8000
"""
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Тренировочные ограничения по жалобе", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1180px; }
      h1 { color: #0f766e; font-size: 1.9rem; margin-bottom: .2rem; }
      .subtitle { color: #5b8f8a; margin-bottom: 1.6rem; }
      .panel {
        border: 1px solid #cbe9e4; border-radius: 12px;
        padding: 1.1rem 1.2rem; background: #fbfffe; min-height: 430px;
      }
      .panel-head {
        font-weight: 600; color: #0f766e; letter-spacing: .04em;
        text-transform: uppercase; font-size: .78rem;
        border-bottom: 2px solid #99f6e4; padding-bottom: .5rem; margin-bottom: 1rem;
      }
      .group {
        background: linear-gradient(90deg, #0f766e, #0891b2);
        color: #fff; padding: .7rem 1rem; border-radius: 8px;
        font-size: 1.25rem; font-weight: 600; margin-bottom: .8rem;
      }
      .summary {
        background: #f0fdfa; border-left: 4px solid #14b8a6;
        padding: .7rem .9rem; border-radius: 6px; color: #134e4a; margin-bottom: 1rem;
      }
      .sect { font-weight: 600; color: #0f766e; margin: .9rem 0 .35rem; font-size: .95rem; }
      .card { border-left: 4px solid; border-radius: 6px; padding: .6rem .9rem; margin-bottom: .5rem; }
      .card ul { margin: 0; padding-left: 1.1rem; }
      .card li { margin: .18rem 0; color: #134e4a; }
      .c-forbid { border-color: #0f766e; background: #e6fffb; }
      .c-caution { border-color: #0891b2; background: #eff9fd; }
      .c-equip  { border-color: #5eead4; background: #f2fdfb; }
      .waiting { color: #8bb5b0; text-align: center; padding-top: 5.5rem; font-size: .95rem; }
      .sent { background: #e6fffb; border: 1px solid #99f6e4; border-radius: 8px;
              padding: .8rem 1rem; color: #0f766e; }
      .refine { background: #fff8e6; border: 1px solid #e8cf95; border-radius: 8px;
                padding: .8rem 1rem; color: #7a5b12; }
      .note { color: #7aa5a0; font-size: .8rem; margin-top: 1.1rem; }
      .unsure { background: #fff8e6; border-left: 4px solid #d9a441;
                padding: .8rem 1rem; border-radius: 6px; color: #7a5b12; margin-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("# Тренировочные ограничения по жалобе")
st.markdown(
    '<div class="subtitle">Клиент описывает состояние своими словами — тренер получает '
    "группу ограничений и рекомендации по нагрузке.</div>",
    unsafe_allow_html=True,
)

EXAMPLES = [
    "болит поясница, отдаёт в ногу при наклоне",
    "изжога и тяжесть в животе после еды",
    "кашель, одышка, тяжело дышать",
    "артроз коленного сустава, боли при нагрузке",
]#для демострации, позже убрать!!


MAX_ATTEMPTS = 3


def call_api(text: str) -> dict:
    resp = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=90)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(show_spinner=False)
def fetch_groups() -> list[dict]:
    resp = requests.get(f"{API_URL}/groups", timeout=30)
    resp.raise_for_status()
    return resp.json()


def bullets(items: list[str], css: str) -> str:
    lis = "".join(f"<li>{i}</li>" for i in items)
    return f'<div class="card {css}"><ul>{lis}</ul></div>'


with st.sidebar:
    st.markdown("**Примеры жалоб**")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state["text"] = ex
            st.session_state.pop("result", None)
    try:
        requests.get(f"{API_URL}/health", timeout=3).raise_for_status()
    except Exception:
        st.divider()
        st.error("API недоступен")

client_col, trainer_col = st.columns([1, 1.35], gap="large")

with client_col:
    st.markdown('<div class="panel-head">Экран клиента</div>', unsafe_allow_html=True)
    text = st.text_area(
        "Опишите, что беспокоит",
        key="text",
        height=130,
        placeholder="например: болит поясница, отдаёт в ногу при наклоне",
    )
    send = st.button("Отправить тренеру", type="primary", use_container_width=True)
    none = st.button("Жалоб нет", use_container_width=True)

    if send:
        if not text.strip():
            st.warning("Опишите жалобу или нажмите «Жалоб нет».")
        else:
            try:
                with st.spinner("Отправляю..."):
                    res = call_api(text)
                st.session_state["result"] = res
                if res["needs_clarification"]:
                    st.session_state["attempts"] = st.session_state.get("attempts", 0) + 1
                else:
                    st.session_state["attempts"] = 0
            except Exception as exc:
                st.error(f"Сервис недоступен: {exc}")
    # жалоб нет — модель не запускаем
    if none:
        st.session_state["result"] = {"none": True}
        st.session_state["attempts"] = 0

    result = st.session_state.get("result")
    attempts = st.session_state.get("attempts", 0)
    # после MAX_ATTEMPTS перестаём мучить клиента и отдаём решение тренеру
    if result and result.get("needs_clarification") and attempts < MAX_ATTEMPTS:
        st.markdown(
            '<div class="refine">Опишите жалобу точнее: где именно болит, при каких '
            "движениях, как давно. Так тренер сможет подобрать безопасную нагрузку.</div>",
            unsafe_allow_html=True,
        )
    elif result:
        st.markdown(
            '<div class="sent">Анкета передана тренеру. Он учтёт это при составлении '
            "программы.</div>",
            unsafe_allow_html=True,
        )

with trainer_col:
    st.markdown('<div class="panel-head">Экран тренера</div>', unsafe_allow_html=True)
    data = st.session_state.get("result")

    if not data:
        st.markdown(
            '<div class="waiting">Ожидание анкеты от клиента</div>', unsafe_allow_html=True
        )
    elif data.get("none"):
        st.markdown('<div class="group">Ограничений нет</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="summary">Клиент жалоб не предъявляет. Тренировки без специальных '
            "ограничений: разминка, контроль техники, постепенный рост нагрузки.</div>",
            unsafe_allow_html=True,
        )
    elif data["needs_clarification"] and st.session_state.get("attempts", 0) >= MAX_ATTEMPTS:
        st.markdown(
            f'<div class="unsure"><b>Модель не определилась за {MAX_ATTEMPTS} уточнения.</b> '
            "Ниже — распределение вероятностей по всем группам. Выберите группу сами.</div>",
            unsafe_allow_html=True,
        )
        for alt in data["distribution"]:
            st.progress(alt["confidence"], text=f'{alt["group"]} — {alt["confidence"]:.0%}')

        groups = [g["group"] for g in fetch_groups()]
        choice = st.selectbox("Группа ограничений", groups, key="manual_group")
        rec = next(g for g in fetch_groups() if g["group"] == choice)

        st.markdown(f'<div class="group">{choice}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary">{rec["summary"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="sect">Исключить</div>', unsafe_allow_html=True)
        st.markdown(bullets(rec["forbidden"], "c-forbid"), unsafe_allow_html=True)
        st.markdown('<div class="sect">С осторожностью</div>', unsafe_allow_html=True)
        st.markdown(bullets(rec["caution"], "c-caution"), unsafe_allow_html=True)
        st.markdown('<div class="sect">Тренажёры и снаряды</div>', unsafe_allow_html=True)
        st.markdown(bullets(rec["equipment"], "c-equip"), unsafe_allow_html=True)
        st.caption("Выбор тренера — будущая разметка для дообучения модели.")
    elif data["needs_clarification"]:
        st.markdown(
            f'<div class="unsure"><b>Модель не уверена ({data["confidence"]:.0%}).</b> '
            f'{data["message"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sect">Возможные группы</div>', unsafe_allow_html=True)
        for alt in data["alternatives"]:
            st.progress(alt["confidence"], text=f'{alt["group"]} — {alt["confidence"]:.0%}')
    else:
        st.markdown(f'<div class="group">{data["group"]}</div>', unsafe_allow_html=True)
        st.progress(data["confidence"], text=f'Уверенность модели — {data["confidence"]:.0%}')
        st.markdown(f'<div class="summary">{data["summary"]}</div>', unsafe_allow_html=True)

        st.markdown('<div class="sect">Исключить</div>', unsafe_allow_html=True)
        st.markdown(bullets(data["forbidden"], "c-forbid"), unsafe_allow_html=True)

        st.markdown('<div class="sect">С осторожностью</div>', unsafe_allow_html=True)
        st.markdown(bullets(data["caution"], "c-caution"), unsafe_allow_html=True)

        st.markdown('<div class="sect">Тренажёры и снаряды</div>', unsafe_allow_html=True)
        st.markdown(bullets(data["equipment"], "c-equip"), unsafe_allow_html=True)

    if data and not data.get("none"):
        st.markdown(
            f'<div class="note">{data["disclaimer"]}</div>', unsafe_allow_html=True
        )
