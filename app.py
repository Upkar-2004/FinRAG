"""
FinRAG demo UI. Ask a question about one of the 5 FY2022 10-K filings (AMD,
American Express, Boeing, PepsiCo, 3M) and get a grounded, cited answer --
built on the exact retrieve_with_expansion() + generate_answer() pipeline
documented in docs/finrag_learning_report.md.

Layout: FinRAG is the hero -- a large wordmark at the top of the left
brand/research rail (ui.render_sidebar()), not a separate headline in the
main workspace. The main workspace is just a composer, with a persistent
photographic atmosphere behind it (ui.render_atmosphere()) that stays
visible whether or not a question has been asked yet. Below the composer:
nothing extra before the first question (the atmosphere fills that space
on its own), and the answer once one exists -- splitting into
[answer document | evidence panel] only then, decided AFTER the
form-submission branch runs so a first-ever submission gets the two-column
layout in the SAME run it's answered, not one run later.

Styling is native-first (.streamlit/config.toml) with one deliberate,
scoped exception -- ui.py's THEME_CSS -- for what config.toml has no slot
for (layered background photography, translucent surfaces, citation
badges, evidence cards, sidebar rows, focus ring, motion, breakpoints).
See ui.py's module docstring.

Run:
    streamlit run app.py
"""

import streamlit as st
from groq import BadRequestError, RateLimitError

import config
import ui
from src.finrag import telemetry
from src.finrag.generate import generate_answer
from src.finrag.retrieve import retrieve_with_expansion

st.set_page_config(page_title="FinRAG", page_icon=":material/candlestick_chart:", layout="wide")
ui.inject_theme_css()
ui.render_atmosphere(
    ui.background_data_uri("assets/market-board.jpeg"),
    ui.background_data_uri("assets/currency-texture.jpeg"),
)

st.session_state.setdefault("question", "")
st.session_state.setdefault("last_result", None)
st.session_state.setdefault("last_chunks", None)
st.session_state.setdefault("last_question", None)
st.session_state.setdefault("last_scope", [])


def _set_question(q: str) -> None:
    st.session_state.question = q


EXAMPLE_QUESTIONS = [
    "Has Boeing reported any materially important ongoing legal battles from FY2022?",
    "What are the major products and services that AMD sells as of FY22?",
    "Does AMD have a reasonably healthy liquidity profile based on its quick ratio for FY22?",
    "What are the geographies that American Express primarily operates in as of 2022?",
]

doc_names = ui.render_sidebar(EXAMPLE_QUESTIONS, _set_question)
scope_labels = [ui.short_company(config.CORPUS_DOCS[d]) for d in doc_names]

ui.render_mobile_brand()
ui.render_status_badge()

has_text = bool(st.session_state.question.strip())
with st.form("composer", border=False):
    query_col, submit_col = st.columns([5, 1], vertical_alignment="bottom")
    with query_col:
        question = st.text_input(
            "Ask a question",
            key="question",
            label_visibility="collapsed",
            placeholder="Ask about revenue, margins, liquidity, risk factors...",
            icon=":material/search:",
        )
    with submit_col:
        submitted = st.form_submit_button(
            "Ask", type="primary", icon=":material/arrow_forward:", width="stretch", disabled=not has_text
        )

display_result = st.session_state.last_result
display_chunks = st.session_state.last_chunks
display_question = st.session_state.last_question
display_scope = st.session_state.last_scope

if submitted:
    if question and question.strip():
        with st.spinner("Retrieving relevant filing text...", show_time=True):
            with telemetry.timed("retrieval", question=question, doc_names=doc_names) as rec:
                chunks = retrieve_with_expansion(question, k=config.TOP_K, doc_names=doc_names)
                rec["num_results"] = len(chunks)
        with st.spinner("Generating grounded answer...", show_time=True):
            try:
                result = generate_answer(question, chunks)
            except RateLimitError:
                st.error("API quota is currently exhausted. Please try again shortly.")
                st.stop()
            except BadRequestError as e:
                st.error(f"The model's tool call failed after retries: {e}")
                st.stop()
        st.session_state.last_result = result
        st.session_state.last_chunks = chunks
        st.session_state.last_question = question
        st.session_state.last_scope = scope_labels
        display_result, display_chunks, display_question, display_scope = result, chunks, question, scope_labels
    else:
        st.warning("Enter a question first.", icon=":material/info:")

has_answer = display_result is not None
show_evidence = has_answer and bool(display_chunks)

if show_evidence:
    # Ratio tuned so the evidence column lands near the spec'd 300-360px at a
    # ~1440px viewport with the sidebar open (Streamlit columns are proportional,
    # not fixed-px, so this is an approximation, not a hard clamp).
    workspace_col, evidence_col = st.columns([2.3, 1], gap="large")
else:
    workspace_col, evidence_col = st.container(), None

with workspace_col:
    if has_answer:
        ui.render_answer_document(display_question, display_result, display_chunks, display_scope)
    # else: nothing -- the atmosphere fills the space; no separate empty-state headline.

if evidence_col is not None:
    with evidence_col:
        ui.render_evidence_panel(display_chunks)
