"""
UI rendering + styling helpers for app.py.

Streamlit has no component tree the way a JS framework does -- every
function below just calls st.* to draw its section into the current script
run, then returns; there's no isolated, re-usable component instance. They're
named after the roles they play (sidebar/brand rail, atmosphere, composer,
answer document, evidence panel, citation badge...) so the mapping to a
conventional component breakdown is obvious, even though the implementation
is "a Python function," not a component class.

Colors/fonts/radius/borders still lives natively in .streamlit/config.toml
where a slot exists. THEME_CSS below is the deliberate, explicitly-requested
exception for what config.toml has no slot for: the layered photography
background, translucent/blurred surfaces, citation badges, evidence cards,
compact sidebar rows, the focus ring, motion, and responsive breakpoints.
"""

import base64
import html
import json
import re
from pathlib import Path

import streamlit as st

import config

THEME_CSS = """
:root {
  --bg: #060708;
  --bg-secondary: #0B0E11;
  --surface: rgba(15, 19, 23, 0.88);
  --surface-elevated: rgba(20, 25, 30, 0.92);
  --surface-hover: rgba(26, 32, 38, 0.92);
  --border: rgba(223, 208, 179, 0.12);
  --border-strong: rgba(208, 174, 114, 0.32);
  --text-primary: #EEE9DE;
  --text-secondary: #A8A49C;
  --text-muted: #787B7E;
  --brass: #B9975B;
  --brass-hi: #D0AE72;
  --positive: #55B88A;
  --negative: #D36B6B;
  --focus: #D0AE72;
  --mono: 'IBM Plex Mono', ui-monospace, Menlo, monospace;
  --serif: 'Fraunces', Georgia, serif;
  --motion: 180ms ease;
}

/* ---- reduced motion / focus ---------------------------------------- */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition: none !important; animation: none !important; }
}
*:focus-visible { outline: 2px solid var(--focus) !important; outline-offset: 2px; }
[data-testid="stSidebar"] button, .finrag-row, .finrag-evidence-card,
.finrag-citation, .finrag-copy-btn, .finrag-mobile-brand {
  transition: background-color var(--motion), border-color var(--motion), color var(--motion);
}
body, [data-testid="stAppViewContainer"] { overflow-x: hidden; }

/* ---- atmosphere: layered photography behind the main workspace only ----
   (never behind the sidebar -- spec wants the research rail solid/opaque).
   Two independent-opacity images can't share one element's `opacity`, so
   each gets its own absolutely-positioned pseudo-element; the warm
   brass/black wash is the real element's OWN background (paints behind
   both position:absolute pseudo-elements by default stacking rules), not a
   third pseudo-element CSS can't give us. Image URLs are injected per-run
   via a small scoped <style> block -- see render_atmosphere(). */
[data-testid="stMain"] {
  position: relative;
  overflow: hidden;
  background:
    radial-gradient(ellipse 900px 700px at 90% 6%, rgba(185, 151, 91, 0.08), transparent 60%),
    linear-gradient(165deg, #0B0E11 0%, #060708 55%);
}
[data-testid="stMain"] [data-testid="stMainBlockContainer"] { position: relative; z-index: 2; }
[data-testid="stMain"]::before {
  content: "";
  position: absolute;
  top: 0; right: 0;
  width: 52%;
  height: 100%;
  background-size: cover;
  background-position: center right;
  opacity: 0.38;
  z-index: 0;
  pointer-events: none;
  -webkit-mask-image: linear-gradient(to bottom left, black 0%, black 45%, transparent 88%);
  mask-image: linear-gradient(to bottom left, black 0%, black 45%, transparent 88%);
}
[data-testid="stMain"]::after {
  content: "";
  position: absolute;
  bottom: 0; left: 0;
  width: 40%;
  height: 46%;
  background-size: cover;
  background-position: center;
  opacity: 0.16;
  z-index: 1;
  pointer-events: none;
  -webkit-mask-image: radial-gradient(ellipse at bottom left, black 0%, transparent 72%);
  mask-image: radial-gradient(ellipse at bottom left, black 0%, transparent 72%);
}
@media (max-width: 1024px) {
  [data-testid="stMain"]::before { width: 62%; opacity: 0.30; }
  [data-testid="stMain"]::after { width: 50%; opacity: 0.13; }
}
@media (max-width: 640px) {
  [data-testid="stMain"]::before {
    width: 78%; height: 46%; top: 0; right: 0; bottom: auto;
    opacity: 0.24;
    -webkit-mask-image: linear-gradient(to bottom left, black 0%, black 30%, transparent 80%);
    mask-image: linear-gradient(to bottom left, black 0%, black 30%, transparent 80%);
  }
  [data-testid="stMain"]::after { opacity: 0.10; }
}

/* ---- sidebar / brand rail ------------------------------------------- */
[data-testid="stSidebar"] { min-width: 340px !important; max-width: 390px !important; }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: 28px; }
.finrag-wordmark {
  font-family: var(--serif);
  font-size: 72px;
  line-height: 1.0;
  font-weight: 500;
  letter-spacing: -0.015em;
  margin: 0 0 10px 0;
  color: var(--text-primary);
}
.finrag-wordmark .brass { color: var(--brass-hi); }
.finrag-tagline {
  font-size: 13.5px;
  color: var(--text-secondary);
  line-height: 1.5;
  margin-bottom: 28px;
  max-width: 30ch;
}
.finrag-sidebar-label {
  font-family: var(--mono);
  font-size: 10.5px;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin: 20px 0 8px 0;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
  justify-content: flex-start;
  background: transparent;
  border: none !important;
  border-bottom: 1px solid var(--border) !important;
  border-radius: 0;
  padding: 11px 4px;
  min-height: 0;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover { background: var(--surface-hover); }
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] p {
  text-align: left;
  white-space: normal;
  font-size: 13px;
  line-height: 1.4;
  color: var(--text-secondary);
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover p { color: var(--text-primary); }
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] p::after {
  content: " \\2192";
  color: var(--brass-hi);
  opacity: 0;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover p::after { opacity: 1; }

/* small brand mark shown ONLY when the sidebar is collapsed on narrow
   viewports, so "FinRAG" still reads at the top of the page -- see
   render_mobile_brand(). Hidden on desktop; the sidebar wordmark covers it. */
.finrag-mobile-brand { display: none; }
@media (max-width: 640px) {
  .finrag-mobile-brand {
    display: block;
    font-family: var(--serif);
    font-size: 48px;
    font-weight: 500;
    color: var(--text-primary);
    margin: 4px 0 4px 0;
  }
  .finrag-mobile-brand .brass { color: var(--brass-hi); }
  .finrag-wordmark { font-size: 56px; }
}

/* ---- status badge (near composer) ------------------------------------ */
.finrag-status-row {
  display: flex;
  align-items: center;
  gap: 8px;
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.06em;
  color: var(--text-muted);
  margin: 2px 0 14px 2px;
}
.finrag-status-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--positive);
  box-shadow: 0 0 0 3px rgba(85, 184, 138, 0.15);
}

/* ---- composer --------------------------------------------------------- */
[data-testid="stForm"] {
  background: var(--surface);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border-radius: 14px;
  padding: 18px 18px 14px 18px;
}
[data-testid="stForm"]:focus-within { border: 1px solid var(--border-strong); }
[data-testid="stForm"] [data-testid="stBaseButton-primary"] { font-weight: 600; letter-spacing: 0.01em; }

/* ---- answer / evidence translucent surfaces --------------------------- */
.st-key-answer-surface {
  background: var(--surface);
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 30px 32px 22px 32px;
}
.finrag-eyebrow {
  font-family: var(--mono);
  font-size: 10.5px;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: var(--brass-hi);
  margin-bottom: 8px;
}
.finrag-answer-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  font-family: var(--mono);
  font-size: 12px;
  color: var(--text-secondary);
  font-variant-numeric: tabular-nums;
  padding: 12px 0;
  margin: 8px 0 18px 0;
  border-top: 1px solid var(--border);
  border-bottom: 1px solid var(--border);
}
.finrag-answer-meta b { color: var(--text-primary); font-weight: 500; }
.finrag-answer-body {
  max-width: 74ch;
  font-size: 15.5px;
  line-height: 1.6;
  color: var(--text-primary);
  font-variant-numeric: tabular-nums;
}
.finrag-answer-body p { margin: 0 0 14px 0; }
.finrag-answer-body ol, .finrag-answer-body ul { margin: 0 0 14px 0; padding-left: 22px; }
.finrag-answer-body li { margin-bottom: 7px; }
.finrag-citation {
  display: inline-flex;
  align-items: baseline;
  font-family: var(--mono);
  font-size: 0.79em;
  color: var(--brass-hi);
  background: rgba(185, 151, 91, 0.14);
  border: 1px solid var(--border-strong);
  border-radius: 5px;
  padding: 1px 6px;
  text-decoration: none;
  white-space: nowrap;
}
a.finrag-citation:hover { background: rgba(185, 151, 91, 0.24); color: var(--text-primary); }
.finrag-citation--unlinked { opacity: 0.7; }

.finrag-copy-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-family: Inter, sans-serif;
  font-size: 12.5px;
  color: var(--text-secondary);
  background: transparent;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 6px 11px;
  cursor: pointer;
  min-height: 32px;
}
.finrag-copy-btn:hover { color: var(--text-primary); border-color: var(--border-strong); background: var(--surface-hover); }

.finrag-evidence-card {
  background: var(--surface-elevated);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px 14px;
  margin-bottom: 10px;
  scroll-margin-top: 80px;
}
.finrag-evidence-card:target { border-color: var(--brass); background: var(--surface-hover); }
.finrag-evidence-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 8px;
  font-family: var(--mono);
  font-size: 11.5px;
  color: var(--text-muted);
  margin-bottom: 6px;
  font-variant-numeric: tabular-nums;
}
.finrag-evidence-company { color: var(--text-primary); font-weight: 500; font-family: Inter, sans-serif; font-size: 13px; }
.finrag-evidence-excerpt { font-size: 12.5px; line-height: 1.55; color: var(--text-secondary); }

.sr-only {
  position: absolute; width: 1px; height: 1px;
  padding: 0; margin: -1px; overflow: hidden;
  clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0;
}

/* ---- responsive --------------------------------------------------------- */
@media (max-width: 480px) {
  [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
  [data-testid="stForm"] [data-testid="stBaseButton-primary"] { min-height: 44px; }
  .st-key-answer-surface { padding: 20px 18px; }
}
"""


def inject_theme_css() -> None:
    # unsafe_allow_javascript=True is required for <style> to survive DOMPurify (its
    # default profile excludes the style TAG entirely; ADD_TAGS:['script','style'] is
    # only applied when this is True -- confirmed in Streamlit's Html.js).
    #
    # That alone isn't enough, though: Streamlit's PYTHON-side st.html() (elements/html.py)
    # special-cases body strings that are ENTIRELY a <style> tag -- it routes them to a
    # separate "event" delta generator and, on that path, never sets
    # html_proto.unsafe_allow_javascript at all (the line is skipped, not just defaulted),
    # regardless of what's passed here. A hidden marker span keeps the body from being
    # "only style tags" (the exact check Streamlit runs), forcing the normal code path
    # that actually honors this flag and lands in the main container.
    st.html(f"<style>{THEME_CSS}</style><span style=\"display:none\"></span>", unsafe_allow_javascript=True)


def background_data_uri(path: str) -> str | None:
    """Embed a local image as a base64 data URI, or None if it doesn't
    exist -- lets a background wire up the moment the file is added, no
    code changes needed."""
    p = Path(path)
    if not p.exists():
        return None
    encoded = base64.b64encode(p.read_bytes()).decode()
    return f"data:image/{p.suffix.lstrip('.')};base64,{encoded}"


def short_company(name: str) -> str:
    """'AMD (semiconductors)' -> 'AMD' -- cleaner for UI labels and citations."""
    return name.split(" (")[0]


def render_atmosphere(market_uri: str | None, currency_uri: str | None) -> None:
    """Wires the two background-image URLs into [data-testid="stMain"]'s
    ::before (market board, dominant, right side) and ::after (currency
    texture, secondary, lower-left) pseudo-elements defined in THEME_CSS.
    Call once, early -- persists behind BOTH the empty state and the
    answer state, since the atmosphere should stay visible throughout,
    not just before the first question."""
    rules = []
    if market_uri:
        rules.append(f"[data-testid=\"stMain\"]::before {{ background-image: url('{market_uri}'); }}")
    if currency_uri:
        rules.append(f"[data-testid=\"stMain\"]::after {{ background-image: url('{currency_uri}'); }}")
    if rules:
        # unsafe_allow_javascript=True + the hidden marker span -- see
        # inject_theme_css()'s comment for why BOTH are required.
        st.html(
            f"<style>{''.join(rules)}</style><span style=\"display:none\"></span>",
            unsafe_allow_javascript=True,
        )


def render_mobile_brand() -> None:
    """Small 'FinRAG' mark, visible ONLY below the 640px breakpoint (see
    .finrag-mobile-brand in THEME_CSS). Streamlit's sidebar collapses/hides
    on narrow viewports by default, which would otherwise take the sidebar
    wordmark with it -- this keeps the brand visible at the top even then,
    without duplicating it on desktop where the sidebar already shows it."""
    st.html('<div class="finrag-mobile-brand">Fin<span class="brass">RAG</span></div>')


def render_status_badge() -> None:
    st.html(
        '<div class="finrag-status-row">'
        '<span class="finrag-status-dot" aria-hidden="true"></span>'
        "FY2022 FILINGS &nbsp;&middot;&nbsp; GROUNDED"
        "</div>"
    )


def render_sidebar(example_questions: list[str], on_pick) -> list[str]:
    """Renders the left brand/research rail: hero wordmark + tagline,
    company scope filter, suggested questions. Returns the selected
    doc_names -- empty list means no filter, search the whole corpus
    (retrieve()'s existing default)."""
    short_to_doc = {short_company(v): k for k, v in config.CORPUS_DOCS.items()}
    with st.sidebar:
        st.html('<div class="finrag-wordmark">Fin<span class="brass">RAG</span></div>')
        st.html('<div class="finrag-tagline">Financial intelligence, grounded in the filings.</div>')

        st.html('<div class="finrag-sidebar-label">Research</div>')
        selected_short = st.multiselect(
            "Scope to filings",
            options=list(short_to_doc.keys()),
            placeholder="All filings",
            label_visibility="collapsed",
        )
        st.html('<div class="finrag-sidebar-label">Suggested questions</div>')
        for i, q in enumerate(example_questions):
            st.button(q, key=f"example_{i}", on_click=on_pick, args=(q,), width="stretch")
    return [short_to_doc[s] for s in selected_short]


# ---------------------------------------------------------------------------
# Answer document: eyebrow, question title, metadata row, styled body,
# citation badges linked to the evidence panel, copy action.
# ---------------------------------------------------------------------------
_CITATION_RE = re.compile(r"\(([A-Za-z0-9&.'\- ]+), page (\d+)\)")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_NUM_LIST_RE = re.compile(r"^\d+[.)]\s+")
_BULLET_RE = re.compile(r"^[-•*]\s+")


def _find_evidence_index(company_text: str, page_num: str, chunks: list[dict]) -> int | None:
    needle = company_text.strip().lower()
    for i, c in enumerate(chunks):
        meta = c["metadata"]
        if str(meta["page_number"]) != page_num:
            continue
        if needle in meta["company"].lower():
            return i
    return None


def _citations_to_badges(body_html: str, chunks: list[dict]) -> str:
    def _replace(m: re.Match) -> str:
        company_text, page_num = m.group(1).strip(), m.group(2)
        idx = _find_evidence_index(company_text, page_num, chunks)
        label = f"{company_text} · p.{page_num}"
        if idx is not None:
            return f'<a href="#evidence-{idx}" class="finrag-citation">{label}</a>'
        return f'<span class="finrag-citation finrag-citation--unlinked">{label}</span>'

    return _CITATION_RE.sub(_replace, body_html)


def _render_lines(lines: list[str]) -> str:
    """Group CONSECUTIVE same-type lines within one block into their own
    paragraph/list run, rather than requiring an entire block to be
    uniformly list-like -- real model output routinely puts an intro line
    directly above a numbered list with no blank line between them."""
    out = []
    i = 0
    while i < len(lines):
        if _NUM_LIST_RE.match(lines[i]):
            run = []
            while i < len(lines) and _NUM_LIST_RE.match(lines[i]):
                run.append(f"<li>{_NUM_LIST_RE.sub('', lines[i])}</li>")
                i += 1
            out.append(f"<ol>{''.join(run)}</ol>")
        elif _BULLET_RE.match(lines[i]):
            run = []
            while i < len(lines) and _BULLET_RE.match(lines[i]):
                run.append(f"<li>{_BULLET_RE.sub('', lines[i])}</li>")
                i += 1
            out.append(f"<ul>{''.join(run)}</ul>")
        else:
            run = []
            while i < len(lines) and not _NUM_LIST_RE.match(lines[i]) and not _BULLET_RE.match(lines[i]):
                run.append(lines[i])
                i += 1
            out.append(f"<p>{'<br>'.join(run)}</p>")
    return "".join(out)


def _format_answer_html(answer_text: str, chunks: list[dict]) -> str:
    """Plain model text -> typeset HTML: paragraphs, simple numbered/bulleted
    lists, bold, and citation markers upgraded to badges linked to the
    matching evidence card (left unlinked, never fabricated, if no chunk
    matches). Escapes first -- the model's text is effectively untrusted
    input (it can be steered by content hidden in a retrieved filing chunk),
    so this never trusts it to already be safe HTML."""
    escaped = html.escape(answer_text.strip())
    blocks = re.split(r"\n\s*\n", escaped)

    rendered = []
    for block in blocks:
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        rendered.append(_render_lines(lines))

    body = _BOLD_RE.sub(r"<strong>\1</strong>", "".join(rendered))
    body = _citations_to_badges(body, chunks)
    return f'<div class="finrag-answer-body">{body}</div>'


def render_copy_button(text: str) -> None:
    payload = html.escape(json.dumps(text), quote=True)
    snippet = """
    <button class="finrag-copy-btn" type="button" aria-label="Copy answer text"
            onclick="var b=this, prev=b.textContent;
                     navigator.clipboard.writeText(__PAYLOAD__).then(function(){
                       b.textContent='Copied';
                       setTimeout(function(){ b.textContent=prev; }, 1400);
                     });">
      Copy answer
    </button>
    """.replace("__PAYLOAD__", payload)
    # st.html() strips inline event handlers by default (unsafe_allow_javascript=False) --
    # needed here for the onclick clipboard call. Scoped to just this one snippet, whose
    # only dynamic content is `payload` (json.dumps + html.escape of the answer text).
    st.html(snippet, unsafe_allow_javascript=True)


def render_answer_document(question: str, result: dict, chunks: list[dict], scope_labels: list[str]) -> None:
    scope = ", ".join(scope_labels) if scope_labels else "All filings"
    best_sim = max((c["similarity"] for c in chunks), default=0.0)
    low_confidence = best_sim < config.MIN_RELEVANCE_SCORE
    state = "Low confidence" if low_confidence else "Answered"

    with st.container(key="answer-surface"):
        st.html('<div class="finrag-eyebrow">Grounded answer</div>')
        st.header(question)  # h2 -- "FinRAG" in the sidebar is the page's one h1
        st.html(f"""
        <div class="finrag-answer-meta">
          <span>Scope <b>{html.escape(scope)}</b></span>
          <span>Fiscal year <b>FY2022</b></span>
          <span>Sources <b>{len(chunks)}</b></span>
          <span>Status <b>{state}</b></span>
        </div>
        """)

        st.html(_format_answer_html(result["answer"], chunks))

        badge_col, copy_col = st.columns([3, 1], vertical_alignment="center")
        with badge_col:
            if result["used_tool"]:
                st.badge("Calculator used", icon=":material/calculate:", color="orange")
            if low_confidence:
                st.badge("Low-confidence match", icon=":material/warning:", color="red")
        with copy_col:
            render_copy_button(result["answer"])


def render_evidence_panel(chunks: list[dict]) -> None:
    st.html('<div class="finrag-sidebar-label">Evidence</div>')
    cards = []
    for i, c in enumerate(chunks):
        meta = c["metadata"]
        excerpt = html.escape(c["text"][:280] + ("..." if len(c["text"]) > 280 else ""))
        company = html.escape(short_company(meta["company"]))
        cards.append(f"""
        <div class="finrag-evidence-card" id="evidence-{i}">
          <div class="finrag-evidence-head">
            <span class="finrag-evidence-company">{company}</span>
            <span>p.{meta['page_number']} · sim {c['similarity']:.2f}</span>
          </div>
          <div class="finrag-evidence-excerpt">{excerpt}</div>
        </div>
        """)
    st.html("".join(cards))
