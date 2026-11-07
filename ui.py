"""ui.py - look & feel of ContextFix (CSS + small HTML builders)."""
from __future__ import annotations

import html

import streamlit as st

KIND_CLASS = {"Spelling": "spelling", "Context": "context", "Grammar": "grammar"}


def esc(s: str) -> str:
    """HTML-escape text and make it safe inside st.markdown (no blank lines, no $ math)."""
    return html.escape(s).replace("$", "&#36;").replace("\r\n", "\n").replace("\n", "<br>")


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: #fbfaf5; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1180px; padding-top: 1.2rem; padding-bottom: 4rem; }
h1, h2, h3 { letter-spacing: -0.02em; }

/* top bar */
.cf-top { display:flex; align-items:center; justify-content:space-between; padding: 6px 0 14px 0; }
.cf-brand { display:flex; align-items:center; gap:12px; }
.cf-logo { width:34px; height:34px; border-radius:9px; background:#2340e6; color:#fff; display:flex;
  align-items:center; justify-content:center; font-weight:700; font-size:15px; font-family:'JetBrains Mono',monospace; }
.cf-brand b { font-size:15px; color:#16181d; display:block; line-height:1.1; }
.cf-brand small { font-size:9.5px; letter-spacing:.14em; color:#6b7280; text-transform:uppercase; }
.cf-pill { font-size:12px; color:#4b5563; border:1px solid #e5e2d6; background:#fffefa; border-radius:999px; padding:5px 12px; }
.cf-pill:before { content:''; display:inline-block; width:7px; height:7px; border-radius:50%; background:#10b981; margin-right:8px; }
.cf-nav { display:inline-flex; align-items:center; gap:8px; margin: 0 0 16px 0; padding:7px 14px; border-radius:9px;
  background:#e9edff; color:#2340e6; font-weight:600; font-size:13px; }

/* hero */
.cf-hero { background: radial-gradient(1200px 400px at 85% 0%, #2b2f9a 0%, rgba(43,47,154,0) 60%), linear-gradient(135deg,#0b1030 0%,#12163f 55%,#101a55 100%);
  border-radius: 22px; padding: 44px 46px; display:flex; gap:40px; align-items:center; justify-content:space-between; flex-wrap:wrap; }
.cf-hero h1 { color:#fff; font-size: clamp(34px, 5vw, 54px); line-height:1.02; font-weight:800; margin:0 0 18px 0; padding:0; }
.cf-hero h1 span { color:#4f8bff; display:block; }
.cf-hero p { color:#b8c0e6; max-width:470px; font-size:15px; line-height:1.55; margin:0 0 22px 0; }
.cf-btn { display:inline-block; background:#fff; color:#0b1030 !important; font-weight:600; font-size:14px; padding:10px 18px;
  border-radius:10px; text-decoration:none !important; }
.cf-stats { display:grid; grid-template-columns: 1fr 1fr; gap:22px 46px; }
.cf-stats small { display:block; color:#8f9ad6; font-size:10px; letter-spacing:.14em; text-transform:uppercase; font-family:'JetBrains Mono',monospace; }
.cf-stats b { display:block; color:#fff; font-size:30px; font-weight:700; line-height:1.1; margin-top:2px; }
.cf-stats em { color:#8f9ad6; font-size:12px; font-style:normal; }

/* generic cards */
.cf-card { background:#fffefa; border:1px solid #e8e5d8; border-radius:18px; padding:22px 24px; height:100%; }
.cf-label { font-family:'JetBrains Mono',monospace; font-size:10.5px; letter-spacing:.14em; text-transform:uppercase; color:#2340e6; font-weight:600; }
.cf-steps { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:14px; }
.cf-steps small { color:#2340e6; font-family:'JetBrains Mono',monospace; font-size:11px; }
.cf-steps b { display:block; font-size:15px; margin:6px 0 4px 0; color:#16181d; }
.cf-steps span { font-size:12.5px; color:#6b7280; line-height:1.4; display:block; }
.cf-objective { background:#eef2ff; border:1px solid #dbe3ff; border-radius:18px; padding:22px 24px; height:100%; }
.cf-objective p { color:#374151; font-size:13.5px; line-height:1.6; margin:10px 0 0 0; }

.cf-section-no { font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:.12em; color:#2340e6; text-transform:uppercase; margin-top:42px; }
.cf-section-title { font-size: clamp(28px,4vw,40px); font-weight:800; line-height:1.08; color:#16181d; margin:8px 0 8px 0; max-width:760px; }
.cf-section-sub { color:#6b7280; font-size:14px; margin-bottom:18px; }

/* result panel */
.cf-count { font-size:12px; color:#6b7280; text-align:right; margin-top:-6px; }
.cf-verdict-top { display:flex; justify-content:space-between; align-items:center; }
.cf-issue-pill { font-size:11.5px; font-weight:600; color:#92400e; background:#fef3c7; padding:4px 10px; border-radius:999px; }
.cf-issue-pill.ok { color:#166534; background:#dcfce7; }
.cf-chip { display:inline-block; font-size:11.5px; font-weight:600; padding:3px 10px; border-radius:999px; margin:0 6px 4px 0; }
.cf-chip:before { content:''; display:inline-block; width:6px; height:6px; border-radius:50%; margin-right:6px; background:currentColor; }
.k-spelling { color:#be123c; background:#ffe4e6; }
.k-context  { color:#b45309; background:#fef3c7; }
.k-grammar  { color:#6d28d9; background:#ede9fe; }
.cf-box { background:#f6f4ea; border:1px solid #ece8d6; border-radius:12px; padding:14px 16px; font-size:14px; line-height:1.75; color:#16181d; word-wrap:break-word; }
.cf-box.out { background:#eef2ff; border-color:#dbe3ff; }
.cf-box-label { font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:.14em; color:#6b7280; text-transform:uppercase; margin:14px 0 6px 0; }
.hl { border-radius:4px; padding:0 3px; font-weight:500; }
.hl-spelling { background:#ffe4e6; color:#be123c; text-decoration: underline wavy #e11d48; text-underline-offset:3px; }
.hl-context  { background:#fef3c7; color:#92400e; text-decoration: underline wavy #d97706; text-underline-offset:3px; }
.hl-grammar  { background:#ede9fe; color:#5b21b6; text-decoration: underline wavy #7c3aed; text-underline-offset:3px; }
.fix { background:#dcfce7; color:#166534; border-radius:4px; padding:0 3px; font-weight:600; }

/* suggestion cards */
.cf-sug-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }
.cf-sug-top small { color:#9ca3af; font-size:11px; }
.cf-change { font-size:15px; font-weight:600; margin:2px 0 6px 0; }
.cf-change s { color:#9ca3af; font-weight:500; margin-right:4px; }
.cf-change i { font-style:normal; color:#9ca3af; margin:0 6px; }
.cf-change b { color:#166534; }
.cf-change.off b { color:#9ca3af; text-decoration: line-through; }
.cf-msg { font-size:12.5px; color:#6b7280; line-height:1.45; margin-bottom:8px; }
.cf-method { font-family:'JetBrains Mono',monospace; font-size:10.5px; color:#2340e6; margin-bottom:8px; }
.cf-bar { display:flex; align-items:center; gap:10px; margin:3px 0; font-size:12px; }
.cf-bar .w { width:84px; font-family:'JetBrains Mono',monospace; color:#374151; overflow:hidden; text-overflow:ellipsis; }
.cf-bar .t { flex:1; height:6px; background:#ece8d6; border-radius:99px; overflow:hidden; }
.cf-bar .t span { display:block; height:100%; background:linear-gradient(90deg,#4f46e5,#7c3aed); border-radius:99px; }
.cf-bar .p { width:48px; text-align:right; font-family:'JetBrains Mono',monospace; color:#6b7280; }
.cf-legend { display:flex; gap:18px; flex-wrap:wrap; font-size:12px; color:#6b7280; margin-top:10px; }
.cf-legend span:before { content:''; display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; background:var(--c); }

/* streamlit widgets */
div[data-testid="stTextArea"] textarea { background:#fbfaf2; border:1px solid #e8e5d8; border-radius:12px; font-size:14.5px; line-height:1.6; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius:16px; border-color:#e8e5d8; background:#fffefa; }
.stButton > button { border-radius:10px; font-weight:600; font-size:13px; border:1px solid #e1ddcc; padding:.3rem .8rem; }
.stButton > button[kind="primary"] { background:#2340e6; border-color:#2340e6; color:#fff; }
.stButton > button[kind="primary"]:hover { background:#1a30c0; border-color:#1a30c0; }
.cf-foot { display:flex; justify-content:space-between; color:#9ca3af; font-size:12px; margin-top:50px; border-top:1px solid #e8e5d8; padding-top:14px; }
@media (max-width: 760px) { .cf-steps { grid-template-columns: 1fr; } .cf-hero { padding: 28px 22px; } }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def topbar() -> None:
    st.markdown(
        '<div class="cf-top"><div class="cf-brand"><div class="cf-logo">&lt;/&gt;</div>'
        '<div><b>ContextFix</b><small>Language modeling</small></div></div>'
        '<div class="cf-pill">Local corpus · n-gram model</div></div>'
        '<div class="cf-nav">✦ Corrector</div>',
        unsafe_allow_html=True,
    )


def hero(stats: dict[str, int]) -> None:
    st.markdown(
        '<div class="cf-hero"><div><h1>Meaning lives<span>between the words.</span></h1>'
        '<p>Explore how local word probabilities reveal a sentence that is spelled right, but feels wrong. '
        'Every score comes from an inspectable English corpus and a huge dictionary.</p>'
        '<a class="cf-btn" href="#corrector">Try the corrector &rarr;</a></div>'
        '<div class="cf-stats">'
        f'<div><small>Training sentences</small><b>{stats["sentences"]:,}</b><em>corpus examples</em></div>'
        f'<div><small>Dictionary</small><b>{stats["vocab"]:,}</b><em>known words</em></div>'
        f'<div><small>Bigrams</small><b>{stats["bigrams"]:,}</b><em>word-pair types</em></div>'
        f'<div><small>Trigrams</small><b>{stats["trigrams"]:,}</b><em>three-word sequences</em></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )


def intro_cards() -> None:
    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.markdown(
            '<div class="cf-card"><div class="cf-label">↗ A correction, made visible</div><div class="cf-steps">'
            '<div><small>01</small><b>Tokenize</b><span>Split any text - sentence, paragraph or story - into words and punctuation.</span></div>'
            '<div><small>02</small><b>Score context</b><span>Compare candidate words using the left and right n-grams around them.</span></div>'
            '<div><small>03</small><b>Suggest carefully</b><span>Show a change only when local evidence is strong enough.</span></div>'
            '</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(
            '<div class="cf-objective"><div class="cf-label">Project objective</div>'
            '<p>To develop an intelligent, Grammarly-like text-correction application that detects spelling, '
            'grammatical and contextual errors and suggests accurate corrections using NLP techniques such as '
            'variable-length n-gram language models, conditional probability and Bayesian parameter estimation '
            '(Dirichlet-smoothed estimates). Its goal is to improve accuracy and clarity through context-aware suggestions.</p></div>',
            unsafe_allow_html=True)


def section_heading() -> None:
    st.markdown(
        '<div id="corrector"></div><div class="cf-section-no">01 · Contextual correction</div>'
        '<div class="cf-section-title">A spelling checker sees letters. This one reads the room.</div>'
        '<div class="cf-section-sub">Paste a sentence, a paragraph or a whole story. Every word is checked for spelling, '
        'grammar and context errors, and you decide on each suggestion.</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- results
def chips(counts: dict[str, int]) -> str:
    return "".join(f'<span class="cf-chip k-{KIND_CLASS[k]}">{k} · {v}</span>' for k, v in counts.items())


def render_original(text, issues, rejected) -> str:
    out, pos = [], 0
    for iss in sorted(issues, key=lambda i: i.start):
        if iss.start < pos:
            continue
        out.append(esc(text[pos:iss.start]))
        out.append(f'<span class="hl hl-{KIND_CLASS[iss.kind]}" title="{html.escape(iss.message)}">{esc(iss.original)}</span>')
        pos = iss.end
    out.append(esc(text[pos:]))
    return f'<div class="cf-box">{"".join(out)}</div>'


def render_corrected(text, issues, rejected) -> str:
    out, pos = [], 0
    for iss in sorted(issues, key=lambda i: i.apply_start):
        if iss.apply_start < pos:
            continue
        out.append(esc(text[pos:iss.apply_start]))
        if iss.id in rejected:
            out.append(esc(text[iss.apply_start:iss.end]))
        elif iss.replacement:
            out.append(f'<span class="fix">{esc(iss.replacement)}</span>')
        pos = iss.end
    out.append(esc(text[pos:]))
    return f'<div class="cf-box out">{"".join(out)}</div>'


def score_bars(cands) -> str:
    rows = []
    for w, p in cands[:4]:
        rows.append(f'<div class="cf-bar"><span class="w">{html.escape(w)}</span>'
                    f'<span class="t"><span style="width:{max(p * 100, 1.5):.1f}%"></span></span>'
                    f'<span class="p">{p * 100:.1f}%</span></div>')
    return "".join(rows)


def suggestion_html(iss, rejected: bool) -> str:
    kind = KIND_CLASS[iss.kind]
    repl = esc(iss.replacement) if iss.replacement else "(remove)"
    body = (
        f'<div class="cf-sug-top"><span class="cf-chip k-{kind}">{iss.kind}</span><small>Sentence {iss.sentence}</small></div>'
        f'<div class="cf-change{" off" if rejected else ""}"><s>{esc(iss.original)}</s><i>&rarr;</i><b>{repl}</b></div>'
        f'<div class="cf-msg">{html.escape(iss.message)}</div>'
    )
    if iss.candidates:
        top = iss.candidates[0][1] * 100
        body += f'<div class="cf-method">{top:.1f}% relative score</div>' + score_bars(iss.candidates)
    else:
        body += f'<div class="cf-method">{html.escape(iss.method)}</div>'
    return body


def legend() -> None:
    st.markdown(
        '<div class="cf-legend"><span style="--c:#e11d48">Spelling</span><span style="--c:#d97706">Context</span>'
        '<span style="--c:#7c3aed">Grammar</span><span style="--c:#22c55e">Applied correction (unless rejected)</span></div>',
        unsafe_allow_html=True)


def footer() -> None:
    st.markdown('<div class="cf-foot"><span>ContextFix · Language Modeling</span>'
                '<span>Local statistical NLP · transparent counts · no external AI services</span></div>',
                unsafe_allow_html=True)
