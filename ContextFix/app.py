"""
app.py - ContextFix: Context-Aware Grammar and Typing Error Recovery System
Run with:   streamlit run app.py
"""
import streamlit as st

import ui
from engine import ContextFix, apply_issues

st.set_page_config(page_title="ContextFix", page_icon="✍️", layout="wide")
ui.inject_css()

MAX_CHARS = 50_000
PAGE_SIZE = 8

SAMPLES = {
    "Paragraph with many errors": (
        "i recieve teh book yesterday and it was definately good. She go to a apple store with there friends.\n\n"
        "He dont like the movie because it was to long. This is better then that, and the cat chased it's toy."
    ),
    "A short story": (
        "Once upon a time there lived a yound girl named Maya who loved to read bookes about adventurs. "
        "Every morning she walked to the libary and askd the librarian for a new storry. "
        "One day she find a old map hidden between two pages, and it show a path to a secret garden.\n\n"
        "Maya dont tell nobody about it. Their was alot of peopel in the town, but she wanted to go alone. "
        "She packed a apple, a bottle of water, and her fathers compass. When she reached the garden, "
        "the flowers were blooming and the birds were singing. It was the most beautful place she had ever seen."
    ),
    "I am going too college.": "I am going too college.",
    "I ate to much food at lunch.": "I ate to much food at lunch.",
    "Their going to present their research.": "Their going to present their research.",
    "Your going to enjoy the show.": "Your going to enjoy the show.",
    "This book is better then that one.": "This book is better then that one.",
}


@st.cache_resource(show_spinner="Building the language model from the corpus (first start only, ~20 s)...")
def load_model() -> ContextFix:
    return ContextFix()


model = load_model()

# ------------------------------------------------------------------ session state
ss = st.session_state
ss.setdefault("text", "")
ss.setdefault("analysis", None)
ss.setdefault("rejected", set())
ss.setdefault("page", 0)
ss.setdefault("kind_filter", "All")


def use_sample(name: str):
    ss["text"] = SAMPLES[name]


def accept_all():
    ss["rejected"] = set()


def reject_all():
    ss["rejected"] = {i.id for i in ss["analysis"].issues}


def decide(issue_id: str, accepted: bool):
    r = set(ss["rejected"])
    r.discard(issue_id) if accepted else r.add(issue_id)
    ss["rejected"] = r


def go_page(delta: int):
    ss["page"] = max(0, ss["page"] + delta)


def reset_page():
    ss["page"] = 0


# ------------------------------------------------------------------ sidebar settings
with st.sidebar:
    st.header("⚙️ Model settings")
    order = st.slider("Max context length (n-gram order)", 1, 3, 3,
                      help="1 = unigram (word frequency only), 2 = bigram, 3 = trigram. The model automatically "
                           "uses a SHORTER history when the longer one was never seen (variable-length LM).")
    lam = st.slider("Bayesian prior strength (λ)", 1.0, 50.0, 10.0, step=1.0,
                    help="Dirichlet prior weight. Small λ trusts the observed counts; large λ trusts the "
                         "shorter-context prior more.")
    ignore_raw = st.text_area("Words to ignore (names, slang…)", placeholder="Javeria, Hyderabad, btech",
                              help="Separate words with commas or new lines. They will never be flagged.")
    ignore = {w.strip().lower() for w in ignore_raw.replace("\n", ",").split(",") if w.strip()}
    st.caption(f"Dictionary: {model.stats()['vocab']:,} words")

# ------------------------------------------------------------------ page
ui.topbar()
ui.hero(model.stats())
st.write("")
ui.intro_cards()
ui.section_heading()

left, right = st.columns(2, gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="cf-label">Your text</div>', unsafe_allow_html=True)
        st.text_area("Your text", key="text", height=300, max_chars=MAX_CHARS, label_visibility="collapsed",
                     placeholder="Type or paste anything here - a sentence, a paragraph, a whole story...")
        st.markdown(f'<div class="cf-count">{len(ss["text"]):,} / {MAX_CHARS:,} characters</div>', unsafe_allow_html=True)
        st.markdown('<div class="cf-label" style="color:#6b7280;margin:6px 0">Try a sample</div>', unsafe_allow_html=True)
        names = list(SAMPLES)
        for row in (names[:2], names[2:4], names[4:]):
            cols = st.columns(len(row))
            for c, name in zip(cols, row):
                c.button(name if len(name) < 34 else name[:31] + "…", key=f"sample_{name}",
                         on_click=use_sample, args=(name,))
        if st.button("✦ Analyze text", type="primary", key="analyze"):
            if ss["text"].strip():
                with st.spinner("Scoring every word in context..."):
                    ss["analysis"] = model.analyze(ss["text"], order=order, lam=lam, ignore=ignore)
                ss["rejected"] = set()
                ss["page"] = 0
            else:
                st.warning("Please type or paste some text first.")

analysis = ss["analysis"]

with right:
    with st.container(border=True):
        if analysis is None:
            st.markdown('<div class="cf-label">Model output</div><div class="cf-section-title" style="font-size:24px">'
                        'Context verdict</div><div class="cf-section-sub">Press <b>Analyze text</b> to see highlighted '
                        'errors and the corrected version here.</div>', unsafe_allow_html=True)
        else:
            n = len(analysis.issues)
            pill = (f'<span class="cf-issue-pill">{n} issue{"s" if n != 1 else ""} in '
                    f'{analysis.sentences_with_issues} sentence{"s" if analysis.sentences_with_issues != 1 else ""}</span>'
                    if n else '<span class="cf-issue-pill ok">No issues found</span>')
            st.markdown(f'<div class="cf-verdict-top"><div><div class="cf-label">Model output</div>'
                        f'<div class="cf-section-title" style="font-size:24px;margin:4px 0 8px 0">Context verdict</div></div>{pill}</div>'
                        + ui.chips(analysis.counts()), unsafe_allow_html=True)
            if ss["text"] != analysis.text:
                st.info("The text changed since the last analysis - press **Analyze text** to refresh.")
            st.markdown('<div class="cf-box-label">Original text · issues highlighted</div>'
                        + ui.render_original(analysis.text, analysis.issues, ss["rejected"]), unsafe_allow_html=True)
            st.markdown('<div class="cf-box-label">Corrected text</div>'
                        + ui.render_corrected(analysis.text, analysis.issues, ss["rejected"]), unsafe_allow_html=True)
            corrected = apply_issues(analysis.text, analysis.issues, ss["rejected"])
            st.write("")
            st.download_button("⬇ Download corrected text", corrected, file_name="corrected.txt", key="dl")
            with st.expander("Copy corrected text"):
                st.code(corrected, language=None)

# ------------------------------------------------------------------ review section
if analysis is not None and analysis.issues:
    st.markdown('<div class="cf-section-no" style="margin-top:34px">Review</div>'
                '<div class="cf-section-title" style="font-size:28px">Suggestions, one by one</div>',
                unsafe_allow_html=True)
    b1, b2, b3, _, f1 = st.columns([1, 1, 1, 0.4, 3])
    b1.button("✓ Accept all", on_click=accept_all, key="acc_all")
    b2.button("✕ Reject all", on_click=reject_all, key="rej_all")
    b3.button("↺ Reset", on_click=accept_all, key="reset")
    kind_filter = f1.radio("Show", ["All", "Spelling", "Context", "Grammar"], horizontal=True, key="kind_filter",
                           on_change=reset_page, label_visibility="collapsed")

    shown = [i for i in analysis.issues if kind_filter == "All" or i.kind == kind_filter]
    pages = max(1, -(-len(shown) // PAGE_SIZE))
    ss["page"] = min(ss["page"], pages - 1)
    chunk = shown[ss["page"] * PAGE_SIZE:(ss["page"] + 1) * PAGE_SIZE]

    cols = st.columns(2, gap="medium")
    for k, iss in enumerate(chunk):
        is_rejected = iss.id in ss["rejected"]
        with cols[k % 2]:
            with st.container(border=True):
                st.markdown(ui.suggestion_html(iss, is_rejected), unsafe_allow_html=True)
                a, r = st.columns(2)
                a.button("✓ Accept", key=f"a_{iss.id}", on_click=decide, args=(iss.id, True),
                         type="secondary" if is_rejected else "primary")
                r.button("✕ Reject", key=f"r_{iss.id}", on_click=decide, args=(iss.id, False),
                         type="primary" if is_rejected else "secondary")
    if pages > 1:
        p1, p2, p3 = st.columns([1, 2, 1])
        p1.button("← Previous", on_click=go_page, args=(-1,), disabled=ss["page"] == 0, key="prev")
        p2.markdown(f'<div style="text-align:center;color:#6b7280;font-size:13px;padding-top:6px">'
                    f'Page {ss["page"] + 1} of {pages} · {len(shown)} suggestions</div>', unsafe_allow_html=True)
        p3.button("Next →", on_click=go_page, args=(1,), disabled=ss["page"] >= pages - 1, key="next")
    ui.legend()

# ------------------------------------------------------------------ explanation
with st.expander("🔬 How it works (the two course concepts)"):
    st.markdown("""
**1. Variable-length language model.** The model estimates the probability of a word from the words before it.
It tries a trigram (2 words of history); if that history was never seen in the corpus it **falls back to a bigram,
then to a unigram**. So the amount of context actually used changes from word to word.

**2. Bayesian parameter estimation.** A plain maximum-likelihood estimate gives probability 0 to anything unseen.
Instead, each word distribution gets a **Dirichlet prior** and we use the posterior mean, with the next-shorter model
as the prior:
""")
    st.latex(r"P_n(w\mid h)=\frac{c(h,w)+\lambda\,P_{n-1}(w\mid h')}{c(h)+\lambda}")
    st.markdown("""
At the bottom of the chain the prior is a frequency list of ~300,000 dictionary words. (With a uniform prior this
reduces to Laplace / add-one smoothing.)

**Three kinds of checks**
- **Spelling** - noisy channel: choose the candidate *c* maximising `P(c | context) × P(typo | c)`.
- **Context** - real-word errors such as *their / there / they're* or *then / than*: compare the language-model
  probability of every member of a confusion set in the sentence.
- **Grammar** - small rules: *a / an*, subject-verb agreement, capital letters, repeated words, missing apostrophes.
""")

ui.footer()
