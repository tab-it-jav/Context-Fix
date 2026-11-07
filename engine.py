"""
engine.py  -  the NLP brain of ContextFix
==========================================

Two Unit-V ideas power this file:

1. VARIABLE-LENGTH LANGUAGE MODEL (n-gram with a growing history)
   The model tries a trigram (2 words of history), but if that history was never
   seen in the training corpus it automatically falls back to a bigram, and then a
   unigram.  So the length of the context that is actually used VARIES from word
   to word.

2. BAYESIAN PARAMETER ESTIMATION
   Plain maximum-likelihood estimates (count / total) give 0 to every unseen
   n-gram.  Instead we treat the word probabilities as the parameters of a
   multinomial with a Dirichlet prior and use the posterior mean:

        P_n(w | h) = ( c(h, w) + lambda * P_{n-1}(w | h') ) / ( c(h) + lambda )

   where the prior P_{n-1} is the next-shorter model.  At the bottom of the chain
   the prior is a big dictionary frequency list.  (With a uniform prior this
   reduces to Laplace / add-one smoothing.)

On top of the model we build three kinds of checks:
   * Spelling  - noisy-channel:  argmax  P(candidate | context) * P(typo | candidate)
   * Context   - real-word errors (their/there, then/than ...) by comparing the
                 language-model probability of each member of a confusion set
   * Grammar   - small rule set (a/an, subject-verb agreement, capital letters ...)
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import nltk
from spellchecker import SpellChecker

DATA_DIR = Path(__file__).parent / "data"
LETTERS = "abcdefghijklmnopqrstuvwxyz"

LM_WEIGHT = 0.6      # how much the language model counts compared with the typo (channel) model

# --------------------------------------------------------------------------------------
# Linguistic resources
# --------------------------------------------------------------------------------------
TWO_LETTER_WORDS = set(
    "am an as at ax be by do eh go ha he hi if in is it me my no of oh ok on or ox so to up us we yo".split()
)

CONFUSION_SETS = [
    ("then", "than"),
    ("its", "it's"),
    ("their", "there", "they're"),
    ("your", "you're"),
    ("to", "too"),
    ("affect", "effect"),
    ("accept", "except"),
    ("lose", "loose"),
    ("whose", "who's"),
    ("principal", "principle"),
    ("were", "where", "we're"),
    ("weather", "whether"),
    ("quiet", "quite"),
    ("hear", "here"),
    ("advice", "advise"),
    ("by", "buy"),
]
CONFUSION_INDEX = {w: s for s in CONFUSION_SETS for w in s}

CONTRACTION_FIX = {
    "dont": "don't", "doesnt": "doesn't", "didnt": "didn't", "cant": "can't", "wont": "won't",
    "isnt": "isn't", "arent": "aren't", "wasnt": "wasn't", "werent": "weren't",
    "hasnt": "hasn't", "havent": "haven't", "hadnt": "hadn't", "couldnt": "couldn't",
    "shouldnt": "shouldn't", "wouldnt": "wouldn't", "mustnt": "mustn't",
    "im": "I'm", "ive": "I've", "youre": "you're", "theyre": "they're", "weve": "we've",
    "youve": "you've", "theyve": "they've", "youll": "you'll", "theyll": "they'll",
    "thats": "that's", "whats": "what's", "theres": "there's", "wouldve": "would've",
    "couldve": "could've", "shouldve": "should've",
}

I_FORMS = {"i": "I", "i'm": "I'm", "i've": "I've", "i'll": "I'll", "i'd": "I'd"}

# Verbs whose base form equals their past tense (put, hit, read ...) are left out on purpose.
BASE_VERBS = set("""
go do have want like need make take come see know think say get give find tell work call try ask feel
leave mean keep begin seem help show hear play run move live believe bring write sit stand lose pay meet
include continue learn change lead understand watch follow stop create speak spend grow open walk win
offer remember love consider appear buy wait serve die send expect build stay fall reach kill remain
suggest raise pass sell require report decide pull eat drink sleep cook drive ride enjoy study teach
prefer hate wish use look turn start hold talk sound add happen plan wear save finish fix break catch
choose forget carry fly swim sing dance laugh cry smile visit clean wash miss answer arrive agree allow
become describe develop explain hope imagine improve invite join knock listen order paint push relax
repeat return rely share shout sleep travel trust wonder worry
""".split())


def third_person(verb: str) -> str:
    irregular = {"go": "goes", "do": "does", "have": "has"}
    if verb in irregular:
        return irregular[verb]
    if re.search(r"(s|x|z|ch|sh)$", verb):
        return verb + "es"
    if re.search(r"[^aeiou]y$", verb):
        return verb[:-1] + "ies"
    return verb + "s"


THIRD_TO_BASE = {third_person(v): v for v in BASE_VERBS}

# If one of these comes right before a pronoun, the pronoun is NOT a plain subject
# ("let it go", "does she go", "to him", "told you ...") so agreement rules stay silent.
NOT_SUBJECT_CONTEXT = set("""
let lets make makes made help helps watch see saw hear heard tell told give gave show showed ask asked thank
thanks get got keep find found love like want need with for to of from about at on in by between than near
without against into over upon onto through toward around behind beyond within among under above
do does did will would can could should shall may might must dont doesnt didnt wont cant
don't doesn't didn't won't can't couldn't wouldn't shouldn't
""".split())
NOT_REPEAT_ERRORS = {"that", "had", "very", "no", "so", "bye", "ha", "yeah", "really", "knock", "blah"}
ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "st", "inc", "ltd", "no"}
# Words that can follow a CONTRACTION (it's / they're / you're / who's / we're) but not a possessive.
FUNC_AFTER_CONTR = set("""a an the not very so too just also still always never really now already even been being
going getting coming doing having gonna quite almost about to in on at for here there all
""".split())
# Adjectives that normally follow a contraction ("it's better", "you're right").
ADJ_AFTER_CONTR = set("""right wrong welcome ready sure correct late early good bad great nice beautiful better best
worse worst important true false possible impossible easy hard difficult fine okay ok interesting amazing wonderful
different same only first last next more less much many few enough able free open closed over done
""".split())
MODALS = {"should", "could", "would", "must", "might"}

# --------------------------------------------------------------------------------------
# Tokenizer / sentence splitter
# --------------------------------------------------------------------------------------
TOKEN_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*|\d+(?:[.,]\d+)*|\s+|.", re.S)


@dataclass
class Tok:
    text: str
    start: int
    end: int
    kind: str  # word | num | space | punct


def tokenize(text: str) -> list[Tok]:
    toks = []
    for m in TOKEN_RE.finditer(text):
        t = m.group()
        if t[0].isalpha():
            kind = "word"
        elif t[0].isdigit():
            kind = "num"
        elif t.isspace():
            kind = "space"
        else:
            kind = "punct"
        toks.append(Tok(t, m.start(), m.end(), kind))
    return toks


def split_sentences(toks: list[Tok]) -> list[list[int]]:
    """Return sentences as lists of token indexes (works for paragraphs, stories, many lines)."""
    sents, cur = [], []
    pending = False                      # a sentence ended and only a closing quote is left to absorb
    for i, t in enumerate(toks):
        cur.append(i)
        boundary = False
        if pending:
            pending = False
            boundary = True
        elif t.kind == "punct" and t.text in ".!?":
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            if nxt is None or nxt.kind == "space" or nxt.text in "\"')]”’":
                prev_word = next((toks[j].text.lower() for j in range(i - 1, -1, -1) if toks[j].kind == "word"), "")
                boundary = not (prev_word in ABBREVIATIONS or (len(prev_word) == 1 and t.text == "."))
        elif t.kind == "space" and "\n" in t.text:
            boundary = True
        if boundary:
            nxt_t = toks[i + 1] if i + 1 < len(toks) else None
            if t.text in ".!?" and nxt_t is not None and nxt_t.text in "\"”'’":
                pending = True    # keep the closing quote with this sentence, split right after it
                continue
            sents.append(cur)
            cur = []
    if cur:
        sents.append(cur)
    return [s for s in sents if any(toks[i].kind == "word" for i in s)]


def norm(word: str) -> str:
    return word.replace("’", "'").lower()


def apply_case(orig: str, new: str) -> str:
    if not new:
        return new
    if len(orig) > 1 and orig.isupper():
        return new.upper()
    if orig[:1].isupper():
        return new[:1].upper() + new[1:]
    return new


def article_for(word: str, raw: str) -> str:
    if len(raw) > 1 and raw.isupper():
        return "an" if raw[0] in "AEFHILMNORSX" else "a"
    if word.startswith(("hour", "honest", "honor", "honour", "heir")):
        return "an"
    if word in ("one", "once", "ones") or word.startswith(
        ("univers", "unif", "unil", "unio", "unit", "uniq", "unis", "unic", "use", "usu", "uti", "eu", "ubiq")
    ):
        return "a"
    return "an" if word[0] in "aeiou" else "a"


# --------------------------------------------------------------------------------------
# Spelling helpers
# --------------------------------------------------------------------------------------
def edits1(word: str) -> set[str]:
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits if b for c in LETTERS]
    inserts = [a + c + b for a, b in splits for c in LETTERS]
    return set(deletes + transposes + replaces + inserts)


# --------------------------------------------------------------------------------------
# The variable-length, Bayesian-smoothed n-gram model
# --------------------------------------------------------------------------------------
class NGramLM:
    def __init__(self, max_order: int = 3):
        self.K = max_order
        self.counts = {n: Counter() for n in range(1, max_order + 1)}
        self.ctx = {n: Counter() for n in range(2, max_order + 1)}
        self.N = 0
        self.dict_freq: dict[str, int] = {}
        self.dict_total = 1
        self.dict_size = 1
        self.sentences = 0

    def add_sentence(self, words: list[str], weight: int = 1):
        seq = ["<s>"] * (self.K - 1) + words + ["</s>"]
        for i in range(self.K - 1, len(seq)):
            self.counts[1][(seq[i],)] += weight
            self.N += weight
            for n in range(2, self.K + 1):
                g = tuple(seq[i - n + 1: i + 1])
                self.counts[n][g] += weight
                self.ctx[n][g[:-1]] += weight
        self.sentences += weight

    def dict_prob(self, w: str) -> float:
        return (self.dict_freq.get(w, 0) + 0.5) / (self.dict_total + 0.5 * self.dict_size)

    def prob(self, w: str, hist: list[str], order: int, lam: float, mu: float) -> tuple[float, int]:
        """Posterior-mean probability of w given its history. Returns (prob, order actually used)."""
        p = (self.counts[1].get((w,), 0) + mu * self.dict_prob(w)) / (self.N + mu)
        used = 1
        for n in range(2, order + 1):
            h = tuple(hist[-(n - 1):])
            tot = self.ctx[n].get(h, 0)
            if tot == 0:          # history never seen -> stop growing the context (variable length!)
                break
            c = self.counts[n].get(h + (w,), 0)
            p = (c + lam * p) / (tot + lam)
            used = n
        return p, used


# --------------------------------------------------------------------------------------
# Result objects
# --------------------------------------------------------------------------------------
@dataclass
class Issue:
    id: str
    kind: str                 # Spelling | Context | Grammar
    start: int                # character span of the flagged word in the original text
    end: int
    apply_start: int          # where the replacement begins (differs from start only for deletions)
    original: str
    replacement: str          # "" means "delete this word"
    sentence: int
    message: str
    method: str
    candidates: list[tuple[str, float]] = field(default_factory=list)  # (word, relative score 0-1)


@dataclass
class Analysis:
    text: str
    issues: list[Issue]
    n_sentences: int
    n_words: int

    def counts(self) -> dict[str, int]:
        c = Counter(i.kind for i in self.issues)
        return {k: c.get(k, 0) for k in ("Spelling", "Context", "Grammar")}

    @property
    def sentences_with_issues(self) -> int:
        return len({i.sentence for i in self.issues})


def apply_issues(text: str, issues: list[Issue], rejected: set[str]) -> str:
    out, pos = [], 0
    for iss in sorted(issues, key=lambda x: x.apply_start):
        if iss.id in rejected or iss.apply_start < pos:
            continue
        out.append(text[pos:iss.apply_start])
        out.append(iss.replacement)
        pos = iss.end
    out.append(text[pos:])
    return "".join(out)


# --------------------------------------------------------------------------------------
# The corrector
# --------------------------------------------------------------------------------------
class ContextFix:
    def __init__(self, seed_weight: int = 4, max_order: int = 3):
        for pkg, path in (("brown", "corpora/brown"), ("words", "corpora/words")):
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(pkg, quiet=True)
        from nltk.corpus import brown, words as nltk_words

        spell = SpellChecker()
        self.lm = NGramLM(max_order)
        self.lm.dict_freq = dict(spell.word_frequency.dictionary)
        self.lm.dict_total = max(1, sum(self.lm.dict_freq.values()))

        word_re = re.compile(r"[a-z]+(?:'[a-z]+)*$")

        def clean(tokens):
            out = []
            for t in tokens:
                t = norm(t)
                if word_re.match(t):
                    out.append(t)
            return out

        brown_counts: Counter = Counter()
        for sent in brown.sents():
            ws = clean(sent)
            if ws:
                self.lm.add_sentence(ws)
                brown_counts.update(ws)

        seed_path = DATA_DIR / "seed_corpus.txt"
        self.seed_sentences = 0
        if seed_path.exists():
            for line in seed_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                for s in re.split(r"(?<=[.!?])\s+", line):
                    ws = clean(tokenize_words(s))
                    if ws:
                        self.lm.add_sentence(ws, weight=seed_weight)
                        self.seed_sentences += 1

        # ---- the "unlimited" dictionary: three big sources merged -----------------------
        known = set(self.lm.dict_freq)
        known |= {w.lower() for w in nltk_words.words()}
        known |= {w for w, c in brown_counts.items() if c >= 3}
        self.known = {
            w for w in known
            if w.isalpha() and w.isascii() and (len(w) >= 3 or w in TWO_LETTER_WORDS or w in ("a", "i"))
        }
        self.known |= (ABBREVIATIONS - {"no"})
        self.lm.dict_size = len(self.known)
        self.freq = self.lm.dict_freq

    # ------------------------------------------------------------------ stats for the UI
    def stats(self) -> dict[str, int]:
        return {
            "sentences": self.lm.sentences,
            "vocab": len(self.known),
            "bigrams": len(self.lm.counts[2]),
            "trigrams": len(self.lm.counts[3]),
        }

    # ------------------------------------------------------------------ spelling candidates
    @lru_cache(maxsize=100_000)
    def _candidates(self, word: str) -> tuple[tuple[str, int], ...]:
        e1 = edits1(word)
        cands = {w: 1 for w in e1 if w in self.known}
        # only look at 2-edit candidates when nothing sensible exists at 1 edit
        if not any(self.freq.get(w, 0) > 0 for w in cands) and len(word) >= 4:
            for e in e1:
                for e2 in edits1(e):
                    if e2 in self.known and e2 not in cands:
                        cands[e2] = 2
        cands.pop(word, None)
        return tuple(cands.items())

    # ------------------------------------------------------------------ error (channel) model
    @staticmethod
    def _channel(typed: str, cand: str, d: int) -> float:
        """log P(typed | intended).  A small hand-set model of how people mistype."""
        if d >= 2:
            lp = math.log(0.001)
        elif len(typed) == len(cand):
            diff = [i for i in range(len(typed)) if typed[i] != cand[i]]
            swapped = (len(diff) == 2 and diff[1] == diff[0] + 1
                       and typed[diff[0]] == cand[diff[1]] and typed[diff[1]] == cand[diff[0]])
            lp = math.log(0.5) if swapped else math.log(0.06)      # swapped neighbours vs. wrong letter
        elif cand.endswith("ed") and cand == typed[:-1] + "ed" and typed.endswith("d"):
            lp = math.log(0.25)                                    # jumpd -> jumped (vowel dropped)
        else:
            lp = math.log(0.04)                                    # a letter was dropped / added
            if cand.startswith(typed) or typed.startswith(cand):
                lp -= 1.0                                          # ...at the very END is rarer
        if cand[0] != typed[0]:
            lp -= 0.7                                              # first letters are rarely wrong
        return lp

    # ------------------------------------------------------------------ scoring
    def _window_logp(self, seq: list[str], i: int, cand: str, order: int, lam: float, mu: float) -> float:
        """log-probability of `cand` at position i PLUS the next (order-1) words (right context)."""
        K = self.lm.K
        padded = ["<s>"] * (K - 1) + seq + ["</s>"]
        p = i + K - 1
        padded[p] = cand
        total = 0.0
        for j in range(p, min(p + order, len(padded))):
            hist = padded[max(0, j - (K - 1)): j]
            pr, _ = self.lm.prob(padded[j], hist, order, lam, mu)
            total += math.log(pr)
        return total

    @staticmethod
    def _softmax(pairs: list[tuple[str, float]]) -> list[tuple[str, float]]:
        m = max(s for _, s in pairs)
        exps = [(w, math.exp(s - m)) for w, s in pairs]
        z = sum(e for _, e in exps)
        return sorted(((w, e / z) for w, e in exps), key=lambda x: -x[1])

    # ------------------------------------------------------------------ main entry point
    def analyze(self, text: str, order: int = 3, lam: float = 10.0, mu: float = 20000.0,
                ignore: set[str] | None = None) -> Analysis:
        ignore = {w.lower() for w in (ignore or set())}
        toks = tokenize(text)
        sents = split_sentences(toks)
        issues: list[Issue] = []
        n_words = sum(1 for t in toks if t.kind == "word")
        counter = [0]
        dialogue_tag = False
        for s_no, sent in enumerate(sents, 1):
            issues.extend(self._analyze_sentence(text, toks, sent, s_no, order, lam, mu, ignore, counter,
                                                 dialogue_tag))
            real = [toks[i] for i in sent if toks[i].kind != "space"]
            dialogue_tag = (len(real) >= 2 and real[-1].text in "\"”'’" and real[-2].text in "?!")
        issues.sort(key=lambda i: i.start)
        return Analysis(text, issues, len(sents), n_words)

    # ------------------------------------------------------------------ one sentence
    def _analyze_sentence(self, text, toks, sent, s_no, order, lam, mu, ignore, counter,
                          dialogue_tag=False) -> list[Issue]:
        W = [toks[i] for i in sent if toks[i].kind == "word"]
        n = len(W)
        cur: list[str | None] = [norm(w.text) for w in W]
        found: dict[int, Issue] = {}

        def new_issue(p, kind, repl, msg, method, cands=None, delete=False):
            counter[0] += 1
            w = W[p]
            start = w.start
            if delete:
                a = start
                while a > 0 and text[a - 1] in " \t":
                    a -= 1
                repl_text = ""
            else:
                a = start
                repl_text = apply_case(w.text, repl)
            iss = Issue(f"i{counter[0]}", kind, w.start, w.end, a, w.text, repl_text, s_no, msg, method, cands or [])
            found[p] = iss
            return iss

        def gap(p):  # text between word p-1 and word p
            return text[W[p - 1].end: W[p].start]

        def subject_position(p):
            if p == 0:
                return True
            if any(ch in gap(p) for ch in ',;:(—"“\n'):
                return True
            return cur[p - 1] not in NOT_SUBJECT_CONTEXT

        def live():
            idx = [k for k in range(n) if cur[k] is not None]
            return [cur[k] for k in idx], {k: j for j, k in enumerate(idx)}

        # ============== PASS 1: rule-based checks (agreement, contractions, repeats ...) ==============
        for p in range(n):
            w = cur[p]
            # --- subject-verb agreement
            if w in ("he", "she", "it") and p + 1 < n and subject_position(p) and (p + 1) not in found:
                nx = cur[p + 1]
                fix = None
                if nx in BASE_VERBS:
                    fix = third_person(nx)
                elif nx in ("dont", "don't"):
                    fix = "doesn't"
                elif nx == "are":
                    fix = "is"
                if fix:
                    new_issue(p + 1, "Grammar", fix,
                              f"'{W[p].text}' is third-person singular, so the verb should be '{fix}'.",
                              "Rule-based correction")
                    cur[p + 1] = fix
            elif w in ("i", "we", "you", "they") and p + 1 < n and subject_position(p) and (p + 1) not in found:
                nx = cur[p + 1]
                fix = None
                if nx in THIRD_TO_BASE:
                    fix = THIRD_TO_BASE[nx]
                elif nx in ("doesnt", "doesn't"):
                    fix = "don't"
                elif nx == "is":
                    fix = "am" if w == "i" else "are"
                elif nx == "are" and w == "i":
                    fix = "am"
                elif nx == "was" and w in ("we", "you", "they"):
                    fix = "were"
                if fix:
                    new_issue(p + 1, "Grammar", fix,
                              f"'{W[p].text}' needs the verb form '{fix}'.", "Rule-based correction")
                    cur[p + 1] = fix

        for p in range(n):
            w = cur[p]
            if p in found or w is None:
                continue
            raw_low = norm(W[p].text)
            if raw_low in I_FORMS and W[p].text != I_FORMS[raw_low] and raw_low != "i'm" or \
               (raw_low in ("i'm", "i've", "i'll", "i'd") and W[p].text != I_FORMS[raw_low]):
                new_issue(p, "Grammar", I_FORMS[raw_low], "The pronoun 'I' is always capitalised.",
                          "Rule-based correction")
                cur[p] = I_FORMS[raw_low].lower()
            elif w in CONTRACTION_FIX and w not in ignore:
                fix = CONTRACTION_FIX[w]
                new_issue(p, "Spelling", fix, f"Missing apostrophe: '{W[p].text}' should be '{fix}'.",
                          "Rule-based correction")
                cur[p] = fix.lower()
            elif w == "alot":
                new_issue(p, "Spelling", "a lot", "'alot' is not a word; write 'a lot' as two words.",
                          "Rule-based correction")
                cur[p] = "a lot"
            elif w == "of" and p > 0 and cur[p - 1] in MODALS:
                new_issue(p, "Grammar", "have", f"'{cur[p - 1]} of' is a common mistake; use '{cur[p - 1]} have'.",
                          "Rule-based correction")
                cur[p] = "have"
            elif p > 0 and w == cur[p - 1] and w not in NOT_REPEAT_ERRORS and \
                    gap(p).strip() == "" and "\n" not in gap(p) and (p - 1) not in found:
                new_issue(p, "Grammar", "", f"Repeated word: '{W[p].text}' appears twice in a row.",
                          "Rule-based correction", delete=True)
                cur[p] = None

        # ============== PASS 2: spelling (noisy channel + context) ==============
        def is_unknown(p):
            raw, low = W[p].text, cur[p]
            if low is None or len(raw) < 2 or not raw.isascii() or "'" in low or "’" in raw:
                return False
            if low in ignore or low in self.known or (raw.isupper() and len(raw) > 1):
                return False
            if raw[0].isupper() and p != 0:        # probably a proper noun
                return False
            a, b = W[p].start, W[p].end
            before = text[a - 1] if a > 0 else ""
            after = text[b] if b < len(text) else ""
            after2 = text[b + 1] if b + 1 < len(text) else ""
            if before in "@/_#\\." and before != "" or after in "@/_\\" or (after == "." and after2.isalpha()):
                return False                       # looks like an e-mail / URL / file name
            return True

        def rank_spelling(p):
            low = norm(W[p].text)          # always generate candidates from what the user TYPED
            cands = self._candidates(low)
            if not cands:
                return []
            seq, idx = live()
            i = idx[p]
            scored = []
            for c, d in cands:
                ch = self._channel(low, c, d)
                scored.append((c, LM_WEIGHT * self._window_logp(seq, i, c, order, lam, mu) + ch))
            scored.sort(key=lambda x: -x[1])
            return self._softmax(scored[:5])

        misspelled = [p for p in range(n) if p not in found and is_unknown(p)]
        ranked: dict[int, list] = {}
        for _ in range(2):  # second sweep re-scores with already-repaired neighbours
            for p in misspelled:
                r = rank_spelling(p)
                if r:
                    ranked[p] = r
                    cur[p] = r[0][0]
        for p in misspelled:
            r = ranked.get(p)
            if not r:
                continue
            best = r[0][0]
            raw = W[p].text
            if raw[0].isupper() and p == 0 and raw.lower() == norm(raw) and best == raw.lower():
                continue
            new_issue(p, "Spelling", best,
                      f"'{raw.lower()}' is not in the dictionary; best-ranked candidate is '{best}'.",
                      "Noisy channel + n-gram", cands=r)

        # ============== PASS 3: real-word (context) errors ==============
        for p in range(n):
            w = cur[p]
            if p in found or w is None or w not in CONFUSION_INDEX or w in ignore:
                continue
            seq, idx = live()
            i = idx[p]
            alts = CONFUSION_INDEX[w]
            scored = [(c, self._window_logp(seq, i, c, order, lam, mu)) for c in alts]
            rel = self._softmax(scored)
            best, best_p = rel[0]
            lp = dict(scored)
            # "evidence" = how much the CONTEXT (not just raw word frequency) supports a word.
            uni = {c: math.log(self.lm.prob(c, ["<s>"] * (self.lm.K - 1), 1, lam, mu)[0]) for c in alts}
            evidence_gain = (lp[best] - uni[best]) - (lp[w] - uni[w])
            nxt = cur[p + 1] if p + 1 < n else None
            allowed = True
            if best != w and ("'" in best) != ("'" in w):
                is_ing = nxt is not None and nxt.endswith("ing") and len(nxt) > 4
                if "'" in best:      # moving TO a contraction: next word must look like it follows one
                    allowed = nxt is not None and (nxt in FUNC_AFTER_CONTR or is_ing)
                else:                # moving AWAY from a contraction: next word must NOT look like that
                    allowed = nxt is not None and nxt not in FUNC_AFTER_CONTR and \
                        nxt not in ADJ_AFTER_CONTR and not is_ing
            if allowed and best != w and lp[best] - lp[w] >= 2.9 and evidence_gain >= 2.5 and best_p >= 0.9:
                new_issue(p, "Context", best,
                          f"In this context '{best}' is more probable than '{W[p].text.lower()}'.",
                          "Context n-gram", cands=rel)
                cur[p] = best

        # ============== PASS 4: a / an ==============
        for p in range(n - 1):
            w = cur[p]
            if p in found or w not in ("a", "an") or cur[p + 1] is None:
                continue
            if gap(p + 1).strip() != "" or "\n" in gap(p + 1):
                continue
            nx = cur[p + 1]
            want = article_for(nx, W[p + 1].text)
            if want != w:
                new_issue(p, "Grammar", want,
                          f"Use '{want}' before the {'vowel' if want == 'an' else 'consonant'} sound in '{W[p + 1].text}'.",
                          "Rule-based correction")

        # ============== PASS 5: capital letter at the start of the sentence ==============
        if n and cur[0] is not None:
            first = W[0].text
            if first[0].islower() and not dialogue_tag:
                if 0 in found and found[0].replacement:
                    r = found[0].replacement
                    found[0].replacement = r[0].upper() + r[1:]
                elif 0 not in found:
                    new_issue(0, "Grammar", first[0].upper() + first[1:],
                              "A sentence should start with a capital letter.", "Rule-based correction")
        return list(found.values())


def tokenize_words(s: str) -> list[str]:
    return [t.text for t in tokenize(s) if t.kind == "word"]
