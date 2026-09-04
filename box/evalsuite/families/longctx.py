"""
families/longctx.py - long-context retrieval and reasoning (RULER-style, fully synthetic prompts).

Three sub-families, all assembled AT RUN TIME to a calibrated token budget from public filler text:

  niah_multikey  needle-in-a-haystack: "The secret code for <name> is <7 digits>." hidden in Paul Graham
                 essay prose together with 4 distractor needles of the same form (other names, other codes);
                 needle depth in {0.10, 0.35, 0.60, 0.85} x context {8k, 16k, 32k}; the question names the key.
  var_tracking   chained variable assignments "VAR ABCD = 48213", "VAR EFGH = ABCD", ... (5 hops, forward order,
                 spread over the passage) plus 3 distractor chains with their own values; the question asks the
                 numeric value of the LAST variable of the target chain.
  qa_multi_doc   a SQuAD v1.1 dev question (public URL, no gating) whose gold paragraph is placed at a seeded
                 depth among SQuAD paragraphs from OTHER articles plus up to 8 hard distractors from the SAME
                 article (topically confusable, the answer string is guaranteed absent from them).

Item selection (concentrated, high-discrimination): SQuAD questions are filtered to short (<= 3 words),
annotator-consistent answers that occur exactly once in the gold paragraph and nowhere else in the article,
are not contained in the question, and whose question has LOW lexical overlap with the answer sentence
(paraphrase needed, 0.15 <= overlap <= 0.70); one question per article, numeric answers capped at a third.

Token budget: items carry meta.target_tokens (8192 / 16384 / 32768).  prepare_run() calibrates
chars-per-token against the server's /tokenize (per filler corpus, then verifies one fully rendered 32k item per
sub-family and rescales) so that every prompt fits  min(target, max_model_len - 1536 - max_tokens)  tokens.
--family-opt longctx.fixed=true skips calibration (FIXED_CHARS_PER_TOKEN, assumed max_model_len 32768;
override with longctx.chars_per_token=.. / longctx.max_model_len=..).  Without /tokenize the same fallback is
used and noted.  Filler and documents live in a header record of the item file, so a run needs nothing but
data/items/longctx.jsonl.

Scoring: \\boxed{} -> a line-initial "Answer:"/"answer is" phrase (a phrase buried in trailing prose only as a
fallback) -> last non-empty line; numeric answers compared numerically (digit grouping removed; several distinct
numbers -> the one the candidate attributes to the asked key, else the candidate never committed and is wrong -
crediting the first number would pass "the codes are A, B, C, D, E" whenever the target is listed first),
text answers by SQuAD-style normalisation (case,
digit grouping, punctuation, articles, whitespace) with exact or length-guarded containment match (no containment
for candidates that hedge, negate or refuse, or carry numbers the gold does not have).
"""
from __future__ import annotations

import collections
import html.parser
import json
import os
import re
import string
from typing import Any, Callable, Optional

import common
from common import DEFAULT_SEED, Verdict
from families import _base

NAME = "longctx"
DESCRIPTION = "RULER-style long-context retrieval: multi-key NIAH, variable tracking, multi-document QA (8k/16k/32k)"
SUBFAMILIES = ["niah_multikey", "var_tracking", "qa_multi_doc"]
PRIORITY = 40
DEFAULT_MAX_TOKENS = {"default": 512, "reasoning": 1024}
ITEM_TIME_FALLBACK_S = 45.0
NOTES = [
    "longctx: synthetic RULER-style items; prompts are assembled at run time to a token budget calibrated against "
    "/tokenize (32k items are capped at max_model_len - 1536 - max_tokens), so pair runs only across identical tokenizers",
]

LENGTHS = [8192, 16384, 32768]
LENGTH_LABEL = {8192: "8k", 16384: "16k", 32768: "32k"}
DEPTHS = [0.10, 0.35, 0.60, 0.85]
SAFETY_MARGIN_TOKENS = 1536
FIXED_CHARS_PER_TOKEN = 3.7          # conservative English-prose ratio when /tokenize is unavailable or fixed=true
ASSUMED_MAX_MODEL_LEN = 32768
N_DISTRACTOR_NEEDLES = 4
VT_HOPS = 5
VT_DISTRACTOR_CHAINS = 3
QA_HARD_DOCS = 8
QA_POOL_SIZE = 420

PG_BASE = "https://paulgraham.com/"
PG_ESSAYS = ["greatwork", "worked", "superlinear", "wealth", "hs", "start", "ds", "growth", "love", "essay", "taste",
             "avg", "hp", "say", "gh", "before", "think", "cities", "ambitious", "heresy", "genius", "lesson",
             "conformism", "useful", "richnow", "users", "good", "ineq", "re", "own", "hwh"]
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"

ADJECTIVES = ["crimson", "amber", "silver", "velvet", "hollow", "quiet", "golden", "frozen", "wandering", "copper",
              "midnight", "scarlet", "ivory", "emerald", "rusty", "gentle", "distant", "marble", "violet", "cobalt",
              "sleepy", "ancient", "humble", "jade", "obsidian", "paper", "sapphire", "wooden", "misty", "saffron"]
NOUNS = ["lantern", "violin", "compass", "harbor", "kettle", "orchard", "ladder", "meadow", "anchor", "quill",
         "saddle", "canyon", "beacon", "thimble", "walnut", "glacier", "pebble", "falcon", "teapot", "bridge",
         "mirror", "lighthouse", "garden", "hammock", "tunnel", "windmill", "sparrow", "chisel", "trumpet", "cellar"]
VAR_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"     # no I / O (digit look-alikes)

# ---------------------------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------------------------

NIAH_INTRO = ('Below is a long passage of text. Hidden inside it are a few special sentences of the form '
              '"The secret code for <name> is <number>." Read the whole passage carefully, then answer the question '
              'at the end.\n\n<passage>\n')
VT_INTRO = ('Below is a long passage of text. Scattered through it are variable assignment statements of the form '
            '"VAR ABCD = 12345" (a numeric value) or "VAR ABCD = EFGH" (ABCD takes the value of EFGH). Read the whole '
            'passage carefully, then answer the question at the end.\n\n<passage>\n')
QA_INTRO = ('Below are several numbered documents. Read them carefully, then answer the question at the end using '
            'only information from the documents.\n\n')
ANSWER_LINE = 'End your response with a single line of the form "Answer: <{what}>".'


def _niah_question(key: str) -> str:
    return f"What is the secret code for {key}?"


def _vt_question(var: str) -> str:
    return f"What is the numeric value of VAR {var}?"


# ---------------------------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------------------------

class _PGText(html.parser.HTMLParser):
    """Text of the essay body (<font face="verdana">) of a paulgraham.com page, <br>/<p> -> newlines."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title: list[str] = []
        self.in_title = False
        self.in_body = False
        self.done = False
        self.depth = 0
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "title":
            self.in_title = True
        if tag in ("script", "style"):
            self.skip += 1
        if tag == "font" and not self.done:
            if self.in_body:
                self.depth += 1
            elif (a.get("face") or "").lower() == "verdana":
                self.in_body, self.depth = True, 1
        if self.in_body and tag in ("br", "p", "tr", "div"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        if tag in ("script", "style") and self.skip:
            self.skip -= 1
        if tag == "font" and self.in_body:
            self.depth -= 1
            if self.depth == 0:
                self.in_body, self.done = False, True

    def handle_data(self, data):
        if self.in_title:
            self.title.append(data)
        elif self.in_body and not self.skip:
            self.parts.append(data)


def _pg_text(raw: bytes) -> tuple[str, str]:
    try:
        page = raw.decode("utf-8")
    except UnicodeDecodeError:
        page = raw.decode("cp1252", "replace")
    p = _PGText()
    p.feed(page)
    text = "".join(p.parts)
    paras = [re.sub(r"\s+", " ", para).strip() for para in re.split(r"\n\s*\n", text)]
    paras = [q for q in paras if len(q) >= 40]                      # drop dates, single-word lines, image alts
    return " ".join(p.title).strip(), "\n\n".join(paras)


_PUNCT = set(string.punctuation) | set("“”‘’«»–—…")
_ARTICLES = re.compile(r"\b(a|an|the)\b")


_DIGIT_GROUP = re.compile(r"(?<=\d)[,\u202f\u00a0](?=\d{3}\b)")
_SPACE_GROUPED = re.compile(r"\s*-?\d{1,3}(?: \d{3})+\s*")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def norm_text(s: str) -> str:
    """SQuAD-style answer normalisation: lower, drop digit grouping, punctuation and articles, collapse whitespace
    ('45,000 pounds' == '45000 pounds')."""
    s = _DIGIT_GROUP.sub("", (s or "").lower())
    s = "".join(" " if ch in _PUNCT else ch for ch in s)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _numbers(s: str) -> list[str]:
    """Distinct numbers in order of appearance, digit grouping (4,831,920 / LaTeX 4{,}831{,}920 / 4\\,831\\,920 /
    a candidate that is exactly one space-grouped number '4 831 920') removed."""
    s = (s or "").replace("{,}", ",").replace("\\,", ",").replace("\\;", ",")
    if _SPACE_GROUPED.fullmatch(s):
        s = s.replace(" ", "")
    out: list[str] = []
    for n in _NUMBER.findall(_DIGIT_GROUP.sub("", s)):
        if n not in out:
            out.append(n)
    return out


def _num_equal(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return False


_STOP = set("the a an of in on at to for from by with and or is are was were be been being what which who whom whose "
            "when where why how did do does that this these those it its as into than then there their they he she his "
            "her him many much more most some any one two first name named called".split())


def _content_words(s: str) -> set[str]:
    return {w for w in norm_text(s).split() if w not in _STOP and len(w) > 1}


def _sentence_around(text: str, pos: int) -> str:
    start = max(text.rfind(". ", 0, pos), text.rfind("\n", 0, pos))
    end_candidates = [i for i in (text.find(". ", pos), text.find("\n", pos)) if i != -1]
    end = min(end_candidates) if end_candidates else len(text)
    return text[start + 1:end + 1]


def _word_boundary_find(hay_lower: str, needle_lower: str) -> int:
    """Number of word-boundary occurrences (needle already lower-cased)."""
    return len(re.findall(r"(?<![A-Za-z0-9])" + re.escape(needle_lower) + r"(?![A-Za-z0-9])", hay_lower))


_ANSWER_STOP = {"of", "the", "a", "an", "and", "de", "for", "in", "on"}
_DESCRIPTOR_OF = re.compile(r"^(?:the\s+)?[A-Z][a-z]+\s+of\s+\S", re.I)
_QUALIFIER = re.compile(r"^(?:at\s+least|at\s+most|about|over|under|more\s+than|less\s+than|approximately|nearly|around|"
                        r"up\s+to|roughly|almost|some|only|just|early|late|mid)\s+\S", re.I)


def _ambiguous_span(answer: str) -> bool:
    """True when a shorter span of the gold is itself a complete answer a terse model would give, so exact/
    containment scoring would wrongly reject it: 'the journal Nature' (descriptor + proper noun -> 'Nature'),
    'City of Malindi' / 'University of X' (capitalised descriptor + of -> 'Malindi'), 'at least 90%' / 'early 1960s'
    (qualifier + value -> '90%').  Such golds are skipped at selection time."""
    a = re.sub(r"^(?:the|a|an)\s+", "", answer.strip(), flags=re.I)
    words = [w for w in a.split() if w.lower() not in _ANSWER_STOP]
    caps = [w for w in words if w[:1].isupper()]
    lows = [w for w in words if w[:1].islower()]
    if caps and lows:
        return True
    return bool(_DESCRIPTOR_OF.match(answer.strip()) or _QUALIFIER.match(a))


def _stratified_order(groups: list[list[dict]]) -> None:
    """Assign `order` so that context lengths alternate (nested --limit subsets cover every length)."""
    for i, it in enumerate(_base.interleave(groups)):
        it["order"] = i


# ---------------------------------------------------------------------------------------------
# prepare(): fetch sources, build items + corpus header
# ---------------------------------------------------------------------------------------------

def _fetch_filler(data_dir: str, refresh: bool, log: Callable[[str], None]) -> tuple[list[dict], dict]:
    essays: list[dict] = []
    sources: dict = {}
    for slug in PG_ESSAYS:
        url = f"{PG_BASE}{slug}.html"
        dest = os.path.join(data_dir, "raw", "paulgraham", f"{slug}.html")
        r = common.fetch(url, dest, refresh=refresh, log=log)
        with open(dest, "rb") as f:
            title, text = _pg_text(f.read())
        if len(text) < 5000:
            raise IOError(f"paulgraham.com/{slug}.html: extracted only {len(text)} chars of essay text")
        essays.append({"slug": slug, "title": title, "url": url, "chars": len(text), "text": text})
        # the page bytes carry a server timestamp comment (sha256 differs per download); the extracted
        # essay text is what the items depend on, so its digest is recorded too
        sources[f"paulgraham:{slug}"] = {"url": url, "sha256": r["sha256"], "bytes": r["bytes"],
                                         "text_sha256": common.sha256_bytes(text.encode("utf-8")), "title": title}
    return essays, sources


def _build_niah(rng, n_per_cell: int) -> list[dict]:
    keys = [f"the {a} {n}" for a in ADJECTIVES for n in NOUNS]
    rng.shuffle(keys)
    values: set[int] = set()

    def fresh_value() -> int:
        while True:
            v = rng.randint(1000000, 9999999)
            if v not in values and len(set(str(v))) >= 4:
                values.add(v)
                return v

    items: list[dict] = []
    k = 0
    key_i = 0
    for rep in range(n_per_cell):
        for depth in DEPTHS:
            for L in LENGTHS:
                target_key, key_i = keys[key_i], key_i + 1
                target_val = fresh_value()
                distractors = []
                used_depths = [depth]
                for _ in range(N_DISTRACTOR_NEEDLES):
                    dk, key_i = keys[key_i], key_i + 1
                    while True:
                        dd = round(rng.uniform(0.03, 0.97), 3)
                        if all(abs(dd - u) >= 0.05 for u in used_depths):
                            break
                    used_depths.append(dd)
                    distractors.append({"key": dk, "value": str(fresh_value()), "depth": dd})
                lab = LENGTH_LABEL[L]
                items.append({
                    "id": f"longctx-niah-{lab}-d{int(round(depth * 100)):02d}-{k:02d}", "family": NAME,
                    "subfamily": "niah_multikey", "question": _niah_question(target_key),
                    "mock_marker": _niah_question(target_key), "answer": str(target_val),
                    "needle": {"key": target_key, "value": str(target_val), "depth": depth},
                    "distractors": distractors, "filler_start": round(rng.random(), 6),
                    "meta": {"source": "synthetic+paulgraham", "target_tokens": L, "depth": depth,
                             "n_distractors": N_DISTRACTOR_NEEDLES},
                })
                k += 1
    return items


def _build_vt(rng, n_per_len: int) -> list[dict]:
    names_used: set[str] = set()

    def fresh_name() -> str:
        while True:
            nm = "".join(rng.choice(VAR_LETTERS) for _ in range(4))
            if nm not in names_used and len(set(nm)) >= 3:
                names_used.add(nm)
                return nm

    items: list[dict] = []
    k = 0
    for rep in range(n_per_len):
        for L in LENGTHS:
            values: set[int] = set()

            def fresh_value() -> int:
                while True:
                    v = rng.randint(10000, 99999)
                    if v not in values and all(abs(v - u) > 100 for u in values):
                        values.add(v)
                        return v

            target = [fresh_name() for _ in range(VT_HOPS + 1)]
            tval = fresh_value()
            stmts = [{"text": f"VAR {target[0]} = {tval}", "depth": None, "chain": 0}]
            for i in range(1, len(target)):
                stmts.append({"text": f"VAR {target[i]} = {target[i - 1]}", "depth": None, "chain": 0})
            # target chain in forward order, evenly spread with jitter
            n = len(stmts)
            for i, st in enumerate(stmts):
                base = 0.08 + (0.84 * i / (n - 1))
                st["depth"] = round(min(0.97, max(0.03, base + rng.uniform(-0.03, 0.03))), 3)
            used = [st["depth"] for st in stmts]
            for c in range(1, VT_DISTRACTOR_CHAINS + 1):
                clen = rng.randint(3, 5)
                names = [fresh_name() for _ in range(clen)]
                dval = fresh_value()
                texts = [f"VAR {names[0]} = {dval}"] + [f"VAR {names[i]} = {names[i - 1]}" for i in range(1, clen)]
                for t in texts:
                    while True:
                        dd = round(rng.uniform(0.03, 0.97), 3)
                        if all(abs(dd - u) >= 0.012 for u in used):
                            break
                    used.append(dd)
                    stmts.append({"text": t, "depth": dd, "chain": c})
            lab = LENGTH_LABEL[L]
            items.append({
                "id": f"longctx-vt-{lab}-{k:02d}", "family": NAME, "subfamily": "var_tracking",
                "question": _vt_question(target[-1]), "mock_marker": _vt_question(target[-1]), "answer": str(tval),
                "statements": stmts, "target_chain": target, "filler_start": round(rng.random(), 6),
                "meta": {"source": "synthetic+paulgraham", "target_tokens": L, "hops": VT_HOPS,
                         "n_distractor_chains": VT_DISTRACTOR_CHAINS, "depth": stmts[VT_HOPS]["depth"]},
            })
            k += 1
    return items


def _squad_candidates(data: dict) -> tuple[list[dict], dict[str, dict], dict[str, list[str]]]:
    """(candidate questions, paragraphs by pid, pids by article)."""
    paras: dict[str, dict] = {}
    by_article: dict[str, list[str]] = collections.OrderedDict()
    cands: list[dict] = []
    for ai, art in enumerate(data["data"]):
        title = art["title"].replace("_", " ")
        pids = []
        for pi, p in enumerate(art["paragraphs"]):
            pid = f"{ai}-{pi}"
            paras[pid] = {"title": title, "text": " ".join(p["context"].split()), "article": ai}
            pids.append(pid)
        by_article[title] = pids
        lows = {pid: paras[pid]["text"].lower() for pid in pids}
        for pi, p in enumerate(art["paragraphs"]):
            ctx = paras[f"{ai}-{pi}"]["text"]
            ctx_low = lows[f"{ai}-{pi}"]
            for qa in p["qas"]:
                q = " ".join(qa["question"].split())
                answers = [" ".join(a["text"].split()) for a in qa["answers"] if a["text"].strip()]
                if not answers or len(q.split()) < 6 or len(q) < 30:   # >= 30 chars: a safe mock marker
                    continue
                normed = collections.Counter(norm_text(a) for a in answers)
                if len(normed) > 2 or any(len(n.split()) > 4 or not n for n in normed):
                    continue
                top_norm = normed.most_common(1)[0][0]
                primary = min((a for a in answers if norm_text(a) == top_norm), key=len)
                if not (3 <= len(primary) <= 40) or len(primary.split()) > 3:
                    continue
                if top_norm in norm_text(q):
                    continue
                if re.fullmatch(r"\d{1,2}", primary):
                    continue
                if _ambiguous_span(primary):
                    continue
                if _word_boundary_find(ctx_low, primary.lower()) != 1:
                    continue
                plow = primary.lower()
                if any(plow in lows[o] and _word_boundary_find(lows[o], plow) for o in pids if o != f"{ai}-{pi}"):
                    continue
                pos = ctx_low.find(primary.lower())
                sent = _sentence_around(ctx, pos)
                qw = _content_words(q)
                if not qw:
                    continue
                overlap = len(qw & _content_words(sent)) / len(qw)
                if not (0.15 <= overlap <= 0.70):
                    continue
                cands.append({"qid": qa["id"], "question": q, "answers": sorted(set(answers), key=answers.index),
                              "answer": primary, "pid": f"{ai}-{pi}", "article": title, "overlap": round(overlap, 3),
                              "numeric": bool(re.search(r"\d", primary))})   # any digit: '29.7%', '45,000 pounds', '1883'
    return cands, paras, by_article


def _build_qa(rng, n_per_len: int, data: dict, allow_short: bool, log: Callable[[str], None]) -> tuple[list[dict], dict, list[str], int]:
    cands, paras, by_article = _squad_candidates(data)
    per_article: dict[str, list[dict]] = collections.defaultdict(list)
    for c in cands:
        per_article[c["article"]].append(c)
    articles = list(by_article)
    rng.shuffle(articles)
    need = n_per_len * len(LENGTHS)
    chosen: list[dict] = []
    n_numeric = 0
    for title in articles:
        cs = per_article.get(title) or []
        if not cs:
            continue
        rng.shuffle(cs)
        cs.sort(key=lambda c: c["overlap"])              # lowest lexical overlap first (paraphrase needed)
        pick = None
        for c in cs:
            if c["numeric"] and n_numeric >= need // 3:
                continue
            pick = c
            break
        if pick is None:
            continue
        n_numeric += int(pick["numeric"])
        chosen.append(pick)
        if len(chosen) >= need:
            break
    if len(chosen) < need:
        if not allow_short:
            raise common.ShortPool(f"qa_multi_doc: {len(chosen)} eligible SQuAD questions < {need}")
        log(f"   longctx/qa_multi_doc: short pool {len(chosen)} < {need}")
    chosen = chosen[:need]
    gold_articles = {c["article"] for c in chosen}
    answers_low = [a.lower() for c in chosen for a in c["answers"]]
    pool_all = [pid for t, pids in by_article.items() if t not in gold_articles for pid in pids]
    pool_all = [pid for pid in pool_all if not any(_word_boundary_find(paras[pid]["text"].lower(), a) for a in answers_low)]
    rng.shuffle(pool_all)
    pool = pool_all[:QA_POOL_SIZE]
    docs: dict[str, dict] = {pid: {"title": paras[pid]["title"], "text": paras[pid]["text"]} for pid in pool}

    items: list[dict] = []
    k = 0
    ci = 0
    for rep in range(n_per_len):
        for L in LENGTHS:
            if ci >= len(chosen):
                break
            c = chosen[ci]
            ci += 1
            depth = DEPTHS[(rep * len(LENGTHS) + LENGTHS.index(L)) % len(DEPTHS)]
            same = [pid for pid in by_article[c["article"]] if pid != c["pid"]]
            rng.shuffle(same)
            hard = sorted(same[:QA_HARD_DOCS])
            for pid in hard + [c["pid"]]:
                docs[pid] = {"title": paras[pid]["title"], "text": paras[pid]["text"]}
            lab = LENGTH_LABEL[L]
            items.append({
                "id": f"longctx-qa-{lab}-d{int(round(depth * 100)):02d}-{k:02d}", "family": NAME,
                "subfamily": "qa_multi_doc", "question": c["question"], "mock_marker": c["question"],
                "answer": c["answer"], "answers": c["answers"], "gold_doc": c["pid"], "hard_docs": hard,
                "pool_offset": rng.randrange(len(pool)), "depth": depth,
                "meta": {"source": "squad-v1.1-dev", "source_id": c["qid"], "article": c["article"],
                         "target_tokens": L, "depth": depth, "overlap": c["overlap"], "n_hard_docs": len(hard)},
            })
            k += 1
    return items, docs, pool, len(cands)


def prepare(data_dir: str, seed: int = DEFAULT_SEED, profile: str = "default", refresh: bool = False,
            log: Callable[[str], None] = print, allow_short: bool = False, n_niah_per_cell: Optional[int] = None,
            n_vt_per_len: Optional[int] = None, n_qa_per_len: Optional[int] = None, **opts) -> dict:
    full = profile == "full"
    n_niah_per_cell = int(n_niah_per_cell) if n_niah_per_cell is not None else (4 if full else 2)
    n_vt_per_len = int(n_vt_per_len) if n_vt_per_len is not None else (8 if full else 4)
    n_qa_per_len = int(n_qa_per_len) if n_qa_per_len is not None else (8 if full else 4)

    essays, sources = _fetch_filler(data_dir, refresh, log)
    filler_text = "\n\n".join(e["text"] for e in essays)
    log(f"   longctx: filler corpus {len(essays)} essays, {len(filler_text)} chars")

    squad_dest = os.path.join(data_dir, "raw", "squad", "dev-v1.1.json")
    r = common.fetch(SQUAD_URL, squad_dest, refresh=refresh, log=log)
    sources["squad-v1.1-dev"] = {"url": SQUAD_URL, "sha256": r["sha256"], "bytes": r["bytes"]}
    with open(squad_dest, "r", encoding="utf-8") as f:
        squad = json.load(f)

    niah = _build_niah(common.seeded_rng(seed, NAME, "niah_multikey"), n_niah_per_cell)
    vt = _build_vt(common.seeded_rng(seed, NAME, "var_tracking"), n_vt_per_len)
    qa, docs, pool, qa_pool = _build_qa(common.seeded_rng(seed, NAME, "qa_multi_doc"), n_qa_per_len, squad, allow_short, log)

    for group in (niah, vt, qa):
        by_len: dict[int, list[dict]] = collections.OrderedDict((L, []) for L in LENGTHS)
        for it in group:
            by_len[it["meta"]["target_tokens"]].append(it)
        rng = common.seeded_rng(seed, NAME, group[0]["subfamily"] if group else "empty", "order")
        buckets = []
        for L in LENGTHS:
            b = list(by_len[L])
            rng.shuffle(b)
            buckets.append(b)
        _stratified_order(buckets)

    header = {"_corpus": {"filler": {"text": filler_text,
                                     "essays": [{k: e[k] for k in ("slug", "title", "url", "chars")} for e in essays]},
                          "docs": docs, "pool": pool}}
    rows = [header] + niah + vt + qa
    common.write_jsonl(_base.items_path(NAME, data_dir), rows)
    counts = {"niah_multikey": len(niah), "var_tracking": len(vt), "qa_multi_doc": len(qa)}
    return {"file": f"items/{NAME}.jsonl", "counts": counts,
            "pools": {"niah_multikey": "synthetic", "var_tracking": "synthetic", "qa_multi_doc": qa_pool},
            "sources": sources, "notes": NOTES + [f"longctx: filler corpus {len(filler_text)} chars from {len(essays)} "
                                                  f"paulgraham.com essays; {len(docs)} SQuAD paragraphs in the item header"]}


# ---------------------------------------------------------------------------------------------
# run-time prompt assembly
# ---------------------------------------------------------------------------------------------

_CORPUS_CACHE: dict[str, dict] = {}


def _corpus(data_dir: Optional[str]) -> dict:
    key = os.path.abspath(data_dir or common.DEFAULT_DATA_DIR)
    if key not in _CORPUS_CACHE:
        hdr = _base.read_headers(NAME, key)
        corpus = hdr.get("_corpus")
        if not corpus:
            raise RuntimeError(f"{_base.items_path(NAME, key)} has no _corpus header - rebuild with prepare_data.py --only longctx")
        _CORPUS_CACHE[key] = corpus
    return _CORPUS_CACHE[key]


class Calibration:
    """chars-per-token per filler corpus + template overhead, from prepare_run() (or the fixed fallback)."""

    def __init__(self, max_model_len: int, max_tokens: int, cpt: float, method: str):
        self.max_model_len = int(max_model_len)
        self.max_tokens = int(max_tokens)
        self.cpt = {"filler": float(cpt), "docs": float(cpt)}
        self.fixed_tokens: dict[str, int] = {}
        self.method = method
        self.tokenize_calls = 0
        self.checks: dict[str, dict] = {}

    def budget(self, target: int) -> int:
        return max(1024, min(int(target), self.max_model_len - SAFETY_MARGIN_TOKENS - self.max_tokens))

    def to_dict(self) -> dict:
        return {"method": self.method, "max_model_len": self.max_model_len, "max_tokens": self.max_tokens,
                "chars_per_token": {k: round(v, 4) for k, v in self.cpt.items()}, "fixed_tokens": self.fixed_tokens,
                "budgets": {LENGTH_LABEL[L]: self.budget(L) for L in LENGTHS}, "tokenize_calls": self.tokenize_calls,
                "checks": self.checks}


_LAST_CAL: Optional[Calibration] = None


def _truthy(v: Any) -> bool:
    return v is True or (isinstance(v, (int, float)) and v != 0) or (isinstance(v, str) and v.strip().lower() in ("1", "true", "yes", "on"))


def _default_calibration(ctx: Any) -> Calibration:
    mml = None
    if ctx is not None:
        mml = ctx.opt("max_model_len") if hasattr(ctx, "opt") else None
        if not mml and getattr(ctx, "client", None) is not None:
            mml = getattr(ctx.client, "max_model_len", None)
    cpt = float(ctx.opt("chars_per_token", FIXED_CHARS_PER_TOKEN)) if (ctx is not None and hasattr(ctx, "opt")) else FIXED_CHARS_PER_TOKEN
    return Calibration(int(mml or ASSUMED_MAX_MODEL_LEN), int(getattr(ctx, "max_tokens", DEFAULT_MAX_TOKENS["default"])), cpt, "fixed")


def _calibration(ctx: Any) -> Calibration:
    cal = getattr(ctx, "_longctx_cal", None) if ctx is not None else None
    return cal if isinstance(cal, Calibration) else _default_calibration(ctx)


def _filler_window(text: str, start_frac: float, n_chars: int) -> str:
    """n_chars of filler starting at a word boundary near start_frac (wrapping around the corpus)."""
    n = len(text)
    n_chars = max(0, min(n_chars, n - 1))
    start = int(start_frac * n) % n
    ws = text.find(" ", start)
    if ws != -1 and ws - start < 200:
        start = ws + 1
    if start + n_chars <= n:
        out = text[start:start + n_chars]
    else:
        out = text[start:] + "\n\n" + text[: n_chars - (n - start)]
    cut = out.rfind(" ")
    return out[:cut].strip() if cut > n_chars * 0.9 else out.strip()


def _insert_sentences(filler: str, inserts: list[tuple[float, str]]) -> str:
    """Insert sentences at depth fractions, each at the next sentence boundary (or word boundary)."""
    pos_text: list[tuple[int, str]] = []
    n = len(filler)
    for depth, sent in inserts:
        p = int(depth * n)
        m = re.compile(r"[.!?]\s").search(filler, p, min(n, p + 600))
        if m:
            at = m.end()
        else:
            ws = filler.find(" ", p)
            at = (ws + 1) if ws != -1 else n
        pos_text.append((at, sent))
    pos_text.sort(key=lambda t: t[0], reverse=True)
    out = filler
    for at, sent in pos_text:
        sep_before = "" if at == 0 or out[at - 1].isspace() else " "
        sep_after = "" if at < len(out) and out[at].isspace() else " "
        out = out[:at] + sep_before + sent + sep_after + out[at:]
    return out


def _est_tokens(s: str, cpt: float) -> int:
    return int(len(s) / cpt) + 1


def _nv_inserts(item: dict) -> list[tuple[float, str]]:
    if item["subfamily"] == "niah_multikey":
        inserts = [(item["needle"]["depth"], f"The secret code for {item['needle']['key']} is {item['needle']['value']}.")]
        inserts += [(d["depth"], f"The secret code for {d['key']} is {d['value']}.") for d in item["distractors"]]
        return inserts
    return [(s["depth"], s["text"] + ".") for s in item["statements"]]


def _nv_tail(item: dict) -> str:
    return f"\n</passage>\n\nQuestion: {item['question']}\n{ANSWER_LINE.format(what='number')}"


def _qa_tail(item: dict) -> str:
    return f"\n\nQuestion: {item['question']}\n{ANSWER_LINE.format(what='short phrase')}"


def _render_niah_vt(item: dict, corpus: dict, cal: Calibration, filler_chars: Optional[int] = None) -> str:
    sub = item["subfamily"]
    intro = NIAH_INTRO if sub == "niah_multikey" else VT_INTRO
    inserts = _nv_inserts(item)
    tail = _nv_tail(item)
    if filler_chars is None:
        budget = cal.budget(item["meta"]["target_tokens"])
        fixed = cal.fixed_tokens.get(sub) or _est_tokens(intro + tail, cal.cpt["filler"]) + 24
        needle_tokens = sum(_est_tokens(s, cal.cpt["filler"]) for _, s in inserts)
        filler_chars = int(max(0, budget - fixed - needle_tokens) * cal.cpt["filler"])
    filler = _filler_window(corpus["filler"]["text"], item["filler_start"], filler_chars)
    return intro + _insert_sentences(filler, inserts) + tail


def _render_qa(item: dict, corpus: dict, cal: Calibration, doc_chars: Optional[int] = None) -> str:
    docs, pool = corpus["docs"], corpus["pool"]
    tail = _qa_tail(item)
    cpt = cal.cpt["docs"]
    if doc_chars is None:
        budget = cal.budget(item["meta"]["target_tokens"])
        fixed = cal.fixed_tokens.get("qa_multi_doc") or _est_tokens(QA_INTRO + tail, cpt) + 24
        doc_chars = int(max(0, budget - fixed) * cpt)

    def doc_len(pid: str) -> int:
        return len(docs[pid]["text"]) + len(docs[pid]["title"]) + 24   # header line + blank lines

    gold, hard = item["gold_doc"], list(item["hard_docs"])
    used = doc_len(gold) + sum(doc_len(p) for p in hard)
    filler_ids: list[str] = []
    off = item["pool_offset"] % max(1, len(pool))
    for i in range(len(pool)):
        pid = pool[(off + i) % len(pool)]
        if pid == gold or pid in hard:
            continue
        if used + doc_len(pid) > doc_chars:
            break
        filler_ids.append(pid)
        used += doc_len(pid)
    # layout: hard distractors at seeded slots, gold at its depth, filler in pool order (independent of the run seed)
    n_docs = 1 + len(hard) + len(filler_ids)
    rng = common.seeded_rng(0, NAME, item["id"], "layout")
    slots = list(range(n_docs))
    gold_slot = min(n_docs - 1, int(round(item["depth"] * (n_docs - 1))))
    slots.remove(gold_slot)
    hard_slots = sorted(rng.sample(slots, len(hard)))
    order: list[Optional[str]] = [None] * n_docs
    order[gold_slot] = gold
    for s, pid in zip(hard_slots, hard):
        order[s] = pid
    fi = iter(filler_ids)
    for i in range(n_docs):
        if order[i] is None:
            order[i] = next(fi)
    parts = [QA_INTRO]
    for i, pid in enumerate(order, 1):
        parts.append(f"Document {i} [Title: {docs[pid]['title']}]\n{docs[pid]['text']}\n\n")
    return "".join(parts).rstrip("\n") + tail


def render_prompt(item: dict, ctx: Any) -> str:
    corpus = _corpus(getattr(ctx, "data_dir", None))
    cal = _calibration(ctx)
    if item["subfamily"] == "qa_multi_doc":
        return _render_qa(item, corpus, cal)
    return _render_niah_vt(item, corpus, cal)


def build_messages(item: dict, ctx: Any) -> list[dict]:
    return [{"role": "user", "content": render_prompt(item, ctx)}]


# ---------------------------------------------------------------------------------------------
# prepare_run(): /tokenize calibration
# ---------------------------------------------------------------------------------------------

async def prepare_run(items: list[dict], ctx: Any) -> None:
    global _LAST_CAL
    log = getattr(ctx, "log", print)
    client = getattr(ctx, "client", None)
    if _truthy(ctx.opt("fixed")) or client is None:
        cal = _default_calibration(ctx)
        cal.method = "fixed (longctx.fixed=true)" if client is not None else "fixed (no client)"
        ctx._longctx_cal = cal
        _LAST_CAL = cal
        log(f"longctx: calibration skipped -> {json.dumps(cal.to_dict(), sort_keys=True)}")
        return
    corpus = _corpus(ctx.data_dir)
    cal = _default_calibration(ctx)
    cal.method = "tokenize"

    async def count(**kw) -> Optional[int]:
        r = await client.tokenize(**kw)
        cal.tokenize_calls += 1
        if not r or r.get("count") is None:
            return None
        return int(r["count"])

    # 1. characters per token on each corpus (three 6k-char windows each)
    filler = corpus["filler"]["text"]
    docs_text = "\n\n".join(corpus["docs"][p]["text"] for p in corpus["pool"][:120])
    ok = True
    for name, text in (("filler", filler), ("docs", docs_text)):
        chars = tokens = 0
        for frac in (0.15, 0.5, 0.85):
            chunk = _filler_window(text, frac, 6000)
            c = await count(prompt=chunk)
            if c is None or c <= 0:
                ok = False
                break
            chars += len(chunk)
            tokens += c
        if not ok:
            break
        cal.cpt[name] = chars / tokens
    if not ok:
        cal = _default_calibration(ctx)
        cal.method = "fixed (/tokenize unavailable)"
        ctx._longctx_cal = cal
        _LAST_CAL = cal
        log(f"longctx: /tokenize unavailable, fixed fallback -> {json.dumps(cal.to_dict(), sort_keys=True)}")
        return
    if client.max_model_len:
        cal.max_model_len = int(client.max_model_len)

    # 2. template overhead per sub-family (chat template included) with zero filler
    reps: dict[str, dict] = {}
    for it in items:
        sub = it["subfamily"]
        if sub not in reps or it["meta"]["target_tokens"] > reps[sub]["meta"]["target_tokens"]:
            reps[sub] = it
    for sub, it in reps.items():
        # the zero-filler render still contains the gold + hard docs (qa) or the needles (niah/vt); those are
        # accounted for separately by the renderers, so subtract their estimate here
        if sub == "qa_multi_doc":
            text = _render_qa(it, corpus, cal, doc_chars=0)
            extra = (len(text) - len(QA_INTRO) - len(_qa_tail(it))) / cal.cpt["docs"]
        else:
            text = _render_niah_vt(it, corpus, cal, filler_chars=0)
            extra = sum(len(s) + 1 for _, s in _nv_inserts(it)) / cal.cpt["filler"]
        c = await count(messages=[{"role": "user", "content": text}])
        if c is not None:
            cal.fixed_tokens[sub] = max(16, int(c - extra) + 8)

    # 3. verify one fully rendered largest item per sub-family and rescale the ratio until it fits
    for sub, it in reps.items():
        corpus_key = "docs" if sub == "qa_multi_doc" else "filler"
        budget = cal.budget(it["meta"]["target_tokens"])
        last = None
        for _round in range(5):
            msgs = build_messages(it, _CtxView(ctx, cal))
            c = await count(messages=msgs)
            if c is None:
                break
            last = c
            if c <= budget and c >= 0.94 * budget:
                break
            cal.cpt[corpus_key] *= (0.985 * budget) / c
        if last is not None and last > budget:     # still over after the loop: shrink hard
            cal.cpt[corpus_key] *= 0.97
            msgs = build_messages(it, _CtxView(ctx, cal))
            c = await count(messages=msgs)
            last = c if c is not None else last
        cal.checks[sub] = {"item": it["id"], "budget": budget, "prompt_tokens": last}
    ctx._longctx_cal = cal
    _LAST_CAL = cal
    log(f"longctx: calibrated -> {json.dumps(cal.to_dict(), sort_keys=True)}")


class _CtxView:
    """ctx proxy carrying a provisional calibration (used while calibrating)."""

    def __init__(self, ctx: Any, cal: Calibration):
        self._ctx = ctx
        self._longctx_cal = cal

    def __getattr__(self, name):
        return getattr(self._ctx, name)


# ---------------------------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------------------------

_TAG_LEFTOVER = re.compile(r"</?\s*(?:think|thinking|reason|reasoning|thought|answer)\s*>", re.I)
_FENCE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*$", re.M)
_LEAD_ANSWER = re.compile(r"^\s*(?:final\s+answer|answer|code|value)\s*(?:is\b|:|=)\s*", re.I)
# like common._ANSWER_PHRASE but 'is' must be a whole word: "the answer isn't stated" is not an answer phrase
_ANSWER_PHRASE = re.compile(r"(?i)\b(?:final\s+answer|answer)\s*(?:is\b|:|=)\s*")
# the prompt asks for a final line 'Answer: <x>'; a match that STARTS a line (after markdown decoration)
# is the committed answer, so trailing prose ("...that answer is supported by Document 3") cannot hijack it
_ANSWER_LINE = re.compile(r"(?im)^[\s>*_#`\-]*(?:final\s+answer|answer)\s*(?:is\b|:|=)\s*")
_CITATION = re.compile(r"\s*[\[(](?:see\s+|from\s+|in\s+)?(?:document|doc|paragraph|passage|source|section)s?\b[^()\[\]]*[\])]\s*$", re.I)
# hedges / refusals / negations: a candidate carrying one of these is never accepted by containment or by the
# first-of-several-numbers rule (the model did not commit to one answer)
_HEDGE_NUM = re.compile(r"(?i)(?<![A-Za-z])(?:or|either|cannot|can't|could\s*not|couldn't|unable|not\s+(?:sure|certain|find|found|"
                        r"determine|possible|clear|stated|mentioned)|unclear|unsure|unknown|don't\s+know|do\s+not\s+know|"
                        r"no\s+(?:idea|answer|information))(?![A-Za-z])|/")
_HEDGE_TEXT = re.compile(r"(?<!\S)(?:or|either|and|vs|versus|not|no|never|neither|nor|cannot|can t|couldn t|unable|unclear|"
                         r"unsure|unknown|don t|doesn t|didn t|isn t|wasn t|aren t|weren t|without|except|unlike|rather)(?!\S)")
_MIN_ANSWER_DIGITS = 4          # needle codes have 7 digits, variable values 5: shorter numbers ("Document 3") are noise
# a number the candidate explicitly rejects ("the code is 4831920, not 1111111") is not a candidate answer;
# when EVERY number is rejected the candidate answered in the negative and is wrong
_NEGATED_NUM = re.compile(r"(?i)\b(?:not|isn'?t|no|never)\s+(?:equal\s+to\s+|exactly\s+)?(-?\d[\d,]{%d,})(?![\d,])"
                          % (_MIN_ANSWER_DIGITS - 1))


def _clean(s: str) -> str:
    s = common.strip_math_delims(s or "")
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\,", "").replace("\\!", "")
    s = s.strip().strip("*_`\"'“”‘’ \t")
    s = _LEAD_ANSWER.sub("", s)
    s = s.lstrip(":= ").strip()
    s = _CITATION.sub("", s)
    s = re.sub(r"[.。!]+$", "", s.strip())
    return s.strip().strip("*_`\"'“”‘’ ")


def _answer_phrase(text: str) -> Optional[str]:
    """Text after the last answer phrase THAT HAS SOMETHING AFTER IT, up to the end of the line.
    A line-initial 'Answer: ...' (the requested format) wins over one buried in prose."""
    for rx in (_ANSWER_LINE, _ANSWER_PHRASE):
        best = None
        for m in rx.finditer(text):
            rest = text[m.end():].split("\n", 1)[0].strip()
            if rest:
                best = rest
        if best:
            return best
    return None


def extract_candidate(visible: str) -> tuple[Optional[str], str]:
    """(candidate, method): boxed -> 'answer is/answer:' phrase -> last non-empty line."""
    text = _TAG_LEFTOVER.sub(" ", visible or "")
    text = _FENCE.sub("", text)
    if not text.strip():
        return None, "empty"
    b = common.last_boxed(text)
    if b is not None and b.strip():
        return _clean(b), "boxed"
    p = _answer_phrase(text)
    if p:
        c = _clean(p)
        if c:
            return c, "phrase"
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return _clean(lines[-1]), "last_line"


def _key_number(cand: str, key: Optional[str]) -> Optional[str]:
    """The number the candidate attributes to the asked key ('... the crimson lantern is 4831920', 'QMTX = 48213'):
    the LAST such statement wins (a model's final claim).  None when the key is unknown or not attributed."""
    if not key:
        return None
    core = re.sub(r"^(?:the|a|an)\s+", "", key.strip(), flags=re.I)
    # key (whole word; a variable name such as QMTX is matched case-sensitively so 'THAT' never matches prose),
    # optional possessive, a separator and up to 4 words - none of which may cross a sentence boundary
    # ("...the crimson lantern. Codes seen: 4831920" attributes nothing to the lantern)
    rx = re.compile(r"(?<![A-Za-z])" + re.escape(core) + r"(?![A-Za-z])(?:['’]s)?[^\w\n.;!?]+(?:[^\s.;:!?]+\s+){0,4}?"
                    r"(-?\d[\d,]{%d,})(?![\d,])" % (_MIN_ANSWER_DIGITS - 1), 0 if core.isupper() else re.I)
    hits = [m.group(1) for m in rx.finditer(cand)]
    if not hits:
        return None
    n = _numbers(hits[-1])
    return n[0] if n else None


def _score_numeric(expected: str, visible: str, key: Optional[str] = None) -> Verdict:
    cand, method = extract_candidate(visible)
    if cand is None:
        return Verdict.unparsed(expected, {"method": method}, ["empty"])
    nums = _numbers(cand)
    if not nums and method == "last_line":
        cand = _TAG_LEFTOVER.sub(" ", visible)[-300:]
        nums = _numbers(cand)
        method = "last_numbers"
    if not nums:
        return Verdict.unparsed(expected, {"method": method, "candidate": cand[:120]}, ["no_number"])
    negated = [n for m in _NEGATED_NUM.finditer(cand) for n in _numbers(m.group(1))]
    if negated:
        kept = [n for n in nums if not any(_num_equal(n, x) for x in negated)]
        if not kept:
            return Verdict(False, extracted=nums[0], expected=expected,
                           detail={"method": method, "numbers": nums[:6], "reason": "negated"}, flags=["negated"])
        nums = kept
    long_nums = [n for n in nums if len(re.sub(r"\D", "", n)) >= _MIN_ANSWER_DIGITS]
    if long_nums and len(long_nums) < len(nums):
        nums = long_nums
    flags = ["multi_number"] if len(nums) > 1 else []
    detail = {"method": method, "numbers": nums[:6]}
    pick = nums[0]
    if len(nums) > 1:
        kn = _key_number(cand, key)
        if kn is not None:
            pick, detail["pick"] = kn, "key_attribution"
        elif _HEDGE_NUM.search(cand):
            return Verdict(False, extracted=nums[0], expected=expected, detail=dict(detail, reason="hedged"), flags=flags + ["hedged"])
        else:
            # several candidate numbers, none attributed to the asked key: the model enumerated the needles
            # instead of answering.  Crediting the first one would score "all the codes are A, B, C, D, E"
            # correct whenever the target happens to be listed first (RULER credit without retrieval).
            return Verdict(False, extracted=nums[0], expected=expected, detail=dict(detail, reason="uncommitted"),
                           flags=flags + ["uncommitted"])
    ok = _num_equal(pick, expected)
    return Verdict(ok, extracted=pick, expected=expected, detail=detail, flags=flags)


def _score_text(answers: list[str], visible: str) -> Verdict:
    expected = answers[0] if answers else ""
    cand, method = extract_candidate(visible)
    if cand is None:
        return Verdict.unparsed(expected, {"method": method}, ["empty"])
    cn = norm_text(cand)
    if not cn:
        return Verdict.unparsed(expected, {"method": method, "candidate": cand[:120]}, ["empty_candidate"])
    golds = [norm_text(a) for a in answers if norm_text(a)]
    detail = {"method": method, "candidate": cand[:120]}
    if cn in golds:
        return Verdict(True, extracted=cand, expected=expected, detail=detail)
    cand_nums = _numbers(cand)
    for g in answers:
        gn = _numbers(g)
        if gn and len(gn) == 1 and re.fullmatch(r"[\d.,\s]+", g.strip()) and len(cand_nums) == 1 and _num_equal(cand_nums[0], gn[0]):
            return Verdict(True, extracted=cand, expected=expected, detail=dict(detail, match="numeric"))
    # containment: the gold as a whole-word span inside a SHORT candidate that commits to it - no hedge / negation /
    # refusal words (unless the gold itself has them), no extra numbers when the gold is numeric
    n_cand = len(cn.split())
    reject: Optional[str] = None
    for g in golds:
        m = re.search(r"(?<!\S)" + re.escape(g) + r"(?!\S)", cn)
        if not m or n_cand > len(g.split()) + 5:
            continue
        tail = cn[m.end():].split()
        if tail[:1] == ["s"] and len(tail) > 1:   # "the mortgage banker's assistant" is not "the mortgage banker"
            reject = "possessive"
            continue
        hedges = {h for h in _HEDGE_TEXT.findall(cn)} - set(_HEDGE_TEXT.findall(g))
        if re.search(r"\w\s*/\s*\w", cand) and "/" not in expected:
            hedges.add("/")
        if hedges:
            return Verdict(False, extracted=cand, expected=expected, detail=dict(detail, reason="hedged", words=sorted(hedges)), flags=["hedged"])
        if _numbers(g) and set(_numbers(cn)) != set(_numbers(g)):     # both SQuAD-normalised ('29 7' vs '29 7 percent')
            return Verdict(False, extracted=cand, expected=expected, detail=dict(detail, reason="extra_numbers"), flags=["hedged"])
        return Verdict(True, extracted=cand, expected=expected, detail=dict(detail, match="containment"), flags=["containment"])
    if reject:
        return Verdict(False, extracted=cand, expected=expected, detail=dict(detail, reason=reject), flags=["hedged"])
    return Verdict(False, extracted=cand, expected=expected, detail=detail)


def score(item: dict, response_text: str, meta: Optional[dict] = None) -> Verdict:
    sub = item.get("subfamily")
    if sub == "qa_multi_doc":
        answers = list(item.get("answers") or [item.get("answer", "")])
        return _score_text(answers, response_text)
    if sub == "var_tracking":
        chain = item.get("target_chain") or []
        key = chain[-1] if chain else None          # the bare variable name ("QMTX = 48213", "VAR QMTX is 48213")
    else:
        key = (item.get("needle") or {}).get("key")
    return _score_numeric(str(item["answer"]), response_text, key)


def mock_response(item: dict):
    """The oracle: the needle / traced value / gold phrase in the requested 'Answer:' line."""
    return f"Found it in the passage.\nAnswer: {item['answer']}"


def aggregate(records: list[dict]) -> dict:
    scored = [r for r in records if r.get("status") not in ("error", "cancelled", "skipped")]
    by_len: dict[str, list[dict]] = collections.defaultdict(list)
    by_depth: dict[str, list[dict]] = collections.defaultdict(list)
    for r in scored:
        m = re.search(r"-(8k|16k|32k)-", r.get("id", ""))
        if m:
            by_len[m.group(1)].append(r)
        d = re.search(r"-d(\d\d)-", r.get("id", ""))
        if d and r.get("sub") == "niah_multikey":
            by_depth[f"0.{d.group(1)}"].append(r)

    def acc(rs):
        return round(sum(1 for r in rs if r.get("correct")) / len(rs), 4) if rs else None

    out = {"acc_by_len": {k: acc(by_len[k]) for k in ("8k", "16k", "32k") if k in by_len},
           "mean_prompt_tokens_by_len": {k: round(sum(r.get("prompt_tokens", 0) for r in by_len[k]) / len(by_len[k]), 1)
                                         for k in ("8k", "16k", "32k") if k in by_len},
           "niah_acc_by_depth": {k: acc(v) for k, v in sorted(by_depth.items())},
           "unparsed_rate": round(sum(1 for r in scored if r.get("status") == "unparsed") / len(scored), 4) if scored else None,
           "context_length_errors": sum(1 for r in records if r.get("error_kind") == "context_length")}
    if _LAST_CAL is not None:
        out["calibration"] = _LAST_CAL.to_dict()
    return out
