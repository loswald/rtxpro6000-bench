"""
families/ifeval.py - instruction following (IFEval, Zhou et al. 2023).

Source: the original 541 prompts of google-research/instruction_following_eval (input_data.jsonl, pinned to
the last commit that touched the file), fetched as a public raw URL - no HF token, no parquet.  The 25
instruction checkers of the Apache-2.0 reference implementation are re-implemented here in the standard
library (see `check()`); the deviations from the reference are listed in NOTES.

Item set (concentrated, high-discrimination): a seeded subset weighted toward prompts carrying 2-3
verifiable instructions, because prompt-level strict accuracy multiplies across instructions and is where
strong and weak open models separate (single-instruction IFEval prompts are >90 % for anything current).

  triple  prompts with 3 instructions            default 24 / full 50  (eligible pool 51)
  double  prompts with 2 instructions            default 24 / full 80  (eligible pool 167)
  single  prompts with 1 hard/medium instruction default 12 / full 40  (eligible pool 186)

Selection: within a sub-family the eligible pool is seeded-shuffled, then stably sorted by a hardness
score (sum of per-instruction-type tiers, HARDNESS below - the types the IFEval paper and later reports
show as the least followed), a first greedy pass guarantees every instruction type is covered by the set,
a second pass fills up.  The chosen items are seeded-shuffled for the run order, so --limit prefixes are
random nested subsets.

Excluded from every pool (counts reported by prepare()):
  * language:response_language with a Latin-script target (de, it, pt, fi, sw, vi): the script heuristic
    used instead of langdetect cannot verify them;
  * prompts whose instruction set is already satisfied by the content-free reply "I am not sure." (all
    constraints negative: no comma / less than N words / ...): zero discrimination, and they would let a
    canned server score;
  * prompts whose instructions are mutually contradictory under the reference checkers (e.g. two_responses
    + number_paragraphs, quotation + repeat_prompt, english_lowercase + constrained_response) or for which
    the deterministic mock builder cannot produce a passing response (`build_mock()` -> None).

Scoring: strict prompt-level (all instructions pass on the visible text) is the headline `correct`;
`score` is the strict instruction-level pass fraction; the loose evaluation (the reference's 8 variants:
first/last line removed, '*' stripped) is in `detail` and summarised by aggregate() as prompt_loose /
instruction_strict / instruction_loose / per-type accuracy.

Output budget: a few IFEval prompts ask for 600-1200 words, which a flat 1536-token cap cannot hold, so
every item carries a `max_tokens` derived from its length-implying instructions (answer_budget_tokens());
the family cap covers the longest item and prepare_run() adds the thinking allowance on --reasoning.

Mock: mock_response(item) builds a response that satisfies every instruction of the item (build_mock()),
so the oracle run scores 1.000; canned "I am not sure." fails every selected item.
"""
from __future__ import annotations

import collections
import json
import os
import re
import shutil
import string
from typing import Any, Callable, Optional

import common
from common import DEFAULT_SEED, Verdict
from families import _base

NAME = "ifeval"
DESCRIPTION = "IFEval instruction following: seeded 60-prompt subset weighted to 2-3 instructions, strict + loose programmatic checkers"
SUBFAMILIES = ["triple", "double", "single"]
PRIORITY = 60
HIDDEN = False
ITEM_TIME_FALLBACK_S = 20.0

# ---- output budgets -------------------------------------------------------------------------
# A handful of IFEval prompts ask for very long answers ("at least 900 words", "at least 100
# sentences").  A flat 1536-token cap makes those unwinnable for every model regardless of ability,
# so each item carries its own budget, derived from its length-implying instructions.
#
# run_eval.py resolves the cap as  min(family_cap, item["max_tokens"])  - a per-item value can only
# LOWER the family cap, never raise it (families/_base.py: "caps this item below the family cap").
# So DEFAULT_MAX_TOKENS must cover the LONGEST item and each item then clamps itself back down.
# The stored budget is the greedy (answer-only) one; prepare_run() adds the thinking allowance on a
# --reasoning run, because the item file holds a single number that both modes read.
WORDS_PER_SENTENCE = 12          # a floor for real prose; the mock builder writes shorter sentences
WORDS_PER_PARAGRAPH = 40         # also used for a section of a multiple_sections answer
TOKENS_PER_WORD = 2.0            # generous: IFEval answers carry markdown, [placeholders], SECTION headers
BUDGET_PREAMBLE_TOK = 256        # lead-in prose, headings, and the model overshooting a floor
BUDGET_BASE_TOK = 1536           # the family's historical greedy cap: the floor for every item
BUDGET_THINK_TOK = 2560          # historical reasoning cap 4096 - historical greedy cap 1536
BUDGET_ROUND_TOK = 256
# 2816 covers the longest prompt in the source (1200-word floor: keys 3429 and 3425)
BUDGET_MAX_ANSWER_TOK = 2816

DEFAULT_MAX_TOKENS = {"default": BUDGET_MAX_ANSWER_TOK,
                      "reasoning": BUDGET_MAX_ANSWER_TOK + BUDGET_THINK_TOK}
NOTES = [
    "ifeval: strict prompt-level accuracy is the headline; loose (reference 8-variant) and instruction-level rates are in extra",
    "ifeval: sentences are counted with a regex splitter (not nltk punkt); words are \\w+ tokens exactly as the reference's RegexpTokenizer(r'\\w+')",
    "ifeval: language:response_language uses a Unicode-script heuristic (>=50% of letters in the target script); Latin-script targets are excluded from the pool",
    "ifeval: keywords are matched as escaped literals (the reference passes them to re unescaped); blank paragraphs are dropped before indexing the nth paragraph",
    "ifeval: pool excludes prompts satisfied by a content-free reply and prompts with contradictory instructions (see manifest ifeval_build)",
    "ifeval: the content-free exclusion probes ONE reply ('I am not sure.'); 8 of the 60 default items (16 of 170 full) "
    "are still satisfied by generic filler prose - chiefly 'single' items whose only constraint is a number_words floor",
    "ifeval: number_sentences uses a regex splitter with a protected-abbreviation pass (titles, dotted "
    "initialisms, 'Fig. 3'-style refs) instead of nltk punkt; an ellipsis splits only before a capitalised "
    "word and 'St.' only in its Saint reading; residual deviations are sentence-final 'etc.'/'Inc.'/'Jr.'/"
    "'et al.', deliberately not protected because protecting a form means never splitting after it again; "
    "19 of the 132 instruction instances are number_sentences",
    "ifeval: each item carries its own max_tokens, derived from its length-implying instructions "
    "(number_words/number_sentences/number_paragraphs/multiple_sections/repeat_prompt); the family cap covers "
    "the longest item and prepare_run() adds a 2560-token thinking allowance on a --reasoning run",
    "ifeval: detectable_format:constrained_response is absent by design - all 10 source prompts carrying it are "
    "single-instruction and the checker is substring containment of one of three fixed phrases, so the item "
    "would be correct for every model and contribute nothing to the Wilson CI or to paired McNemar",
]

PIN_COMMIT = "26d8ccdab6fec61b5c83ad6327ea8bda9e580288"   # google-research/google-research, 2024-06-11 "Fix an eval prompt"
SOURCE_URL = (f"https://raw.githubusercontent.com/google-research/google-research/{PIN_COMMIT}"
              "/instruction_following_eval/data/input_data.jsonl")
SOURCE_SHA256 = "67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49"
SOURCE_NAME = "ifeval_input_data"

DEFAULT_N = {"default": {"triple": 24, "double": 24, "single": 12},
             "full": {"triple": 50, "double": 80, "single": 40}}

CANNED = "I am not sure."   # mock_server --mode canned; must fail every selected prompt

# ---- instruction ids ------------------------------------------------------------------------
KW_EXIST = "keywords:existence"
KW_FREQ = "keywords:frequency"
FORBIDDEN = "keywords:forbidden_words"
LETTER = "keywords:letter_frequency"
LANGUAGE = "language:response_language"
N_SENT = "length_constraints:number_sentences"
N_PARA = "length_constraints:number_paragraphs"
N_WORDS = "length_constraints:number_words"
NTH_PARA = "length_constraints:nth_paragraph_first_word"
PLACEHOLDERS = "detectable_content:number_placeholders"
POSTSCRIPT = "detectable_content:postscript"
BULLETS = "detectable_format:number_bullet_lists"
CONSTRAINED = "detectable_format:constrained_response"
HIGHLIGHTS = "detectable_format:number_highlighted_sections"
SECTIONS = "detectable_format:multiple_sections"
JSON_FMT = "detectable_format:json_format"
TITLE = "detectable_format:title"
TWO = "combination:two_responses"
REPEAT = "combination:repeat_prompt"
END = "startend:end_checker"
QUOTATION = "startend:quotation"
CAP_FREQ = "change_case:capital_word_frequency"
CAPITAL = "change_case:english_capital"
LOWER = "change_case:english_lowercase"
NO_COMMA = "punctuation:no_comma"

ALL_TYPES = (KW_EXIST, KW_FREQ, FORBIDDEN, LETTER, LANGUAGE, N_SENT, N_PARA, N_WORDS, NTH_PARA, PLACEHOLDERS, POSTSCRIPT,
             BULLETS, CONSTRAINED, HIGHLIGHTS, SECTIONS, JSON_FMT, TITLE, TWO, REPEAT, END, QUOTATION, CAP_FREQ, CAPITAL,
             LOWER, NO_COMMA)

# hardness tiers used to concentrate the subset (2 = least followed by current models, 0 = near-saturated)
HARDNESS = {
    N_WORDS: 2, N_SENT: 2, LETTER: 2, CAP_FREQ: 2, REPEAT: 2, NTH_PARA: 2, END: 2, KW_FREQ: 2, SECTIONS: 2, NO_COMMA: 2,
    LOWER: 1, CAPITAL: 1, N_PARA: 1, BULLETS: 1, HIGHLIGHTS: 1, QUOTATION: 1, TWO: 1, PLACEHOLDERS: 1, JSON_FMT: 1, FORBIDDEN: 1,
    KW_EXIST: 0, POSTSCRIPT: 0, TITLE: 0, CONSTRAINED: 0, LANGUAGE: 0,
}

CONSTRAINED_OPTIONS = ("My answer is yes.", "My answer is no.", "My answer is maybe.")

# Unicode script blocks for the response-language heuristic
_ARABIC = [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]
_DEVANAGARI = [(0x0900, 0x097F)]
_CYRILLIC = [(0x0400, 0x04FF)]
LANG_SCRIPTS: dict[str, list[tuple[int, int]]] = {
    "hi": _DEVANAGARI, "mr": _DEVANAGARI, "ne": _DEVANAGARI,
    "pa": [(0x0A00, 0x0A7F)], "gu": [(0x0A80, 0x0AFF)], "bn": [(0x0980, 0x09FF)],
    "ta": [(0x0B80, 0x0BFF)], "te": [(0x0C00, 0x0C7F)], "kn": [(0x0C80, 0x0CFF)],
    "th": [(0x0E00, 0x0E7F)], "ko": [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)],
    "ar": _ARABIC, "fa": _ARABIC, "ur": _ARABIC, "ru": _CYRILLIC, "bg": _CYRILLIC, "uk": _CYRILLIC,
    "ja": [(0x3040, 0x30FF), (0x4E00, 0x9FFF)], "zh": [(0x4E00, 0x9FFF)], "el": [(0x0370, 0x03FF)], "he": [(0x0590, 0x05FF)],
}

# ---- counting helpers (deviations from the nltk-based reference are listed in NOTES; count_words is exact) ----
_WORD_RE = re.compile(r"\w+")   # == the reference's nltk RegexpTokenizer(r"\w+"): hyphens/apostrophes split tokens
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|(?<=[.!?][\"'”’)])\s+")
_PUNCT = string.punctuation + "“”‘’«»"

# Abbreviation periods are not sentence terminators.  Without this pass a bare regex splitter opens a
# new sentence in the middle of one ("Dr. Smith met Prof. Jones at noon." -> 3), which nltk punkt (the
# reference's splitter) does not, and number_sentences verdicts sitting on their bound flip.
# Only forms that are effectively NEVER sentence-final are protected; "etc.", "Inc.", "Ltd.", "Jr.",
# "Sr.", "St." (a street address ends a sentence: "He lives on Main St. The house is blue.") and
# lowercase "no." are deliberately left alone, because protecting them would swallow a real boundary
# and UNDER-count - a worse error than the over-count it would remove.  Protecting a form is a
# one-way trade: it can never split after that token again, so the list holds only titles that
# always precede a name plus a few adverbials that always continue their clause.
_ABBREV_NONFINAL = ("mr", "mrs", "ms", "mx", "dr", "prof", "rev", "hon", "fr",
                    "sgt", "capt", "cmdr", "adm", "lt", "col", "gen", "maj",
                    "gov", "sen", "rep", "pres", "messrs", "mmes",
                    "vs", "cf", "al", "approx")
_ABBREV_RE = re.compile(r"(?<![\w.])(?:" + "|".join(_ABBREV_NONFINAL) + r")\.(?=\s)", re.IGNORECASE)
# reference markers, only where a number follows them ("Fig. 3", "No. 7", "Vol. 2", "pp. 41")
_ABBREV_REF_RE = re.compile(r"(?<![\w.])(?:no|nos|vol|vols|fig|figs|eq|eqs|ch|chap|sec|secs|pp|para|art)\.(?=\s+\d)",
                            re.IGNORECASE)
# "St." has two readings and only one of them is safe to protect.  Street ("He lives on Main St. The
# house is blue.") DOES end a sentence and is preceded by a proper noun or an ordinal; Saint ("We
# visited St. Louis last year.") never does and is preceded by an ordinary word or nothing.  A flat
# include/exclude gets one reading wrong either way, so the period survives only where BOTH a proper
# noun/ordinal precedes it AND a capitalised word follows - Street mid-sentence ("the shop on Baker
# St. closed early") keeps its clause, exactly as punkt does.
_ABBREV_ST_RE = re.compile(r"(?<![\w.])st\.(?=\s)", re.IGNORECASE)
_ST_STREET_LEFT = re.compile(r"(?:[A-Z][A-Za-z]*|\d+(?:st|nd|rd|th))\s+$")
_ST_STREET_RIGHT = re.compile(r"\s+[\"'“‘(\[]?[A-Z]")
# dotted initialisms and latin shorthands: "e.g.", "i.e.", "U.S.", "a.m.", "Ph.D." (no space inside)
_INITIALISM_RE = re.compile(r"(?<![\w.])(?:[A-Za-z]{1,2}\.){2,}")
# An ellipsis followed by a capitalised word IS a sentence boundary ("Wait... Stop now." -> 2, which
# is also what punkt does); anywhere else it is a mid-sentence pause ("He paused... then left." -> 1).
# The boundary form is rewritten to a plain terminator BEFORE the pause form is neutralised.
_ELLIPSIS_BOUNDARY_RE = re.compile(r"\.{3,}(?=\s+[\"'“‘(\[]?[A-Z])")
_ELLIPSIS_RE = re.compile(r"\.{3,}")


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _protect_saint_st(text: str) -> str:
    """Neutralise 'St.' only in its Saint reading (see _ST_STREET_LEFT); Street keeps its terminator."""
    def repl(m):
        # a 64-char left window is enough for the one word/ordinal _ST_STREET_LEFT can match, and
        # keeps this linear on a degenerate response that repeats "St. " thousands of times
        left = text[max(0, m.start() - 64):m.start()]
        street = _ST_STREET_LEFT.search(left) and _ST_STREET_RIGHT.match(text, m.end())
        return m.group(0) if street else m.group(0)[:-1]
    return _ABBREV_ST_RE.sub(repl, text)


def _protect_abbreviations(text: str) -> str:
    """Drop the periods that are not sentence terminators, so _SENT_SPLIT cannot break on them."""
    text = _ELLIPSIS_BOUNDARY_RE.sub(".", text)
    text = _ELLIPSIS_RE.sub("…", text)
    text = _INITIALISM_RE.sub(lambda m: m.group(0).replace(".", ""), text)
    text = _ABBREV_RE.sub(lambda m: m.group(0)[:-1], text)
    text = _protect_saint_st(text)
    return _ABBREV_REF_RE.sub(lambda m: m.group(0)[:-1], text)


def count_sentences(text: str) -> int:
    """Chunks between sentence terminators that contain a word of >= 2 characters ('P.S.' alone is not a
    sentence); abbreviation periods are neutralised first (see _protect_abbreviations)."""
    return sum(1 for s in _SENT_SPLIT.split(_protect_abbreviations(text).strip()) if re.search(r"\w{2,}", s))


def count_capital_words(text: str) -> int:
    n = 0
    for tok in text.split():
        tok = tok.strip(_PUNCT)
        if tok and any(c.isalpha() for c in tok) and tok.isupper():
            n += 1
    return n


def _relation(count: int, relation: Optional[str], n: int) -> bool:
    if relation == "less than":
        return count < n
    return count >= n          # "at least" (the reference's only other value)


def _script_fraction(text: str, ranges: list[tuple[int, int]]) -> tuple[int, int]:
    letters = [c for c in text if c.isalpha()]
    hit = sum(1 for c in letters if any(lo <= ord(c) <= hi for lo, hi in ranges))
    return hit, len(letters)


def strip_json_fences(value: str) -> str:
    v = value.strip()
    for pre in ("```json", "```Json", "```JSON", "```"):
        v = v.removeprefix(pre)
    v = v.removesuffix("```")
    return v.strip()


# ---- the 25 checkers ------------------------------------------------------------------------
def check(inst_id: str, kw: dict, value: str) -> bool:
    """Reference semantics of instruction_following_eval/instructions.py for one instruction on the raw text."""
    kw = kw or {}
    if inst_id == KW_EXIST:
        return all(re.search(re.escape(str(k)), value, re.IGNORECASE) for k in (kw.get("keywords") or []))
    if inst_id == KW_FREQ:
        n = len(re.findall(re.escape(str(kw.get("keyword", ""))), value, re.IGNORECASE))
        return _relation(n, kw.get("relation"), int(kw.get("frequency", 0)))
    if inst_id == FORBIDDEN:
        return not any(re.search(r"\b" + re.escape(str(w)) + r"\b", value, re.IGNORECASE) for w in (kw.get("forbidden_words") or []))
    if inst_id == LETTER:
        letter = str(kw.get("letter", "")).lower()
        return _relation(value.lower().count(letter) if letter else 0, kw.get("let_relation"), int(kw.get("let_frequency", 0)))
    if inst_id == LANGUAGE:
        ranges = LANG_SCRIPTS.get(str(kw.get("language", "")))
        if not ranges:
            return False
        hit, total = _script_fraction(value, ranges)
        return total >= 5 and hit / total >= 0.5
    if inst_id == N_SENT:
        return _relation(count_sentences(value), kw.get("relation"), int(kw.get("num_sentences", 0)))
    if inst_id == N_PARA:
        paragraphs = re.split(r"\s?\*\*\*\s?", value)
        n = len(paragraphs)
        for i, p in enumerate(paragraphs):
            if not p.strip():
                if i in (0, len(paragraphs) - 1):
                    n -= 1
                else:
                    return False
        return n == int(kw.get("num_paragraphs", 0))
    if inst_id == N_WORDS:
        return _relation(count_words(value), kw.get("relation"), int(kw.get("num_words", 0)))
    if inst_id == NTH_PARA:
        paragraphs = [p for p in re.split(r"\n\n", value) if p.strip()]
        num, nth = int(kw.get("num_paragraphs", 0)), int(kw.get("nth_paragraph", 0))
        if len(paragraphs) != num or nth < 1 or nth > len(paragraphs):
            return False
        toks = paragraphs[nth - 1].strip().split()
        if not toks:
            return False
        first = toks[0].lstrip("'").lstrip('"')
        w = ""
        for ch in first:
            if ch in string.punctuation:   # the reference breaks the first word at ANY punctuation
                break
            w += ch.lower()
        return w == str(kw.get("first_word", "")).lower()
    if inst_id == PLACEHOLDERS:
        return len(re.findall(r"\[.*?\]", value)) >= int(kw.get("num_placeholders", 0))
    if inst_id == POSTSCRIPT:
        marker = str(kw.get("postscript_marker", "P.S."))
        low = value.lower()
        if marker == "P.P.S":
            pat = r"\s*p\.\s?p\.\s?s.*$"
        elif marker == "P.S.":
            pat = r"\s*p\.\s?s\..*$"
        else:
            pat = r"\s*" + re.escape(marker.lower()) + r".*$"
        return bool(re.findall(pat, low, flags=re.MULTILINE))
    if inst_id == BULLETS:
        n = len(re.findall(r"^\s*\*[^\*].*$", value, re.MULTILINE)) + len(re.findall(r"^\s*-.*$", value, re.MULTILINE))
        return n == int(kw.get("num_bullets", 0))
    if inst_id == CONSTRAINED:
        return any(opt in value for opt in CONSTRAINED_OPTIONS)
    if inst_id == HIGHLIGHTS:
        n = sum(1 for h in re.findall(r"\*[^\n\*]*\*", value) if h.strip("*").strip())
        n += sum(1 for h in re.findall(r"\*\*[^\n\*]*\*\*", value) if h.removeprefix("**").removesuffix("**").strip())
        return n >= int(kw.get("num_highlights", 0))
    if inst_id == SECTIONS:
        pat = r"\s?" + re.escape(str(kw.get("section_spliter", "SECTION"))) + r"\s?\d+\s?"
        return len(re.split(pat, value)) - 1 >= int(kw.get("num_sections", 0))
    if inst_id == JSON_FMT:
        try:
            json.loads(strip_json_fences(value))
            return True
        except (ValueError, TypeError):
            return False
    if inst_id == TITLE:
        return any(t.lstrip("<").rstrip(">").strip() for t in re.findall(r"<<[^\n]+>>", value))
    if inst_id == TWO:
        parts = value.split("******")
        valid = []
        for i, p in enumerate(parts):
            if not p.strip():
                if i not in (0, len(parts) - 1):
                    return False
            else:
                valid.append(p)
        return len(valid) == 2 and valid[0].strip() != valid[1].strip()
    if inst_id == REPEAT:
        return value.strip().lower().startswith(str(kw.get("prompt_to_repeat", "")).strip().lower())
    if inst_id == END:
        return value.strip().strip('"').lower().endswith(str(kw.get("end_phrase", "")).strip().lower())
    if inst_id == QUOTATION:
        v = value.strip()
        return len(v) > 1 and v[0] == '"' and v[-1] == '"'
    if inst_id == CAP_FREQ:
        return _relation(count_capital_words(value), kw.get("capital_relation"), int(kw.get("capital_frequency", 0)))
    if inst_id == CAPITAL:
        return value.isupper()
    if inst_id == LOWER:
        return value.islower()
    if inst_id == NO_COMMA:
        return "," not in value
    raise KeyError(f"unknown IFEval instruction type {inst_id!r}")


def loose_variants(text: str) -> list[str]:
    """The reference's loose evaluation: the response, minus first line, minus last line, minus both, and
    each of those with every '*' removed (8 variants)."""
    r = text.strip()
    lines = r.split("\n")
    no_first = "\n".join(lines[1:]).strip()
    no_last = "\n".join(lines[:-1]).strip()
    no_both = "\n".join(lines[1:-1]).strip()
    base = [r, no_first, no_last, no_both]
    return base + [b.replace("*", "") for b in base]


def evaluate(instructions: list[dict], text: str) -> dict:
    """Strict and loose per-instruction results for one response."""
    variants = loose_variants(text)
    per = []
    for inst in instructions:
        iid, kw = inst["id"], inst.get("kwargs") or {}
        strict = bool(text.strip()) and check(iid, kw, text)
        loose = strict or any(v.strip() and check(iid, kw, v) for v in variants)
        per.append({"id": iid, "strict": strict, "loose": loose})
    return {"strict": all(p["strict"] for p in per) if per else False,
            "loose": all(p["loose"] for p in per) if per else False,
            "instructions": per}


def strict_all(instructions: list[dict], text: str) -> bool:
    return bool(text.strip()) and all(check(i["id"], i.get("kwargs") or {}, text) for i in instructions)


# ---- deterministic mock builder ------------------------------------------------------------
_VOCAB = ["the", "plan", "works", "well", "when", "people", "share", "clear", "goals", "and", "keep", "steady", "focus",
          "on", "small", "steps", "each", "day", "with", "care", "good", "notes", "help", "us", "track", "what", "matters",
          "most", "so", "that", "every", "task", "moves", "forward", "in", "order", "team", "members", "learn", "from",
          "past", "work", "then", "build", "better", "habits", "over", "time", "simple", "rules", "make", "hard", "choices",
          "easier", "for", "everyone", "involved", "today", "quiet", "rooms", "bring", "calm", "minds", "to", "big",
          "problems", "which", "need", "patient", "thought", "before", "any", "action", "starts"]

_LANG_WORDS: dict[str, list[str]] = {
    "hi": ["यह", "उत्तर", "बहुत", "सरल", "और", "स्पष्ट", "है", "हम", "आज", "काम", "करते", "हैं"],
    "mr": ["हे", "उत्तर", "खूप", "सोपे", "आणि", "स्पष्ट", "आहे", "आम्ही", "आज", "काम", "करतो", "चांगले"],
    "ne": ["यो", "उत्तर", "धेरै", "सरल", "र", "स्पष्ट", "छ", "हामी", "आज", "काम", "गर्छौं", "राम्रो"],
    "pa": ["ਇਹ", "ਜਵਾਬ", "ਬਹੁਤ", "ਸਧਾਰਨ", "ਅਤੇ", "ਸਪਸ਼ਟ", "ਹੈ", "ਅਸੀਂ", "ਅੱਜ", "ਕੰਮ", "ਕਰਦੇ", "ਹਾਂ"],
    "kn": ["ಇದು", "ಉತ್ತರ", "ತುಂಬಾ", "ಸರಳ", "ಮತ್ತು", "ಸ್ಪಷ್ಟ", "ಆಗಿದೆ", "ನಾವು", "ಇಂದು", "ಕೆಲಸ", "ಮಾಡುತ್ತೇವೆ", "ಚೆನ್ನಾಗಿ"],
    "te": ["ఇది", "సమాధానం", "చాలా", "సరళం", "మరియు", "స్పష్టం", "ఉంది", "మేము", "ఈరోజు", "పని", "చేస్తాము", "బాగా"],
    "ta": ["இது", "பதில்", "மிகவும்", "எளிது", "மற்றும்", "தெளிவு", "உள்ளது", "நாங்கள்", "இன்று", "வேலை", "செய்கிறோம்", "நன்றாக"],
    "gu": ["આ", "જવાબ", "ખૂબ", "સરળ", "અને", "સ્પષ્ટ", "છે", "અમે", "આજે", "કામ", "કરીએ", "છીએ"],
    "bn": ["এই", "উত্তর", "খুব", "সহজ", "এবং", "স্পষ্ট", "আছে", "আমরা", "আজ", "কাজ", "করি", "ভালো"],
    "ko": ["이것은", "답변이", "매우", "간단하고", "명확합니다", "우리는", "오늘", "일을", "합니다", "좋은", "계획을", "세웁니다"],
    "th": ["คำตอบ", "นี้", "ง่าย", "และ", "ชัดเจน", "มาก", "เรา", "ทำงาน", "วันนี้", "ดี", "แผน", "ชัด"],
    "ar": ["هذا", "الجواب", "بسيط", "وواضح", "جدا", "نحن", "نعمل", "اليوم", "بشكل", "جيد", "خطة", "واضحة"],
    "fa": ["این", "پاسخ", "بسیار", "ساده", "و", "روشن", "است", "ما", "امروز", "کار", "می‌کنیم", "خوب"],
    "ur": ["یہ", "جواب", "بہت", "آسان", "اور", "واضح", "ہے", "ہم", "آج", "کام", "کرتے", "ہیں"],
    "ru": ["это", "ответ", "очень", "простой", "и", "ясный", "мы", "сегодня", "работаем", "хорошо", "план", "понятен"],
    "bg": ["това", "отговор", "много", "прост", "и", "ясен", "ние", "днес", "работим", "добре", "план", "ясно"],
}

_MOCK_GRID = [(fl, nf) for fl in (7, 4, 11, 2, 15, 25, 40, 80) for nf in range(0, 130)]


def _assemble(instructions: list[dict], n_filler: int, filler_len: int) -> Optional[str]:
    """One candidate response for the instruction set; None when the set is structurally contradictory."""
    insts = [(i["id"], i.get("kwargs") or {}) for i in instructions]   # a type may occur twice (at least N + less than M)
    types = {t for t, _ in insts}
    has = types.__contains__

    def every(t: str) -> list[dict]:
        return [kw for tt, kw in insts if tt == t]

    def first(t: str) -> dict:
        return next((kw for tt, kw in insts if tt == t), {})

    forbidden = {str(w).lower() for kw in every(FORBIDDEN) for w in (kw.get("forbidden_words") or [])}
    avoid_letters = {str(kw["letter"]).lower() for kw in every(LETTER) if kw.get("let_relation") == "less than"}
    avoid_kws = {str(kw["keyword"]).lower() for kw in every(KW_FREQ) if kw.get("relation") == "less than"}
    lang = first(LANGUAGE).get("language")
    base = _LANG_WORDS.get(str(lang)) if lang else _VOCAB
    if base is None:
        return None

    def ok(w: str) -> bool:
        lw = w.lower()
        return (lw not in forbidden and not any(a in lw for a in avoid_letters)
                and not any(a in lw for a in avoid_kws))

    vocab = [w for w in base if ok(w)]
    if len(vocab) < 3:
        return None
    V = len(vocab)
    counter = [0]

    def words(n: int) -> list[str]:
        counter[0] += 1
        off = counter[0] * 7
        return [vocab[(off + j) % V] for j in range(max(1, n))]

    def sent(n: int, prefix: Optional[str] = None) -> str:
        ws = words(n)
        if prefix is not None:
            ws = [prefix] + ws
        else:
            ws[0] = ws[0][:1].upper() + ws[0][1:]
        return " ".join(ws) + "."

    # paragraph structure
    P, mode = 1, "plain"
    nth, first_word, spliter = 0, "", "SECTION"
    if has(NTH_PARA):
        P, mode = int(first(NTH_PARA)["num_paragraphs"]), "nth"
        nth, first_word = int(first(NTH_PARA)["nth_paragraph"]), str(first(NTH_PARA)["first_word"])
        if has(N_PARA) and int(first(N_PARA)["num_paragraphs"]) != P:
            return None
    elif has(N_PARA):
        P, mode = int(first(N_PARA)["num_paragraphs"]), "stars"
    elif has(SECTIONS):
        P, mode = int(first(SECTIONS)["num_sections"]), "sections"
        spliter = str(first(SECTIONS)["section_spliter"])
    if P < 1 or nth > P:
        return None
    paras: list[list[tuple[str, str]]] = [[] for _ in range(P)]

    for k in range(P):
        if mode == "nth" and k == nth - 1:
            paras[k].append(("s", sent(filler_len, prefix=first_word[:1].upper() + first_word[1:])))
        elif k == 0:
            paras[0].append(("s", sent(filler_len)))
    p0 = paras[0]

    if has(REPEAT):
        if has(QUOTATION) or (mode == "nth" and nth == 1):
            return None
        pr = str(first(REPEAT)["prompt_to_repeat"]).strip()
        if mode == "nth" and "\n\n" in pr:
            return None
        p0.insert(0, ("l", pr))
    if has(BULLETS):
        for _ in range(int(first(BULLETS)["num_bullets"])):
            p0.append(("l", "* " + sent(3)))
    if has(HIGHLIGHTS):
        for _ in range(max(int(kw["num_highlights"]) for kw in every(HIGHLIGHTS))):
            ws = words(3)
            p0.append(("s", f"{ws[0][:1].upper() + ws[0][1:]} *{ws[1]}* {ws[2]}."))
    if has(PLACEHOLDERS):
        n = max(int(kw["num_placeholders"]) for kw in every(PLACEHOLDERS))
        ws = words(n + 1)
        p0.append(("s", ws[0][:1].upper() + ws[0][1:] + " " + " ".join(f"[{w}]" for w in ws[1:]) + "."))
    if has(KW_EXIST):
        kws = [str(k) for kw in every(KW_EXIST) for k in (kw.get("keywords") or [])]
        p0.append(("s", vocab[0][:1].upper() + vocab[0][1:] + " " + " ".join(kws) + "."))
    for kw in every(KW_FREQ):
        if kw.get("relation") != "less than":
            p0.append(("s", " ".join([str(kw["keyword"])] * int(kw["frequency"])) + "."))
    for kw in every(LETTER):
        if kw.get("let_relation") != "less than":
            letter, n = str(kw["letter"]), int(kw["let_frequency"])
            if letter.isalpha():
                cands = [w for w in vocab if letter.lower() in w.lower()]
                toks = [cands[j % len(cands)] for j in range(n)] if cands else [letter * n]
            else:
                toks = [f"{letter}{vocab[j % V]}" for j in range(n)]
            p0.append(("s", " ".join(toks) + "."))
    for kw in every(CAP_FREQ):
        if kw.get("capital_relation") != "less than":
            n = int(kw["capital_frequency"])
            caps = [w.upper() for w in vocab[:10]]
            p0.append(("s", " ".join(caps[j % len(caps)] for j in range(n)) + "."))
    if has(CONSTRAINED):
        p0.append(("s", CONSTRAINED_OPTIONS[0]))
    if has(TITLE):
        p0.append(("l", f"<<{vocab[1][:1].upper() + vocab[1][1:]} {vocab[2 % V]}>>"))

    for j in range(n_filler):
        paras[(j + 1) % P].append(("s", sent(filler_len)))

    post: list[tuple[str, str]] = []
    if has(POSTSCRIPT):
        post.append(("l", f"{str(first(POSTSCRIPT)['postscript_marker'])} {sent(3)}"))
    if has(END):
        post.append(("l", str(first(END)["end_phrase"]).strip()))
    resp2: list[tuple[str, str]] = []
    if has(TWO):
        resp2 = [("s", sent(max(3, filler_len)))] + post
    else:
        paras[-1].extend(post)

    def render(units: list[tuple[str, str]]) -> str:
        out, prev = "", None
        for kind, t in units:
            if not out:
                out = t
            elif kind == "l" or prev == "l":
                out += "\n" + t
            else:
                out += " " + t
            prev = kind
        return out

    ptexts = []
    for k in range(P):
        units = paras[k]
        if mode == "sections":
            units = [("l", f"{spliter} {k + 1}")] + units
        ptexts.append(render(units))
    if mode == "nth" and has(N_PARA):
        joiner = "\n***\n\n"
    else:
        joiner = {"nth": "\n\n", "stars": "\n***\n", "sections": "\n\n", "plain": "\n\n"}[mode]
    body = joiner.join(ptexts)
    if has(TWO):
        body = body + "\n******\n" + render(resp2)
    if has(CAPITAL):
        body = body.upper()
    elif has(LOWER):
        body = body.lower()
    if has(JSON_FMT):
        if has(QUOTATION) or has(END):
            return json.dumps(body, ensure_ascii=False)
        return json.dumps({("RESPONSE" if has(CAPITAL) else "response"): body}, ensure_ascii=False)
    if has(QUOTATION):
        return f'"{body}"'
    return body


def build_mock(instructions: list[dict]) -> Optional[str]:
    """A response satisfying every instruction (deterministic), or None when none can be constructed."""
    try:
        for fl, nf in _MOCK_GRID:
            text = _assemble(instructions, nf, fl)
            if text is None:
                return None
            if strict_all(instructions, text):
                return text
    except (KeyError, ValueError, TypeError):
        return None
    return None


# ---- per-item output budget ------------------------------------------------------------------
def required_words(instructions: list[dict]) -> int:
    """Words the response must contain at minimum, over every length-implying instruction.

    A `number_words: less than N` bound caps the total: "at least 60 sentences" together with "less
    than 600 words" can only be met with short sentences, so 599 - not 60*12 - is the requirement.
    """
    lo, cap, echo = 0, None, 0
    for inst in instructions:
        iid, kw = inst["id"], inst.get("kwargs") or {}
        if iid == N_WORDS:
            n = int(kw.get("num_words", 0))
            if kw.get("relation") == "less than":
                cap = n - 1 if cap is None else min(cap, n - 1)
            else:
                lo = max(lo, n)
        elif iid == N_SENT:
            if kw.get("relation") != "less than":
                lo = max(lo, int(kw.get("num_sentences", 0)) * WORDS_PER_SENTENCE)
        elif iid in (N_PARA, NTH_PARA):
            lo = max(lo, int(kw.get("num_paragraphs", 0)) * WORDS_PER_PARAGRAPH)
        elif iid == SECTIONS:
            lo = max(lo, int(kw.get("num_sections", 0)) * WORDS_PER_PARAGRAPH)
        elif iid == REPEAT:
            echo += count_words(str(kw.get("prompt_to_repeat", "")))   # the echo is on top of the answer
    lo += echo          # added AFTER the max() pass, so the result cannot depend on instruction order
    return max(0, min(lo, cap) if cap is not None else lo)


def answer_budget_tokens(instructions: list[dict]) -> int:
    """Completion tokens this item's ANSWER needs, floored at the family's historical greedy cap."""
    need = int(required_words(instructions) * TOKENS_PER_WORD) + BUDGET_PREAMBLE_TOK
    need = -(-need // BUDGET_ROUND_TOK) * BUDGET_ROUND_TOK          # round up to a whole block
    return max(BUDGET_BASE_TOK, min(need, BUDGET_MAX_ANSWER_TOK))


async def prepare_run(items: list[dict], ctx: Any) -> None:
    """On a --reasoning run give every item its thinking allowance on top of the answer budget.

    The item file stores the greedy budget (one number, read by both modes); the family cap is
    BUDGET_MAX_ANSWER_TOK + BUDGET_THINK_TOK there, so without this the per-item cap would hold a
    reasoning model to its answer budget and cut the <think> block.
    """
    log = getattr(ctx, "log", print)
    if not items or not getattr(ctx, "reasoning", False):
        return
    n = 0
    for it in items:
        if it.get("_think_allowance"):
            continue
        it["max_tokens"] = int(it.get("max_tokens", BUDGET_BASE_TOK)) + BUDGET_THINK_TOK
        it["_think_allowance"] = True
        n += 1
    log(f"{NAME}: +{BUDGET_THINK_TOK} reasoning tokens on {n} items "
        f"(caps now {min(it['max_tokens'] for it in items)}..{max(it['max_tokens'] for it in items)})")


# ---- family hooks ---------------------------------------------------------------------------
def _instructions(row: dict) -> list[dict]:
    out = []
    for iid, kw in zip(row.get("instruction_id_list") or [], row.get("kwargs") or []):
        out.append({"id": iid, "kwargs": {k: v for k, v in (kw or {}).items() if v is not None}})
    return out


def _hardness(insts: list[dict]) -> int:
    return sum(HARDNESS.get(i["id"], 1) for i in insts)


def _select(pool: list[dict], n: int, sub: str, seed: int, covered: set, allow_short: bool) -> list[dict]:
    if len(pool) < n:
        if not allow_short:
            raise common.ShortPool(f"{NAME}/{sub}: pool {len(pool)} < requested {n}")
        n = len(pool)
    rng = common.seeded_rng(seed, NAME, sub)
    pool = sorted(pool, key=lambda r: int(r["key"]))
    rng.shuffle(pool)
    pool.sort(key=lambda r: -_hardness(_instructions(r)))          # stable: ties keep the seeded order
    chosen: list[dict] = []
    for r in pool:                                                   # coverage pass
        if len(chosen) >= n:
            break
        types = {i["id"] for i in _instructions(r)}
        if types - covered:
            chosen.append(r)
            covered |= types
    seen = {id(r) for r in chosen}
    for r in pool:                                                   # fill pass (hardest first)
        if len(chosen) >= n:
            break
        if id(r) not in seen:
            chosen.append(r)
            seen.add(id(r))
    rng.shuffle(chosen)                                              # run order: random nested prefixes
    return chosen


def prepare(data_dir: str, seed: int = DEFAULT_SEED, profile: str = "default", refresh: bool = False,
            log: Callable[[str], None] = print, allow_short: bool = False, n_triple: Optional[int] = None,
            n_double: Optional[int] = None, n_single: Optional[int] = None, offline_fixtures: Optional[str] = None,
            **opts) -> dict:
    sizes = dict(DEFAULT_N["full" if profile == "full" else "default"])
    for k, v in (("triple", n_triple), ("double", n_double), ("single", n_single)):
        if v is not None:
            sizes[k] = int(v)

    dest = os.path.join(data_dir, "raw", SOURCE_NAME, "input_data.jsonl")
    notes: list[str] = []
    fixture = os.path.join(offline_fixtures, SOURCE_NAME, "input_data.jsonl") if offline_fixtures else None
    if fixture and os.path.exists(fixture):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(fixture, dest)
        info = {"path": dest, "sha256": common.sha256_file(dest), "bytes": os.path.getsize(dest), "cached": True}
        notes.append(f"raw source taken from offline fixture {fixture}")
    else:
        info = common.fetch(SOURCE_URL, dest, expected_sha256=SOURCE_SHA256, refresh=refresh, log=log)
    rows = common.read_jsonl(dest)
    log(f"   {NAME}: {len(rows)} prompts from {SOURCE_URL} (sha256 {info['sha256'][:16]}.., cached={info['cached']})")

    # eligibility
    excluded: dict[str, list[int]] = collections.defaultdict(list)
    pools: dict[str, list[dict]] = {"triple": [], "double": [], "single": []}
    for r in rows:
        insts = _instructions(r)
        key = int(r["key"])
        if not insts or any(i["id"] not in ALL_TYPES for i in insts):
            excluded["unknown_instruction"].append(key)
            continue
        lang = next((i["kwargs"].get("language") for i in insts if i["id"] == LANGUAGE), None)
        if lang is not None and str(lang) not in LANG_SCRIPTS:
            excluded["latin_script_language"].append(key)
            continue
        if strict_all(insts, CANNED):
            excluded["content_free_reply_passes"].append(key)
            continue
        if build_mock(insts) is None:
            excluded["contradictory_or_unconstructible"].append(key)
            continue
        n = len(insts)
        if n >= 3:
            pools["triple"].append(r)
        elif n == 2:
            pools["double"].append(r)
        else:
            if HARDNESS.get(insts[0]["id"], 0) >= 1:
                pools["single"].append(r)
            else:
                excluded["single_easy_type"].append(key)

    covered: set = set()
    items: list[dict] = []
    counts: dict[str, int] = {}
    for sub in SUBFAMILIES:
        chosen = _select(pools[sub], sizes[sub], sub, seed, covered, allow_short)
        counts[sub] = len(chosen)
        for i, r in enumerate(chosen):
            insts = _instructions(r)
            items.append({"id": f"ifeval-{int(r['key'])}", "family": NAME, "subfamily": sub, "order": i,
                          "messages": [{"role": "user", "content": r["prompt"]}],
                          "instructions": insts,
                          "max_tokens": answer_budget_tokens(insts),
                          "meta": {"source": "google-research/instruction_following_eval", "source_key": int(r["key"]),
                                   "n_instructions": len(insts), "hardness": _hardness(insts),
                                   "required_words": required_words(insts),
                                   "types": [x["id"] for x in insts]}})

    type_cov = collections.Counter(t for it in items for t in it["meta"]["types"])
    missing = [t for t in ALL_TYPES if t not in type_cov]
    excl_counts = {k: len(v) for k, v in sorted(excluded.items())}
    notes.append(f"excluded from pools: {excl_counts}")
    notes.append(f"instruction types covered: {len(type_cov)}/{len(ALL_TYPES)}" + (f" (missing {missing})" if missing else ""))
    notes.append(f"mock_skip items in the built set: 0 (unconstructible prompts are excluded from the pool: "
                 f"{len(excluded['contradictory_or_unconstructible'])})")

    raised = sorted(((it["max_tokens"], it["id"], it["meta"]["required_words"]) for it in items
                     if it["max_tokens"] > BUDGET_BASE_TOK), reverse=True)
    budgets = {it["id"]: it["max_tokens"] for it in items}
    notes.append(f"output budgets: family cap {DEFAULT_MAX_TOKENS['default']} greedy / "
                 f"{DEFAULT_MAX_TOKENS['reasoning']} reasoning; {len(raised)} of {len(items)} items above the "
                 f"{BUDGET_BASE_TOK}-token base: " + ", ".join(f"{i}={m} (>={w}w)" for m, i, w in raised))
    clamped = [it["id"] for it in items
               if int(required_words(it["instructions"]) * TOKENS_PER_WORD) + BUDGET_PREAMBLE_TOK > BUDGET_MAX_ANSWER_TOK]
    if clamped:
        notes.append(f"output budgets CLAMPED at BUDGET_MAX_ANSWER_TOK={BUDGET_MAX_ANSWER_TOK}: {clamped} "
                     f"- raise BUDGET_MAX_ANSWER_TOK if these matter")
    for n in notes:
        log(f"   note: {n}")

    common.write_jsonl(_base.items_path(NAME, data_dir), items)
    return {"file": f"items/{NAME}.jsonl", "counts": counts,
            "pools": {sub: len(pools[sub]) for sub in SUBFAMILIES},
            "sources": {SOURCE_NAME: {"url": SOURCE_URL, "sha256": info["sha256"], "bytes": info["bytes"],
                                      "commit": PIN_COMMIT, "license": "Apache-2.0",
                                      "repo": "https://github.com/google-research/google-research/tree/master/instruction_following_eval"}},
            "notes": notes,
            "manifest_extra": {"ifeval_build": {"excluded": excl_counts, "excluded_keys": {k: sorted(v) for k, v in excluded.items()},
                                                "sizes": sizes, "type_coverage": dict(sorted(type_cov.items())),
                                                "hardness_tiers": HARDNESS,
                                                "max_tokens": {"family_cap": dict(DEFAULT_MAX_TOKENS),
                                                               "think_allowance": BUDGET_THINK_TOK,
                                                               "per_item": dict(sorted(budgets.items()))}}}}


def score(item: dict, response_text: str, meta: Optional[dict] = None) -> Verdict:
    insts = item.get("instructions") or []
    expected = "+".join(i["id"] for i in insts)
    text = response_text or ""
    if not text.strip():
        return Verdict.unparsed(expected, {"strict": False, "loose": False, "instructions": [
            {"id": i["id"], "strict": False, "loose": False} for i in insts]}, ["empty"])
    ev = evaluate(insts, text)
    n = len(ev["instructions"])
    k_strict = sum(1 for p in ev["instructions"] if p["strict"])
    k_loose = sum(1 for p in ev["instructions"] if p["loose"])
    flags: list[str] = []
    if ev["loose"] and not ev["strict"]:
        flags.append("loose_only")
    if text.lstrip().startswith("```"):
        flags.append("code_fence")
    ev["failed"] = [p["id"] for p in ev["instructions"] if not p["strict"]]
    return Verdict(ev["strict"], score=(k_strict / n if n else 0.0),
                   extracted=f"{k_strict}/{n} strict, {k_loose}/{n} loose", expected=expected, detail=ev, flags=flags)


def mock_response(item: dict):
    """A response that satisfies every instruction of the item (None only for an unconstructible set)."""
    return build_mock(item.get("instructions") or [])


def aggregate(records: list[dict]) -> dict:
    scored = [r for r in records if r.get("status") not in ("error", "cancelled", "skipped")
              and isinstance(r.get("detail"), dict) and isinstance(r["detail"].get("instructions"), list)]
    if not scored:
        return {}
    n_items = len(scored)
    prompt_strict = sum(1 for r in scored if r["detail"].get("strict")) / n_items
    prompt_loose = sum(1 for r in scored if r["detail"].get("loose")) / n_items
    per_type: dict[str, dict] = collections.defaultdict(lambda: {"n": 0, "strict": 0, "loose": 0})
    by_n: dict[int, dict] = collections.defaultdict(lambda: {"n": 0, "strict": 0, "loose": 0})
    tot = strict = loose = 0
    for r in scored:
        insts = r["detail"]["instructions"]
        b = by_n[len(insts)]
        b["n"] += 1
        b["strict"] += int(bool(r["detail"].get("strict")))
        b["loose"] += int(bool(r["detail"].get("loose")))
        for p in insts:
            t = per_type[p["id"]]
            t["n"] += 1
            t["strict"] += int(bool(p.get("strict")))
            t["loose"] += int(bool(p.get("loose")))
            tot += 1
            strict += int(bool(p.get("strict")))
            loose += int(bool(p.get("loose")))
    r4 = lambda x: round(x, 4)  # noqa: E731
    return {
        "prompt_strict": r4(prompt_strict), "prompt_loose": r4(prompt_loose),
        "instruction_strict": r4(strict / tot) if tot else None, "instruction_loose": r4(loose / tot) if tot else None,
        "n_instructions": tot,
        "by_n_instructions": {str(k): {"n": v["n"], "prompt_strict": r4(v["strict"] / v["n"]), "prompt_loose": r4(v["loose"] / v["n"])}
                              for k, v in sorted(by_n.items())},
        "per_type": {k: {"n": v["n"], "strict": r4(v["strict"] / v["n"]), "loose": r4(v["loose"] / v["n"])}
                     for k, v in sorted(per_type.items())},
        "loose_only_rate": r4(sum(1 for r in scored if "loose_only" in (r.get("flags") or [])) / n_items),
    }
