"""
families/knowledge.py - knowledge and graduate-level QA.

Two sub-families, both from public, ungated sources fetched with common.fetch() (sha256 recorded):

  mmlu_pro_hard  TIGER-Lab/MMLU-Pro test split, read page by page from the public HF datasets-server
                 rows API (no parquet reader needed).  A seeded, category-balanced subset of the five
                 categories with the widest spread between weak and strong open models - math, physics,
                 chemistry, engineering, law - restricted to full 10-option items (A-J) and, inside the
                 STEM categories, preferring the computation-heavy items MMLU-Pro added on top of the
                 original MMLU (src stemez-*/theoremQA-*/scibench-*; the "ori_mmlu-*" items top up only
                 when a pool is short).  Zero-shot CoT by default; --family-opt knowledge.shots=5 renders
                 the official 5-shot CoT prompt from the validation-split exemplars stored as a header
                 record ({"_exemplars": {category: [...]}}).  Scored by a robust final-letter cascade
                 (\\boxed{C}, "Answer: (C)", "The answer is C.", **C**, bare line, option-text match,
                 last standalone letter).
  simpleqa       OpenAI simple-evals SimpleQA test set (public Azure blob CSV).  A seeded, topic-
                 stratified subset of short-answer items (<= 4-word gold answers), asked for a bare
                 "Answer: <answer>" and scored by normalised containment of the gold answer in the
                 extracted answer span (accent/punctuation/article-insensitive token containment, alias
                 splitting on "or" / "/" / parentheses, numeric equality for Number answers, month
                 canonicalisation for dates).  No LLM judge.
  hle_public     NOT built: Humanity's Last Exam (cais/hle) is gated on HF (rows API -> 401) and no ungated
                 mirror is known, so there is nothing to fetch under the no-gated-datasets rule.

Options (prepare_data --opt knowledge.KEY=VALUE): categories=math,physics,... (MMLU-Pro categories),
n_mmlu=N, n_simpleqa=N.  Run options (--family-opt knowledge.KEY=VALUE): shots=0|5.
"""
from __future__ import annotations

import ast
import csv
import io
import json
import math
import os
import re
import time
import unicodedata
from typing import Callable, Optional

import common
from common import DEFAULT_SEED, ShortPool, Verdict
from families import _base

NAME = "knowledge"
DESCRIPTION = "MMLU-Pro hard-category 10-option MCQ (datasets-server rows API) + SimpleQA short-answer factuality"
SUBFAMILIES = ["mmlu_pro_hard", "simpleqa"]
PRIORITY = 50
HIDDEN = False
DEFAULT_MAX_TOKENS = {"default": 1024, "reasoning": 4096}
ITEM_TIME_FALLBACK_S = 40.0
NOTES = [
    "knowledge/mmlu_pro_hard: seeded category-balanced subset (math, physics, chemistry, engineering, law), "
    "10-option items only, STEM items prefer the non-ori_mmlu (stemez/theoremQA/scibench) sources; ~1-2% of "
    "MMLU-Pro keys are known to be noisy and are not corrected here",
    "knowledge/simpleqa: programmatic containment scoring (all gold tokens must appear in the extracted answer "
    "span; the span is the 'Answer:'/boxed phrase, else the last line of the response); hedged ('X or Y') and "
    "negated ('Y, not X') answers are not credited, matching the official grader's NOT_ATTEMPTED/INCORRECT; "
    "stricter than the official LLM grader on partial names, so absolute numbers are not comparable "
    "with published SimpleQA scores - use it for paired comparisons",
    "knowledge/hle_public: skipped - cais/hle is gated (datasets-server rows API returns 401)",
]

# ---- sources -------------------------------------------------------------------------------------
MMLU_PRO_DATASET = "TIGER-Lab/MMLU-Pro"
ROWS_API = "https://datasets-server.huggingface.co/rows"
PAGE = 100
HF_REPO_API = f"https://huggingface.co/api/datasets/{MMLU_PRO_DATASET}"
SIMPLEQA_URL = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"

# widest weak-vs-strong spread first: the remainder of a quota goes to the first categories
DEFAULT_CATEGORIES = ["engineering", "law", "physics", "chemistry", "math"]
LETTERS = "ABCDEFGHIJ"
PROMPT_VERSION = 1

ZERO_SHOT_HEADER = ('The following is a multiple choice question about {category}. Think step by step and then '
                    'finish your answer with "The answer is (X)" where X is the correct letter choice.\n\n')
FEW_SHOT_HEADER = ('The following are multiple choice questions (with answers) about {category}. Think step by step '
                   'and then finish your answer with "The answer is (X)" where X is the correct letter choice.\n\n')
SIMPLEQA_SUFFIX = '\n\nGive only the final answer, as briefly as possible, in the form "Answer: <answer>".'


# ==================================================================================================
# prepare
# ==================================================================================================

def _fetch_with_sha(url: str, dest: str, refresh: bool, log: Callable[[str], None]) -> tuple[dict, dict]:
    info = common.fetch(url, dest, refresh=refresh, log=log)
    with open(dest, "r", encoding="utf-8") as f:
        return json.load(f), info


def _rows_url(split: str, offset: int) -> str:
    return (f"{ROWS_API}?dataset={MMLU_PRO_DATASET}&config=default&split={split}&offset={offset}&length={PAGE}")


PAGE_PACING_S = 0.8          # between uncached page fetches (the anonymous rows API rate-limits bursts)
RATE_LIMIT_WAITS_S = (60, 90, 120, 180, 240)


def _fetch_page(url: str, dest: str, refresh: bool, log: Callable[[str], None]) -> tuple[dict, dict]:
    """common.fetch() with its own short back-off, wrapped in a patient retry on HTTP 429 (the
    datasets-server's anonymous quota is per burst; a minute or two clears it)."""
    for attempt, wait in enumerate(RATE_LIMIT_WAITS_S + (None,)):
        try:
            data, info = _fetch_with_sha(url, dest, refresh, log)
        except (IOError, OSError) as e:
            if "429" not in str(e) or wait is None:
                raise
            log(f"   {NAME}: rate-limited by the rows API (attempt {attempt + 1}); sleeping {wait}s")
            time.sleep(wait)
            continue
        if not info.get("cached"):
            time.sleep(PAGE_PACING_S)
        return data, info
    raise IOError(f"could not fetch {url}")  # unreachable


def _crawl_split(split: str, data_dir: str, refresh: bool, log: Callable[[str], None]) -> tuple[list[dict], dict]:
    """All rows of one MMLU-Pro split through the paginated rows API; one cached JSON file per page."""
    raw_dir = os.path.join(data_dir, "raw", "mmlu_pro")
    rows: list[dict] = []
    page_shas: list[str] = []
    total_bytes = 0
    offset = 0
    n_total: Optional[int] = None
    while True:
        dest = os.path.join(raw_dir, f"{split}_p{offset // PAGE:04d}.json")
        data, info = _fetch_page(_rows_url(split, offset), dest, refresh, log)
        page_shas.append(info["sha256"])
        total_bytes += int(info["bytes"])
        got = int(data.get("num_rows_total") or 0)
        if n_total is None:
            n_total = got
            log(f"   {NAME}: {MMLU_PRO_DATASET} {split}: {n_total} rows, {math.ceil(n_total / PAGE)} pages")
        elif got != n_total:
            log(f"   {NAME}: WARNING page offset={offset} reports num_rows_total={got} != {n_total} (stale cache? use --refresh)")
        for r in data.get("rows", []):
            row = dict(r["row"])
            row["_row_idx"] = int(r.get("row_idx", offset + len(rows)))
            rows.append(row)
        offset += PAGE
        if offset >= (n_total or 0) or not data.get("rows"):
            break
    source = {"url": _rows_url(split, 0).replace("offset=0", "offset=<0..N step 100>"), "pages": len(page_shas),
              "sha256": common.sha256_bytes("\n".join(page_shas).encode("utf-8")), "bytes": total_bytes,
              "page_sha256": page_shas, "n_rows": len(rows)}
    return rows, source


def _parse_categories(val) -> list[str]:
    if val is None:
        return list(DEFAULT_CATEGORIES)
    if isinstance(val, str):
        cats = [c.strip().lower() for c in val.split(",") if c.strip()]
    else:
        cats = [str(c).strip().lower() for c in val]
    return cats or list(DEFAULT_CATEGORIES)


def _quotas(n: int, keys: list[str]) -> dict[str, int]:
    base, rem = divmod(n, len(keys))
    return {k: base + (1 if i < rem else 0) for i, k in enumerate(keys)}


def _format_options(options: list[str]) -> str:
    return "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))


def _question_block(question: str, options: list[str]) -> str:
    return f"Question: {question.strip()}\nOptions:\n{_format_options(options)}\n"


_STRIPPED_EXPONENT = re.compile(r"(?:×|\bx\b|\*)\s*10\s*(?![\^\d\-−–]|\*\*)")


def _looks_corrupted(question: str, options: list[str]) -> bool:
    """MMLU-Pro's stemez items lost some superscripts in transcription ('3 × 10 (V/m)' for 3 × 10^6):
    such an item is unanswerable noise, not discrimination."""
    return any(_STRIPPED_EXPONENT.search(t or "") for t in [question] + list(options))


def _select_mmlu(rows: list[dict], categories: list[str], n: int, seed: int, allow_short: bool,
                 log: Callable[[str], None]) -> tuple[list[dict], dict, dict]:
    quotas = _quotas(n, categories)
    pools: dict[str, int] = {}
    picked: dict[str, list[dict]] = {}
    for cat in categories:
        cat_rows = [r for r in rows if str(r.get("category", "")).lower() == cat]
        full = [r for r in cat_rows if isinstance(r.get("options"), list) and len(r["options"]) == 10
                and str(r.get("answer", "")) in LETTERS
                and int(r.get("answer_index", -1)) == LETTERS.index(str(r["answer"]))
                and all(isinstance(o, str) and o.strip() for o in r["options"])]
        n_corrupt = sum(1 for r in full if _looks_corrupted(str(r["question"]), r["options"]))
        full = [r for r in full if not _looks_corrupted(str(r["question"]), r["options"])]
        hard = [r for r in full if not str(r.get("src", "")).startswith("ori_mmlu")] if cat != "law" else list(full)
        rest = [r for r in full if r not in hard]
        pools[cat] = len(full)
        rng = common.seeded_rng(seed, NAME, "mmlu_pro_hard", cat)
        hard.sort(key=lambda r: int(r["question_id"]))
        rest.sort(key=lambda r: int(r["question_id"]))
        rng.shuffle(hard)
        rng.shuffle(rest)
        take = (hard + rest)[: quotas[cat]]
        log(f"   {NAME}/mmlu_pro_hard/{cat}: rows={len(cat_rows)} 10-option-clean={len(full)} (stripped-exponent dropped {n_corrupt}) "
            f"hard-pool={len(hard)} take={len(take)}/{quotas[cat]}")
        if len(take) < quotas[cat]:
            if not allow_short:
                raise ShortPool(f"mmlu_pro_hard/{cat}: {len(full)} eligible rows < quota {quotas[cat]}")
            log(f"   {NAME}: short pool for {cat} accepted (--allow-short)")
        picked[cat] = take
    ordered = _base.interleave([picked[c] for c in categories])
    items: list[dict] = []
    for i, r in enumerate(ordered):
        q = str(r["question"]).strip()
        opts = [str(o).strip() for o in r["options"]]
        block = _question_block(q, opts)
        marker = " ".join(f"Question: {q}".split())
        if len(marker) < 40:
            marker = " ".join(f"Question: {q} Options: A. {opts[0]}".split())
        items.append({
            "id": f"mmlupro-{int(r['question_id'])}", "family": NAME, "subfamily": "mmlu_pro_hard", "order": i,
            "messages": [{"role": "user", "content": ZERO_SHOT_HEADER.format(category=r["category"]) + block}],
            "question": q, "options": opts, "answer": str(r["answer"]), "category": str(r["category"]),
            "mock_marker": marker,
            "meta": {"source": MMLU_PRO_DATASET, "source_id": int(r["question_id"]), "src": str(r.get("src", "")),
                     "category": str(r["category"]), "n_options": len(opts), "row_idx": r.get("_row_idx")},
        })
    counts = dict(sorted((c, len(picked[c])) for c in categories))
    return items, pools, counts


def _exemplars(val_rows: list[dict], categories: list[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in sorted(val_rows, key=lambda r: int(r["question_id"])):
        cat = str(r.get("category", "")).lower()
        if cat not in categories:
            continue
        cot = str(r.get("cot_content") or "").strip()
        cot = re.sub(r"^A:\s*", "", cot)
        if not cot:
            cot = f"Let's think step by step. The answer is ({r['answer']})."
        out.setdefault(cat, []).append({"question": str(r["question"]).strip(), "options": [str(o) for o in r["options"]],
                                        "answer": str(r["answer"]), "cot": cot, "question_id": int(r["question_id"])})
    return out


# ---- SimpleQA -------------------------------------------------------------------------------------

def _parse_meta(s: str) -> dict:
    try:
        m = ast.literal_eval(s)
        return m if isinstance(m, dict) else {}
    except Exception:
        return {}


def _select_simpleqa(csv_text: str, n: int, seed: int, allow_short: bool, log: Callable[[str], None]) -> tuple[list[dict], int, dict]:
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    cands: list[dict] = []
    for idx, r in enumerate(rows):
        gold = (r.get("answer") or "").strip()
        prob = (r.get("problem") or "").strip()
        meta = _parse_meta(r.get("metadata") or "")
        if not gold or not prob or len(gold.split()) > 4 or len(gold) > 40 or len(prob) > 600:
            continue
        if not _norm_tokens(gold):
            continue
        cands.append({"idx": idx, "problem": prob, "answer": gold, "topic": str(meta.get("topic") or "Other"),
                      "answer_type": str(meta.get("answer_type") or "Other")})
    topics = sorted({c["topic"] for c in cands})
    quotas = _quotas(n, topics)
    picked: dict[str, list[dict]] = {}
    leftovers: list[dict] = []
    for t in topics:
        by_type: dict[str, list[dict]] = {}
        for c in cands:
            if c["topic"] == t:
                by_type.setdefault(c["answer_type"], []).append(c)
        rng = common.seeded_rng(seed, NAME, "simpleqa", t)
        groups = []
        for at in sorted(by_type):
            g = sorted(by_type[at], key=lambda c: c["idx"])
            rng.shuffle(g)
            groups.append(g)
        rng.shuffle(groups)                 # otherwise small quotas always start with the same answer types
        mixed = _base.interleave(groups)
        picked[t] = mixed[: quotas[t]]
        leftovers.extend(mixed[quotas[t]:])
    short = sum(quotas[t] - len(picked[t]) for t in topics)
    if short:
        rng = common.seeded_rng(seed, NAME, "simpleqa", "topup")
        leftovers.sort(key=lambda c: c["idx"])
        rng.shuffle(leftovers)
        extra = leftovers[:short]
        for c in extra:
            picked[c["topic"]].append(c)
    ordered = _base.interleave([picked[t] for t in topics])
    if len(ordered) < n and not allow_short:
        raise ShortPool(f"simpleqa: {len(ordered)} eligible rows < requested {n}")
    items: list[dict] = []
    for i, c in enumerate(ordered):
        items.append({
            "id": f"simpleqa-{c['idx']:04d}", "family": NAME, "subfamily": "simpleqa", "order": i,
            "messages": [{"role": "user", "content": c["problem"] + SIMPLEQA_SUFFIX}],
            "answer": c["answer"], "aliases": _aliases(c["answer"]), "answer_type": c["answer_type"], "topic": c["topic"],
            "mock_marker": " ".join(c["problem"].split())[:240],
            "meta": {"source": "openai/simple-evals simple_qa_test_set.csv", "source_id": c["idx"],
                     "topic": c["topic"], "answer_type": c["answer_type"]},
        })
    log(f"   {NAME}/simpleqa: csv rows={len(rows)} eligible={len(cands)} topics={len(topics)} take={len(items)}/{n}")
    return items, len(cands), dict(sorted((t, len(picked[t])) for t in topics))


def prepare(data_dir: str, seed: int = DEFAULT_SEED, profile: str = "default", refresh: bool = False,
            log: Callable[[str], None] = print, allow_short: bool = False, categories=None, n_mmlu=None,
            n_simpleqa=None, simpleqa_url: Optional[str] = None, **opts) -> dict:
    cats = _parse_categories(categories)
    n_m = int(n_mmlu) if n_mmlu is not None else (80 if profile == "full" else 40)
    n_s = int(n_simpleqa) if n_simpleqa is not None else (60 if profile == "full" else 30)
    notes: list[str] = list(NOTES)
    sources: dict = {}

    # -- MMLU-Pro: repo revision + test/validation crawl ------------------------------------------
    revision = None
    try:
        repo, _ = _fetch_with_sha(HF_REPO_API, os.path.join(data_dir, "raw", "mmlu_pro", "repo.json"), refresh, log)
        revision = repo.get("sha")
    except Exception as e:  # the revision is informational; the page hashes pin the content
        log(f"   {NAME}: could not read {HF_REPO_API}: {e}")
    test_rows, src_test = _crawl_split("test", data_dir, refresh, log)
    val_rows, src_val = _crawl_split("validation", data_dir, refresh, log)
    src_test["revision"] = revision
    src_val["revision"] = revision
    sources["mmlu_pro_test_rows_api"] = src_test
    sources["mmlu_pro_validation_rows_api"] = src_val
    mmlu_items, mmlu_pools, mmlu_by_cat = _select_mmlu(test_rows, cats, n_m, seed, allow_short, log)
    exemplars = _exemplars(val_rows, cats)

    # -- SimpleQA ---------------------------------------------------------------------------------
    sq_items: list[dict] = []
    sq_pool = 0
    sq_by_topic: dict = {}
    url = simpleqa_url or SIMPLEQA_URL
    dest = os.path.join(data_dir, "raw", "simpleqa", "simple_qa_test_set.csv")
    try:
        info = common.fetch(url, dest, refresh=refresh, log=log)
        sources["simpleqa_test_set"] = {"url": url, "sha256": info["sha256"], "bytes": info["bytes"]}
        with open(dest, "r", encoding="utf-8", newline="") as f:
            csv_text = f.read()
        sq_items, sq_pool, sq_by_topic = _select_simpleqa(csv_text, n_s, seed, allow_short, log)
    except ShortPool:
        raise
    except Exception as e:
        if not allow_short:
            raise
        notes.append(f"simpleqa skipped: {url} unreachable ({type(e).__name__}: {e})")
        log(f"   {NAME}: simpleqa skipped - {type(e).__name__}: {e}")

    header = {"_exemplars": exemplars, "_prompt_version": PROMPT_VERSION, "_categories": cats,
              "_headers": ["ZERO_SHOT_HEADER", "FEW_SHOT_HEADER", "SIMPLEQA_SUFFIX"]}
    common.write_jsonl(_base.items_path(NAME, data_dir), [header] + mmlu_items + sq_items)
    counts = {"mmlu_pro_hard": len(mmlu_items), "simpleqa": len(sq_items)}
    pools = {"mmlu_pro_hard": sum(mmlu_pools.values()), "simpleqa": sq_pool}
    return {"file": f"items/{NAME}.jsonl", "counts": counts, "sources": sources, "pools": pools, "notes": notes,
            "manifest_extra": {"knowledge_selection": {"mmlu_pro_categories": mmlu_by_cat, "mmlu_pro_pools_10opt": mmlu_pools,
                                                       "simpleqa_topics": sq_by_topic, "prompt_version": PROMPT_VERSION}}}


# ==================================================================================================
# prompt builder (shots)
# ==================================================================================================

_HEADER_CACHE: dict[str, dict] = {}


def _headers(data_dir: Optional[str]) -> dict:
    key = data_dir or common.DEFAULT_DATA_DIR
    if key not in _HEADER_CACHE:
        _HEADER_CACHE[key] = _base.read_headers(NAME, data_dir)
    return _HEADER_CACHE[key]


def build_messages(item: dict, ctx) -> list[dict]:
    if item.get("subfamily") != "mmlu_pro_hard":
        return item["messages"]
    shots = int(ctx.opt("shots", 0) or 0) if ctx is not None else 0
    if shots <= 0:
        return item["messages"]
    ex = (_headers(getattr(ctx, "data_dir", None)).get("_exemplars") or {}).get(str(item.get("category", "")).lower(), [])
    ex = ex[:shots]
    if not ex:
        return item["messages"]
    parts = [FEW_SHOT_HEADER.format(category=item["category"])]
    for e in ex:
        parts.append(_question_block(e["question"], e["options"]) + f"Answer: {e['cot']}\n\n")
    parts.append(_question_block(item["question"], item["options"]) + "Answer: Let's think step by step.")
    return [{"role": "user", "content": "".join(parts)}]


# ==================================================================================================
# scoring - MCQ
# ==================================================================================================

# NOTE (review): every quantifier below is bounded and no two unbounded whitespace/star runs are
# adjacent.  The earlier "\s*...\**\s*...\s*" chains backtracked polynomially: a response with ~200
# whitespace characters after the word "answer" (a degenerate / truncated reasoning model, or a
# markdown block) took > 3 s inside score(), which runs on the runner's event loop.
_MCQ_BOXED = re.compile(
    r"\\(?:boxed|fbox)[ \t]{0,4}\{[\s]{0,4}(?:\\(?:text|textbf|mathrm|mathbf)[ \t]{0,4}\{[\s]{0,4})?"
    r"[(\[]{0,2}[ \t*]{0,4}([A-J])[ \t*]{0,4}[)\]]{0,2}"
    r"(?:[ \t]{0,4}[.:)][^{}]{0,200}|[\s]{0,4}\}[^{}]{0,200})?\}")
_MCQ_PHRASE = re.compile(
    r"(?i:final\s{0,3}answer|answer|correct\s{0,3}(?:option|choice|answer|letter))"
    r"[\s:=*-]{0,8}(?i:is[\s:=*-]{0,4})?(?i:(?:option|choice|letter)[\s:=*-]{0,4})?"
    r"[(\[]{0,2}[\s*]{0,4}([A-J])[\s*]{0,4}[)\]]{0,2}(?![A-Za-z0-9])")
# a lower-case letter is only trusted inside parentheses ("the answer is (c)"); a bare "c" in prose is
# far more often a variable (speed of light, concentration) than an option letter
_MCQ_PHRASE_LC = re.compile(
    r"(?i:final\s{0,3}answer|answer|correct\s{0,3}(?:option|choice|answer|letter))"
    r"[\s:=*-]{0,8}(?i:is[\s:=*-]{0,4})?\([\s*]{0,3}([a-j])[\s*]{0,3}\)")


_MCQ_BARE_LINE = re.compile(r"^[ \t]{0,8}[(\[]{0,2}[*]{0,3}([A-Ja-j])[*]{0,3}[)\]]{0,2}[ \t]{0,4}[.:;)]{0,2}[ \t*]{0,4}$")
# explicit letter markers anywhere in the tail
_MCQ_TAIL_STRONG = re.compile(r"(?:(?<!not )\(([A-J])\)|(?i:option|choice)[ \t]{0,3}\(?([A-J])\)?(?![A-Za-z0-9]))")
# a bare / punctuated letter is only trusted when the response ENDS there, is not a unit that follows a
# number ("0.5 J.", "3 A.") and is not negated ("... is not C.")
_MCQ_TAIL_END = re.compile(
    r"(?<!\d)(?<!\d )(?<!not )(?<![A-Za-z0-9])([A-J])[\s*]{0,4}[.)\]:;!]{0,3}[\s*]{0,4}$")
# "C or D", "C, D": the model named a second candidate right next to the extracted letter
_HEDGE_AFTER = re.compile(r"^[\s*)\]]{0,3}(?:,|/|;|\bor\b|\band\b)[\s*(\[]{0,3}([A-J])(?![A-Za-z0-9])")
_HEDGE_BEFORE = re.compile(r"(?<![A-Za-z0-9])([A-J])[\s*)\]]{0,3}(?:,|/|;|\bor\b|\band\b)[\s*(\[]{0,3}$")


def _norm_opt(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace("\\(", "").replace("\\)", "").replace("$", "")
    s = re.sub(r"\\[a-zA-Z]+", " ", s)        # \text{...} \mathrm{...} \ - so \boxed{12.4\ \text{kg}} matches "12.4 kg"
    s = re.sub(r"[^\w.%/+-]+", " ", s)
    return " ".join(s.split())


def _last_valid(rx: re.Pattern, text: str, valid: set) -> Optional[str]:
    """Last match of rx whose (first non-empty) capture group is a valid letter (case-folded)."""
    found = None
    for m in rx.finditer(text):
        g = next((x for x in m.groups() if x), None)
        if g and g.upper() in valid:
            found = g.upper()
    return found


# "answer A is the classic trap here" - the letter is the SUBJECT of a following clause, i.e. a remark
# about an option, not the answer.  Such a match is only used when there is no other one.
_LETTER_AS_SUBJECT = re.compile(r"^[\s*)\]]{0,3}(?:is|are|was|were|would|will|looks|seems|appears|might|may|can|could)\b")


def _last_phrase_letter(text: str, valid: set) -> Optional[str]:
    """Last answer-phrase letter, preferring a match that is not a remark about another option."""
    strong = weak = None
    for rx in (_MCQ_PHRASE, _MCQ_PHRASE_LC):
        for m in rx.finditer(text):
            g = (m.group(1) or "").upper()
            if g not in valid:
                continue
            if _LETTER_AS_SUBJECT.match(text[m.end(): m.end() + 12]):
                weak = g
            else:
                strong = g
        if strong or weak:
            break
    return strong or weak


def _is_hedged(text: str, letter: str) -> bool:
    """True when the extracted letter sits inside an uncommitted alternation ("C or D", "C, D"):
    the model named two options, which is not an answer."""
    for m in re.finditer(rf"(?<![A-Za-z0-9]){letter}(?![A-Za-z0-9])", text):
        after = _HEDGE_AFTER.match(text[m.end(): m.end() + 12])
        before = _HEDGE_BEFORE.search(text[max(0, m.start() - 12): m.start()])
        if (after and after.group(1) != letter) or (before and before.group(1) != letter):
            return True
    return False


def extract_mcq_letter(text: str, options: Optional[list[str]] = None, n_options: int = 10) -> tuple[Optional[str], str]:
    """Final-letter cascade for 10-option MCQ. Returns (letter | None, method)."""
    n = len(options) if options else n_options
    valid = set(LETTERS[: max(1, min(10, n))])
    text = (text or "").strip()
    if not text:
        return None, "none"
    hit = _last_valid(_MCQ_BOXED, text, valid)
    if hit:
        return hit, "boxed"
    hit = _last_phrase_letter(text, valid)
    if hit:
        return hit, "answer_phrase"
    # the model restated the option text instead of the letter (\boxed{12.4 kg} as well as
    # "The answer is 12.4 kg" - extract_final_answer prefers the box, then the answer phrase)
    if options:
        phrase, _m = common.extract_final_answer(text, allow_last_integer=False)
        if phrase:
            p = _norm_opt(common.strip_math_delims(phrase))
            normed = [_norm_opt(o) for o in options]
            exact = [i for i, o in enumerate(normed) if o and o == p]
            if len(exact) == 1:
                return LETTERS[exact[0]], "option_text"
            contained = [i for i, o in enumerate(normed) if len(o) >= 4 and o in p]
            if len(contained) == 1:
                return LETTERS[contained[0]], "option_text"
            # the model gave the option's value without its unit ("The answer is 12.4" for "12.4 kg"):
            # only when the phrase covers a whole leading token of exactly one option
            prefix = [i for i, o in enumerate(normed) if len(p) >= 3 and o.startswith(p + " ")]
            if len(prefix) == 1:
                return LETTERS[prefix[0]], "option_text"
    # a bare letter on one of the last three non-empty lines: "C", "(C).", "**C**"
    lines = [ln for ln in text.splitlines() if ln.strip()][-3:]
    for ln in reversed(lines):
        m = _MCQ_BARE_LINE.match(ln)
        if m and m.group(1).upper() in valid:
            return m.group(1).upper(), "bare_line"
    # weak fallbacks over the tail: explicit "(C)" / "option C" anywhere, then a letter the response
    # actually ENDS on.  An unanchored "C." / last-standalone-letter rule reads unit symbols
    # ("0.5 J.", "3 A."), figure references ("panel D)") and the article "A" as answers - a 1-in-10
    # free hit per malformed response.
    tail = text[-300:]
    hit = _last_valid(_MCQ_TAIL_STRONG, tail, valid)
    if hit:
        return hit, "tail_paren"
    m = _MCQ_TAIL_END.search(tail)
    if m and m.group(1) in valid - {"I"}:
        return m.group(1), "last_letter"
    return None, "none"


def _score_mcq(item: dict, text: str) -> Verdict:
    expected = str(item["answer"])
    letter, method = extract_mcq_letter(text, item.get("options"), int(item.get("meta", {}).get("n_options", 10)))
    if letter is None:
        return Verdict.unparsed(expected, {"method": method}, ["no_letter"])
    flags = ["weak_extraction"] if method in ("last_letter", "tail_paren") else []
    if _is_hedged((text or "")[-300:], letter):
        return Verdict(False, "wrong", 0.0, letter, expected, {"method": method, "hedged": True}, flags + ["hedged"])
    return Verdict(letter == expected, extracted=letter, expected=expected, detail={"method": method}, flags=flags)


# ==================================================================================================
# scoring - SimpleQA containment
# ==================================================================================================

_MONTHS = {"jan": "january", "feb": "february", "mar": "march", "apr": "april", "jun": "june", "jul": "july",
           "aug": "august", "sep": "september", "sept": "september", "oct": "october", "nov": "november", "dec": "december"}
_STOP = {"the", "a", "an", "of", "and", "in", "at", "on", "de", "la", "le", "del", "von", "van", "der", "da", "di", "du", "of"}
_NOISE = {"approximately", "approx", "about", "around", "roughly", "circa", "ca", "c", "nearly", "almost", "some"}
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _norm_tokens(s: str, drop: Optional[frozenset] = None) -> list[str]:
    """Normalised tokens.  `drop` defaults to the stop+noise set (used by prepare() for eligibility
    and alias building - do not change that default, the built item set depends on it)."""
    stop = (_STOP | _NOISE) if drop is None else drop
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace("&", " and ").replace("’", "'").replace("`", "'")
    s = re.sub(r"(\d)[,\u202f\u00a0](?=\d{3}\b)", r"\1", s)          # 1,200 -> 1200
    s = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", r"\1", s)                  # 12th -> 12
    s = re.sub(r"'s\b", "", s)                                          # possessives
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)                            # keep decimal points only
    s = re.sub(r"[^\w\s.]", " ", s)
    toks = []
    for t in s.split():
        t = _MONTHS.get(t, t)
        if t in stop:
            continue
        toks.append(t)
    return toks


_ARTICLES = ("the", "a", "an")
_NUMLIKE = re.compile(r"-?\d+(?:\.\d+)?$")


def _gold_tokens(s: str) -> list[str]:
    """Gold tokens for containment: only a LEADING article is dropped.  Dropping stop/noise words
    everywhere silently deleted the discriminating token of short golds - 'Vitamin C' became
    {vitamin} ('c' is in the circa-noise set) and matched 'Vitamin D', 'La La Land' became {land}
    and matched 'Land of the Free', 'Hepatitis A' matched 'Hepatitis B'."""
    toks = _norm_tokens(s, drop=frozenset())
    while len(toks) > 1 and toks[0] in _ARTICLES:
        toks = toks[1:]
    return toks


_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
                "october", "november", "december"]


def _cand_tokens(s: str) -> list[str]:
    """Candidate tokens: nothing is dropped - the candidate only has to be a SUPERSET of the gold, so
    extra articles / hedge words are harmless, while dropping them can only manufacture matches.
    An unambiguous ISO date is additionally spelled out (1996-08-12 -> 1996 august 12) so a correct
    answer in ISO form is not scored wrong against a '12 August 1996' gold."""
    def _iso(m):
        mm = int(m.group(2))
        name = _MONTH_NAMES[mm - 1] if 1 <= mm <= 12 else m.group(2)
        return f"{m.group(1)} {name} {int(m.group(3))} {m.group(0)}"
    return _norm_tokens(_ISO_DATE.sub(_iso, s or ""), drop=frozenset())


def _numbers(s: str) -> list[float]:
    s = re.sub(r"(\d)[,\u202f\u00a0](?=\d{3}\b)", r"\1", s or "")
    out = []
    for m in _NUM.finditer(s):
        try:
            out.append(float(m.group(0)))
        except ValueError:
            pass
    return out


def _aliases(gold: str) -> list[str]:
    """Alias variants of a gold answer: the full string, the part outside parentheses, and
    ' or ' / '/' / ';' alternatives. The parenthetical alone is never an alias (it is a disambiguator)."""
    out: list[str] = []
    base = [gold.strip()]
    if "(" in gold and ")" in gold:
        outside = re.sub(r"\s*\([^)]*\)", "", gold).strip()
        if outside:
            base.append(outside)
    for b in base:
        for part in re.split(r"\s+or\s+|\s*/\s*|\s*;\s*", b):
            part = part.strip(" .")
            if part and _norm_tokens(part) and part not in out:
                out.append(part)
    return out or [gold.strip()]


def _contains(gold: str, cand: str, answer_type: str) -> bool:
    gtoks = _gold_tokens(gold)
    if not gtoks:
        return False
    ctoks = set(_cand_tokens(cand))
    if answer_type == "Number":
        gnums = _numbers(gold)
        if gnums:
            cnums = _numbers(cand)
            if not all(any(math.isclose(g, c, rel_tol=1e-9, abs_tol=1e-9) for c in cnums) for g in gnums):
                return False
            # the numbers agree; any non-numeric gold token ("1.5 million" vs "1.5 billion") must too
            return all(t in ctoks for t in gtoks if not _NUMLIKE.match(t))
    return all(t in ctoks for t in gtoks)


_SENTENCE_CUT = re.compile(r"(?<=\w{3})\.\s+(?=[A-Z(])")


def _trim_answer_span(span: str) -> str:
    """Keep only the first clause of an 'Answer: ...' span: cut at ' (' , ';' and at a sentence end that
    follows a word of >= 3 letters (so 'J. R. R. Tolkien', 'Dr. Who', 'St. Louis' and '3.14' survive), so a
    hedge or negation appended to the answer line ('Answer: X. (Not Y.)') cannot earn containment credit."""
    s = span
    for sep in (" (", ";", " - ", " — "):
        if sep in s:
            s = s.split(sep, 1)[0]
    m = _SENTENCE_CUT.search(s)
    if m:
        s = s[: m.start()]
    return s.strip(" .") or span


_FENCE_ONLY = re.compile(r"^[`~*_#>\-=.\s]+$")
# an uncommitted answer span: "X or Y", "X and Y" (two entities where one was asked for), "maybe X"
_HEDGE_WORD = re.compile(r"(?i)(?:\bor\b|\beither\b|\band\b|\bmaybe\b|\bpossibly\b|\bperhaps\b)")
# in a free-prose tail "and" is ordinary connective tissue, so only real alternation counts there
_HEDGE_WORD_TAIL = re.compile(r"(?i)(?:\bor\b|\beither\b|\bmaybe\b|\bpossibly\b|\bperhaps\b)")
_NEG_WORD = re.compile(r"(?i)\b(?:not|never|isn't|wasn't|aren't|rather than|instead of)\b")
# "6 to 8", "1990-1992": a range instead of the single number that was asked for
_RANGE = re.compile(r"(?i)\d\s*(?:to|through|-|–|—|and|or)\s*\d")


def _tail_span(visible: str) -> str:
    """The last line the model actually ends on (markdown fences / rules skipped).  Scoring the whole
    last 600 characters credits reasoning leakage: 'Radcliffe College is one candidate, Wellesley
    another ... I cannot give a confident answer' contains the gold and was scored correct."""
    lines = [ln.strip() for ln in visible.splitlines() if ln.strip()]
    while lines and _FENCE_ONLY.match(lines[-1]):
        lines.pop()
    return lines[-1][-600:] if lines else ""


def _score_simpleqa(item: dict, text: str) -> Verdict:
    expected = str(item["answer"])
    aliases = list(item.get("aliases") or _aliases(expected))
    atype = str(item.get("answer_type") or item.get("meta", {}).get("answer_type") or "Other")
    visible = (text or "").strip()
    if not visible:
        return Verdict.unparsed(expected, {"method": "none"}, ["empty"])
    span, method = common.extract_final_answer(visible, allow_last_integer=False)
    flags: list[str] = []
    if span is None:
        span = _tail_span(visible)
        method = "tail"
        flags.append("no_answer_phrase")
    span = _trim_answer_span(common.strip_math_delims(span).strip().strip("*").strip()) if method != "tail" else span
    gold_norm = expected.replace("&", " and ")
    hedge_rx = _HEDGE_WORD_TAIL if method == "tail" else _HEDGE_WORD
    gold_hedges = bool(hedge_rx.search(gold_norm)) or bool(_RANGE.search(gold_norm))
    gold_negates = bool(_NEG_WORD.search(expected))
    for alias in aliases:
        if not _contains(alias, span, atype):
            continue
        # a range where a single number was asked for ("6 to 8" against gold 8)
        if atype == "Number" and not gold_hedges and _RANGE.search(span):
            return Verdict(False, "wrong", 0.0, span[:200], expected, {"method": method, "alias": alias}, flags + ["hedged"])
        # the model named alternatives instead of committing ("X or Y"): the official SimpleQA grader
        # calls this NOT_ATTEMPTED, so it must not be scored correct here either
        if not gold_hedges and hedge_rx.search(span):
            return Verdict(False, "wrong", 0.0, span[:200], expected, {"method": method, "alias": alias}, flags + ["hedged"])
        # the gold only appears AFTER a negation ("Wellesley College, not Radcliffe College")
        neg = None if gold_negates else _NEG_WORD.search(span)
        if neg and not _contains(alias, span[: neg.start()], atype):
            return Verdict(False, "wrong", 0.0, span[:200], expected, {"method": method, "alias": alias}, flags + ["negated"])
        return Verdict(True, extracted=span[:200], expected=expected, detail={"method": method, "alias": alias}, flags=flags)
    if method == "tail":
        return Verdict.unparsed(expected, {"method": method}, flags + ["no_match"])
    return Verdict(False, extracted=span[:200], expected=expected, detail={"method": method}, flags=flags)


# ==================================================================================================
# hooks
# ==================================================================================================

def score(item: dict, response_text: str, meta: Optional[dict] = None) -> Verdict:
    if item.get("subfamily") == "simpleqa":
        return _score_simpleqa(item, response_text)
    return _score_mcq(item, response_text)


def mock_response(item: dict):
    if item.get("subfamily") == "simpleqa":
        return f"Answer: {item['answer']}"
    return f"Let me work through the options.\nThe answer is ({item['answer']})."


def aggregate(records: list[dict]) -> dict:
    scored = [r for r in records if r.get("status") not in ("error", "cancelled", "skipped")]
    out: dict = {"unparsed_rate": round(sum(1 for r in scored if r.get("status") == "unparsed") / len(scored), 4) if scored else None}
    methods: dict[str, int] = {}
    for r in scored:
        m = (r.get("detail") or {}).get("method")
        if m:
            methods[m] = methods.get(m, 0) + 1
    out["extraction_methods"] = dict(sorted(methods.items()))
    mcq = [r for r in scored if r.get("sub") == "mmlu_pro_hard"]
    if mcq:
        out["mmlu_pro_hard_weak_extraction"] = sum(1 for r in mcq if "weak_extraction" in (r.get("flags") or []))
    sq = [r for r in scored if r.get("sub") == "simpleqa"]
    if sq:
        out["simpleqa_no_answer_phrase"] = sum(1 for r in sq if "no_answer_phrase" in (r.get("flags") or []))
    return out
