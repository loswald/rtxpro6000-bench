"""
families/code.py - code generation, scored by EXECUTING the model's Python against hidden tests.

Three sub-families (dispatch order slowest first):

  lcb_easy_medium  LiveCodeBench code_generation_lite, newest release window (test6.jsonl, pinned revision):
                   easy + medium problems, stdin/stdout (AtCoder, Codeforces) and functional (LeetCode
                   `class Solution`) problems, public + a seeded sample of private tests (<= 24 per problem).
  humanevalplus    EvalPlus HumanEval+ (base + plus tests): a seeded sample from a curated pool of 60 tasks
                   whose plus tests / specification corners are the ones that still flip strong models
                   (HumanEval/32 is excluded: unpassable as shipped, see NOTES).
  mbppplus         EvalPlus MBPP+ (base + plus tests): a seeded sample of the non-trivial tasks
                   (canonical solution >= 3 lines).

Sources (all public, no token, fetched with common.fetch() into data_dir/raw/<source>/, sha256 recorded):
  * https://huggingface.co/datasets/evalplus/humanevalplus  test.jsonl (prompt, canonical_solution, entry_point, test)
  * https://datasets-server.huggingface.co/rows?dataset=evalplus/mbppplus  (the HF repo is parquet-only, which the
    standard library cannot read; the rows API serves the same `test` scripts with expected outputs as JSON)
  * https://huggingface.co/datasets/livecodebench/code_generation_lite  test6.jsonl @ LCB_REVISION (134 MB)

Scoring: the code is extracted from the last Python fence (or the whole text), assembled with the hidden
tests and executed with `python3 -I -X utf8` in a fresh temp dir, RLIMIT_AS / RLIMIT_CPU / RLIMIT_FSIZE set
inside the child, a 10 s wall timeout per test file, process group killed on timeout.  All tests must pass:
a failing / raising / timing-out run is 'wrong'; no extractable code is 'unparsed'.  run_eval.py already
calls score() from a thread pool, so the blocking subprocess work never stalls the asyncio loop.

EvalPlus test scripts import numpy; the sandbox rewrites `import numpy as np` to a small pure-Python shim
(`_evalplus_np.py`, allclose / testing.assert_allclose / ndarray) so scoring does not depend on numpy.
"""
from __future__ import annotations

import base64
import collections
import io
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zlib
from typing import Callable, Optional

import common
from common import DEFAULT_SEED, Verdict
from families import _base

NAME = "code"
DESCRIPTION = "HumanEval+ hard subset, MBPP+ sample, LiveCodeBench easy/medium (newest window); executed against hidden tests"
SUBFAMILIES = ["lcb_easy_medium", "humanevalplus", "mbppplus"]
PRIORITY = 20
DEFAULT_MAX_TOKENS = {"default": 2048, "reasoning": 8192}
ITEM_TIME_FALLBACK_S = 120.0
NOTES = [
    "code: HumanEval+ items are a seeded sample of a curated 60-task hard pool (not the whole benchmark); "
    "MBPP+ items are a seeded sample of tasks with >= 3-line canonical solutions",
    "code: HumanEval/32 (find_zero) is excluded: the HF test.jsonl assertion is malformed (unpacks a float) and even "
    "the canonical solution violates the find_zero oracle on 7/886 plus inputs, so the task is unpassable as shipped",
    "code: LCB items are easy+medium problems of the newest release window (test6.jsonl), scored on the public tests "
    "plus a seeded sample of private tests (<= 24 per problem, first failure stops) with a 10 s wall limit per test; "
    "this is not the official LiveCodeBench scorer (which runs every private test)",
    "code: sandbox = python3 -I -X utf8 in a temp dir with RLIMIT_AS 2 GiB / RLIMIT_CPU / RLIMIT_FSIZE 64 MB; "
    "no network isolation (the tests do not need network)",
]

# ---- sources ---------------------------------------------------------------------------------
HE_REPO = "evalplus/humanevalplus"
HE_FILE = "test.jsonl"
MBPP_ROWS_URL = ("https://datasets-server.huggingface.co/rows?dataset=evalplus%2Fmbppplus&config=default"
                 "&split=test&offset={offset}&length=100")
MBPP_N_ROWS = 378
LCB_REPO = "livecodebench/code_generation_lite"
LCB_FILE = "test6.jsonl"
LCB_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"   # pinned HF commit of the v6 release window
LCB_MAX_FETCH_BYTES = 300 * 1024 * 1024

# ---- item selection --------------------------------------------------------------------------
# HumanEval tasks whose plus tests / specification corners (rounding away from zero, "ends with a letter",
# nested/cyclic decoders, polynomial roots, digit-sum sign rules, ...) still separate strong from weak models.
HE_HARD_POOL = [10, 26, 33, 38, 39, 41, 47, 50, 64, 65, 70, 75, 76, 83, 91, 95, 99, 103, 105,
                108, 109, 113, 115, 116, 119, 120, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 137, 138,
                139, 140, 141, 142, 144, 145, 146, 147, 148, 149, 150, 151, 154, 155, 156, 158, 159, 160, 161, 162, 163]
N_ITEMS = {"default": {"humanevalplus": 25, "mbppplus": 20, "lcb_easy": 8, "lcb_medium": 22},
           "full": {"humanevalplus": 50, "mbppplus": 40, "lcb_easy": 20, "lcb_medium": 50}}
MBPP_MIN_LINES = 3
LCB_MAX_TESTS = 24
LCB_MAX_TEST_BYTES = 1_000_000       # cumulative input+output bytes kept per problem
LCB_MAX_SINGLE_TEST_BYTES = 300_000
LCB_MIN_TESTS = 3

# ---- sandbox ---------------------------------------------------------------------------------
TEST_TIMEOUT_S = 10.0                # wall limit per test file (spec)
ITEM_BUDGET_S = 45.0                 # total wall budget per item across its test files
MEM_LIMIT_BYTES = 2 * 1024 ** 3
STDOUT_READ_LIMIT = 8 * 1024 * 1024
STDERR_TAIL = 1500

# ---- prompts ---------------------------------------------------------------------------------
HE_INSTRUCTION = ("Complete the following Python function. Reply with the complete implementation of `{entry}` "
                  "(repeat the signature and any imports it needs) in a single ```python code block and nothing else.\n\n"
                  "```python\n{prompt}\n```")
MBPP_INSTRUCTION = ("Please provide a self-contained Python script that solves the following problem in a single "
                    "```python code block. The function must be named `{entry}` and satisfy the example assertion.\n\n"
                    "{prompt}")
LCB_HEADER = ("You are an expert Python programmer. You will be given a question (problem specification) and will "
              "generate a correct Python program that matches the specification and passes all tests.\n\n"
              "### Question:\n{content}\n\n")
LCB_STDIN_FORMAT = ("### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not "
                    "directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that "
                    "when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
                    "```python\n# YOUR CODE HERE\n```\n\n### Answer: (use the provided format with backticks)\n\n")
LCB_FUNC_FORMAT = ("### Format: You will use the following starter code to write the solution to the problem and "
                   "enclose your code within delimiters.\n```python\n{starter}\n```\n\n"
                   "### Answer: (use the provided format with backticks)\n\n")
LCB_PRELUDE = ("from typing import *\nfrom collections import *\nfrom itertools import *\nfrom functools import *\n"
               "from heapq import *\nfrom bisect import *\nimport sys, math, string, re, collections, itertools, "
               "functools, heapq, bisect, random\nsys.setrecursionlimit(20000)\n")

# ======================================================================================
# prepare()
# ======================================================================================

def _norm_ws(s: str) -> str:
    return " ".join((s or "").split())


def _unique_markers(texts: list[str]) -> list[str]:
    """Whitespace-normalised prefixes of the problem texts, long enough to be unique (mock_server markers)."""
    for n in (200, 320, 600, 1200, 4000, None):
        marks = [_norm_ws(t)[:n] if n else _norm_ws(t) for t in texts]
        if len(set(marks)) == len(marks) and all(len(m) >= 24 for m in marks):
            return marks
    return [_norm_ws(t) for t in texts]


class _StrOnlyUnpickler(pickle.Unpickler):
    """LCB private tests are a pickled str (zlib + base64); a pickled str needs no globals, so forbid them all."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"forbidden global in test payload: {module}.{name}")


def _decode_lcb_private(s: str) -> list:
    if not s:
        return []
    raw = zlib.decompress(base64.b64decode(s.encode("utf-8")))
    obj = _StrOnlyUnpickler(io.BytesIO(raw)).load()
    if not isinstance(obj, str):
        raise ValueError("unexpected private_test_cases payload type " + type(obj).__name__)
    return json.loads(obj)


def _fetch_hf_api(repo: str, dest: str, refresh: bool, log) -> Optional[dict]:
    try:
        return common.fetch_json(f"{common.HF_ENDPOINT}/api/datasets/{repo}?blobs=true", dest, refresh=refresh, log=log)
    except Exception as e:  # the API is a convenience (revision pinning / size check); never fatal
        log(f"[code] HF API for {repo} unavailable: {e}")
        return None


def _entry_from_test(test: str, test_list: list) -> Optional[str]:
    hits = re.findall(r"assertion\(\s*([A-Za-z_]\w*)\s*\(\*inp", test)
    if hits:
        return hits[-1]
    if test_list:
        m = re.search(r"assert\s+\(?\s*(?:set|sorted|list|tuple|str|abs|round|len)?\s*\(?\s*([A-Za-z_]\w*)\s*\(", test_list[0])
        if m and m.group(1) not in ("set", "sorted", "list", "tuple", "str", "abs", "round", "len", "math"):
            return m.group(1)
    return None


def _take(pool: list, n: int, rng, sub: str, allow_short: bool, log) -> list:
    if len(pool) < n:
        if not allow_short:
            raise common.ShortPool(f"{sub}: pool {len(pool)} < requested {n}")
        log(f"[code] {sub}: pool {len(pool)} < {n}, taking all (allow_short)")
        n = len(pool)
    return rng.sample(pool, n)


def prepare(data_dir: str, seed: int = DEFAULT_SEED, profile: str = "default", refresh: bool = False,
            log: Callable[[str], None] = print, allow_short: bool = False,
            n_humanevalplus: Optional[int] = None, n_mbppplus: Optional[int] = None,
            n_lcb_easy: Optional[int] = None, n_lcb_medium: Optional[int] = None,
            lcb_after: Optional[str] = None, lcb_skip: bool = False, he_pool: str = "hard",
            mbpp_min_lines: int = MBPP_MIN_LINES, **opts) -> dict:
    sizes = N_ITEMS["full" if profile == "full" else "default"]
    n_he = int(n_humanevalplus) if n_humanevalplus is not None else sizes["humanevalplus"]
    n_mbpp = int(n_mbppplus) if n_mbppplus is not None else sizes["mbppplus"]
    n_easy = int(n_lcb_easy) if n_lcb_easy is not None else sizes["lcb_easy"]
    n_medium = int(n_lcb_medium) if n_lcb_medium is not None else sizes["lcb_medium"]
    raw = os.path.join(data_dir, "raw")
    sources: dict = {}
    pools: dict = {}
    notes: list[str] = list(NOTES)
    items: list[dict] = []

    # ---- HumanEval+ ------------------------------------------------------------------
    api = _fetch_hf_api(HE_REPO, os.path.join(raw, "humanevalplus", "api.json"), refresh, log)
    he_rev = (api or {}).get("sha") or "main"
    he_url = common.hf_url(HE_REPO, HE_FILE, revision=he_rev)
    he_path = os.path.join(raw, "humanevalplus", HE_FILE)
    info = common.fetch(he_url, he_path, refresh=refresh, log=log)
    sources["humanevalplus"] = {"url": he_url, "sha256": info["sha256"], "bytes": info["bytes"], "revision": he_rev}
    he_rows = {r["task_id"]: r for r in common.read_jsonl(he_path)}
    if he_pool == "hard":
        pool_ids = [f"HumanEval/{i}" for i in HE_HARD_POOL if f"HumanEval/{i}" in he_rows]
    else:
        pool_ids = sorted(he_rows, key=lambda t: int(t.split("/")[-1]))
    pools["humanevalplus"] = len(pool_ids)
    rng = common.seeded_rng(seed, NAME, "humanevalplus")
    picked = _take(pool_ids, n_he, rng, "humanevalplus", allow_short, log)
    markers = _unique_markers([he_rows[t]["prompt"] for t in picked])
    for order, (tid, marker) in enumerate(zip(picked, markers)):
        r = he_rows[tid]
        num = int(tid.split("/")[-1])
        items.append({"id": f"heplus-{num:03d}", "family": NAME, "subfamily": "humanevalplus", "order": order,
                      "messages": [{"role": "user", "content": HE_INSTRUCTION.format(entry=r["entry_point"],
                                                                                     prompt=r["prompt"].rstrip())}],
                      "entry_point": r["entry_point"], "prompt": r["prompt"], "canonical": r["canonical_solution"],
                      "test": r["test"], "mock_marker": marker,
                      "meta": {"source": HE_REPO, "source_id": tid, "revision": he_rev}})

    # ---- MBPP+ (datasets-server rows: the HF repo is parquet-only) -------------------
    mbpp_rows: list[dict] = []
    for offset in range(0, MBPP_N_ROWS, 100):
        url = MBPP_ROWS_URL.format(offset=offset)
        dest = os.path.join(raw, "mbppplus", f"rows_{offset:04d}.json")
        page = common.fetch_json(url, dest, refresh=refresh, log=log)
        sources[f"mbppplus/rows_{offset:04d}"] = {"url": url, "sha256": common.sha256_file(dest),
                                                 "bytes": os.path.getsize(dest)}
        for row in page.get("rows", []):
            if row.get("truncated_cells"):
                raise IOError(f"mbppplus rows page {offset}: truncated cells {row['truncated_cells']} (row {row.get('row_idx')})")
            mbpp_rows.append(row["row"])
    if len({r["task_id"] for r in mbpp_rows}) != len(mbpp_rows):
        raise IOError("mbppplus rows: duplicate task ids across pages")
    mbpp_rows.sort(key=lambda r: int(r["task_id"]))
    mbpp_pool = []
    for r in mbpp_rows:
        entry = _entry_from_test(r["test"], r.get("test_list") or [])
        if entry is None:
            log(f"[code] mbppplus task {r['task_id']}: cannot derive entry point, skipped")
            continue
        n_lines = len([ln for ln in r["code"].splitlines() if ln.strip()])
        if n_lines < int(mbpp_min_lines):
            continue
        r["_entry"] = entry
        mbpp_pool.append(r)
    pools["mbppplus"] = len(mbpp_pool)
    rng = common.seeded_rng(seed, NAME, "mbppplus")
    picked_m = _take(mbpp_pool, n_mbpp, rng, "mbppplus", allow_short, log)
    prompts = [r["prompt"].strip() + "\n" + (r.get("test_list") or [""])[0] for r in picked_m]
    markers = _unique_markers(prompts)
    for order, (r, ptxt, marker) in enumerate(zip(picked_m, prompts, markers)):
        tid = int(r["task_id"])
        items.append({"id": f"mbppplus-{tid:03d}", "family": NAME, "subfamily": "mbppplus", "order": order,
                      "messages": [{"role": "user", "content": MBPP_INSTRUCTION.format(entry=r["_entry"], prompt=ptxt)}],
                      "entry_point": r["_entry"], "prompt": ptxt, "canonical": r["code"],
                      "test_imports": list(r.get("test_imports") or []), "test": r["test"], "mock_marker": marker,
                      "meta": {"source": HE_REPO.replace("humanevalplus", "mbppplus"), "source_id": f"Mbpp/{tid}",
                               "via": "datasets-server rows"}})

    # ---- LiveCodeBench easy/medium, newest window ------------------------------------
    lcb_extra: dict = {"file": LCB_FILE, "revision": LCB_REVISION, "hard_available": False,
                       "difficulties": ["easy", "medium"], "after": lcb_after}
    n_lcb = {"easy": 0, "medium": 0}
    if lcb_skip:
        notes.append("code: lcb_easy_medium skipped (lcb_skip)")
    else:
        lcb_api = _fetch_hf_api(LCB_REPO, os.path.join(raw, "livecodebench", "api.json"), refresh, log)
        skip_reason = None
        if lcb_api is not None:
            if lcb_api.get("gated"):
                skip_reason = f"{LCB_REPO} is gated"
            size = next((s.get("size") for s in lcb_api.get("siblings", []) if s.get("rfilename") == LCB_FILE), None)
            if size and size > LCB_MAX_FETCH_BYTES:
                skip_reason = f"{LCB_FILE} is {size / 1e6:.0f} MB > {LCB_MAX_FETCH_BYTES / 1e6:.0f} MB"
        lcb_url = common.hf_url(LCB_REPO, LCB_FILE, revision=LCB_REVISION)
        lcb_path = os.path.join(raw, "livecodebench", LCB_FILE)
        if skip_reason is None:
            try:
                info = common.fetch(lcb_url, lcb_path, refresh=refresh, log=log)
            except IOError as e:
                if any(c in str(e) for c in ("401", "403")):
                    skip_reason = f"{LCB_REPO} not fetchable without a token: {e}"
                else:
                    raise
        if skip_reason is not None:
            log(f"[code] lcb_easy_medium skipped: {skip_reason}")
            notes.append(f"code: lcb_easy_medium skipped: {skip_reason}")
            lcb_extra["skipped"] = skip_reason
        else:
            sources["livecodebench/test6"] = {"url": lcb_url, "sha256": info["sha256"], "bytes": info["bytes"],
                                               "revision": LCB_REVISION}
            pool_by_diff: dict[str, list[dict]] = {"easy": [], "medium": []}
            n_seen = collections.Counter()
            with open(lcb_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    diff = row.get("difficulty")
                    n_seen[diff] += 1
                    if diff not in pool_by_diff:
                        continue
                    if lcb_after and str(row.get("contest_date", "")) < str(lcb_after):
                        continue
                    starter = row.get("starter_code") or ""
                    if re.search(r"\b(ListNode|TreeNode)\b", starter):
                        continue
                    try:
                        public = json.loads(row.get("public_test_cases") or "[]")
                        private = _decode_lcb_private(row.get("private_test_cases") or "")
                    except Exception as e:
                        log(f"[code] lcb {row.get('question_id')}: cannot decode tests ({e}), skipped")
                        continue
                    tests = public + private
                    kinds = {t.get("testtype") for t in tests}
                    if not tests or len(kinds) != 1:
                        continue
                    mode = "functional" if kinds == {"functional"} else "stdin"
                    meta = {}
                    try:
                        meta = json.loads(row.get("metadata") or "{}") or {}
                    except ValueError:
                        pass
                    func_name = meta.get("func_name")
                    if mode == "functional" and (not func_name or "class Solution" not in starter):
                        continue
                    qid = str(row["question_id"])
                    rng_t = common.seeded_rng(seed, NAME, "lcb_tests", row.get("platform"), qid)
                    sel = list(public)
                    if private:
                        k = min(len(private), LCB_MAX_TESTS)
                        sel += [private[i] for i in sorted(rng_t.sample(range(len(private)), k))]
                    kept, total = [], 0
                    for t in sel:
                        sz = len(t.get("input", "")) + len(t.get("output", ""))
                        if sz > LCB_MAX_SINGLE_TEST_BYTES:
                            continue
                        if len(kept) >= LCB_MAX_TESTS or total + sz > LCB_MAX_TEST_BYTES:
                            break
                        kept.append({"input": t["input"], "output": t["output"]})
                        total += sz
                    if len(kept) < LCB_MIN_TESTS:
                        continue
                    pool_by_diff[diff].append({
                        "qid": qid, "platform": row.get("platform"), "difficulty": diff, "mode": mode,
                        "func_name": func_name, "starter": starter, "content": row.get("question_content") or "",
                        "title": row.get("question_title"), "contest_date": row.get("contest_date"),
                        "contest_id": row.get("contest_id"), "tests": kept, "n_tests_total": len(tests)})
            for d in pool_by_diff:
                pool_by_diff[d].sort(key=lambda p: (p["platform"] or "", p["qid"]))
            pools["lcb_easy_medium"] = f"easy {len(pool_by_diff['easy'])} / medium {len(pool_by_diff['medium'])}"
            lcb_extra["n_seen"] = dict(sorted(n_seen.items(), key=lambda kv: str(kv[0])))
            rng = common.seeded_rng(seed, NAME, "lcb_easy_medium")
            chosen = _take(pool_by_diff["medium"], n_medium, rng, "lcb_medium", allow_short, log)
            chosen += _take(pool_by_diff["easy"], n_easy, rng, "lcb_easy", allow_short, log)
            rng.shuffle(chosen)
            markers = _unique_markers([p["content"] for p in chosen])
            dates = [p["contest_date"] for p in chosen if p.get("contest_date")]
            lcb_extra.update({"after": lcb_after or (min(dates)[:10] if dates else None),
                              "before": max(dates)[:10] if dates else None})
            for order, (p, marker) in enumerate(zip(chosen, markers)):
                n_lcb[p["difficulty"]] += 1
                fmt = LCB_FUNC_FORMAT.format(starter=p["starter"].rstrip()) if p["mode"] == "functional" else LCB_STDIN_FORMAT
                pid = re.sub(r"[^A-Za-z0-9_.-]", "-", f"{p['platform']}-{p['qid']}")
                items.append({"id": f"lcb-{pid}", "family": NAME, "subfamily": "lcb_easy_medium", "order": order,
                              "messages": [{"role": "user", "content": LCB_HEADER.format(content=p["content"].strip()) + fmt}],
                              "mode": p["mode"], "func_name": p["func_name"], "starter": p["starter"],
                              "tests": p["tests"], "n_tests_total": p["n_tests_total"], "mock_marker": marker,
                              "meta": {"source": LCB_REPO, "file": LCB_FILE, "revision": LCB_REVISION,
                                       "question_id": p["qid"], "platform": p["platform"], "difficulty": p["difficulty"],
                                       "contest_date": p["contest_date"], "contest_id": p["contest_id"], "title": p["title"]}})

    ids = [it["id"] for it in items]
    if len(set(ids)) != len(ids):
        raise ValueError("code: duplicate item ids")
    if len(set(it["mock_marker"] for it in items)) != len(items):
        raise ValueError("code: mock markers collide across sub-families")
    common.write_jsonl(_base.items_path(NAME, data_dir), items)
    counts = {"lcb_easy_medium": n_lcb["easy"] + n_lcb["medium"], "humanevalplus": len(picked), "mbppplus": len(picked_m)}
    lcb_extra["counts"] = dict(n_lcb)
    return {"file": f"items/{NAME}.jsonl", "counts": counts, "sources": sources, "pools": pools, "notes": notes,
            "manifest_extra": {"lcb_window": lcb_extra}}


# ======================================================================================
# code extraction
# ======================================================================================

_FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+#.-]*)[^\n]*\n(.*?)(?:(```)|\Z)", re.S)
_PY_LANGS = {"", "python", "py", "python3", "py3", "python2"}
_CODE_LINE_RE = re.compile(r"^(?:from\s+[\w.]+\s+import\b|import\s+\w|def\s+\w+\s*\(|class\s+\w+\b|@\w|#|"
                           r"if\s+__name__|print\s*\(|input\s*\(|[A-Za-z_]\w*(?:\[[^\]]*\])?\s*=[^=])")


def _dedent_block(body: str, indent: str) -> str:
    """Remove the fence's own markdown indentation from every line (a fence inside a list item is indented);
    left alone when any non-blank line does not carry it, so real body-only completions never shift."""
    if not indent:
        return body
    out = []
    for ln in body.split("\n"):
        if ln.startswith(indent):
            out.append(ln[len(indent):])
        elif not ln.strip():
            out.append("")
        else:
            return body
    return "\n".join(out)


def _fences(text: str) -> list[tuple[str, str, bool]]:
    """[(language, body, closed)] for every ``` fence; an unterminated final fence runs to the end of the text.
    A fence that starts on its own (indented) line is dedented by that indentation."""
    out = []
    for m in _FENCE_RE.finditer(text):
        pre = text[text.rfind("\n", 0, m.start()) + 1:m.start()]
        indent = pre if (not pre or not pre.strip()) else ""
        out.append((m.group(1).strip().lower(), _dedent_block(m.group(2), indent), m.group(3) is not None))
    return out


def _pick_fence(fences: list[tuple[str, bool]], prefer: list[str]) -> Optional[tuple[str, bool]]:
    """The last fence matching the first pattern that matches anything; else the last fence."""
    for pat in prefer:
        rx = re.compile(pat, re.M)
        hits = [f for f in fences if rx.search(f[0])]
        if hits:
            return hits[-1]
    return fences[-1] if fences else None


def _strip_prose(code: str) -> str:
    """Drop leading lines that are clearly prose (unfenced replies: 'Here is the solution:')."""
    lines = code.split("\n")
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s and _CODE_LINE_RE.match(s):
            return "\n".join(lines[i:])
    return code


def _trim_to_compilable(code: str, prefix: str = "") -> tuple[str, Optional[str]]:
    """Cut trailing prose after the code: retry compile(prefix + code) up to 8 times, cutting the code at a
    column-0 offending line.  `prefix` (the HumanEval prompt for body-only completions) is never cut."""
    pre_lines = prefix.count("\n")
    for _ in range(8):
        try:
            compile(prefix + code, "<candidate>", "exec")
            return code, None
        except SyntaxError as e:
            ln = (e.lineno or 0) - pre_lines
            lines = code.split("\n")
            if 1 < ln <= len(lines) and not lines[ln - 1][:1].isspace():
                cut = "\n".join(lines[:ln - 1]).rstrip()
                if cut.strip():
                    code = cut
                    continue
            return code, f"{type(e).__name__}: {e.msg} (line {max(ln, 1)})"
        except (ValueError, RecursionError) as e:
            return code, f"{type(e).__name__}: {e}"
    return code, "SyntaxError: could not trim to a compilable block"


def _is_body_only(item: dict, code: str) -> bool:
    """HumanEval+ completion-style reply: the code starts indented (function body without the signature)."""
    if item.get("subfamily") != "humanevalplus":
        return False
    first = next((ln for ln in code.split("\n") if ln.strip()), "")
    return first[:1].isspace()


def extract_code(text: str, item: dict) -> tuple[Optional[str], str]:
    """(code, method) - the last Python fence that looks like the solution, else the whole text if code-like."""
    text = (text or "").rstrip()
    text = re.sub(r"\A(?:[ \t]*\n)+", "", text)     # drop leading blank lines but KEEP indentation (body-only replies)
    if not text.strip():
        return None, "empty"
    sub = item.get("subfamily", "")
    if sub == "lcb_easy_medium":
        prefer = [r"^\s*class\s+Solution\b"] if item.get("mode") == "functional" else []
        prefer += [r"^\s*def\s+\w+|^\s*class\s+\w+|input\s*\(|stdin|^\s*print\s*\("]
    else:
        entry = re.escape(item.get("entry_point") or "")
        prefer = [rf"^\s*def\s+{entry}\s*\("] if entry else []
        prefer += [r"^\s*def\s+\w+\s*\("]
    fences = _fences(text)
    py = [(b, closed) for lang, b, closed in fences if lang in _PY_LANGS]
    if py:
        body, closed = _pick_fence(py, prefer)
        code = body.strip("\n")
        return (code, "fence" if closed else "fence_unterminated") if code.strip() else (None, "empty_fence")
    if fences:  # only non-python fences: still try the last one that looks like python
        body, closed = _pick_fence([(b, c) for _, b, c in fences], prefer)
        if body and re.search(r"^\s*(def|class|import|from)\s", body, re.M):
            return body.strip("\n"), "fence_other_lang"
    body = _strip_prose(text)
    if re.search(r"^\s*(def|class|import|from|print|for|while|if|try|with|return)\b|input\s*\(", body, re.M):
        return body.strip("\n"), "whole"
    if _is_body_only(item, body) and "\n" not in body.strip() and re.match(r"\s+[A-Za-z_]", body):
        return body.strip("\n"), "whole_body"      # a one-line indented completion such as '    return x + 1'
    return None, "no_code"


# ======================================================================================
# sandbox
# ======================================================================================

_NP_SHIM = r'''"""numpy stand-in for EvalPlus test scripts (allclose / testing.assert_allclose / ndarray)."""
import math


class ndarray:  # never instantiated; isinstance() checks in is_floats() must not fail
    pass


float64 = float
float32 = float
nan = float("nan")
inf = float("inf")


def _flat(x):
    if isinstance(x, (list, tuple)):
        for i in x:
            yield from _flat(i)
    else:
        yield x


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    fa, fb = list(_flat(a)), list(_flat(b))
    if len(fa) != len(fb):
        return False
    for x, y in zip(fa, fb):
        try:
            x, y = float(x), float(y)
        except (TypeError, ValueError):
            if x != y:
                return False
            continue
        if math.isnan(x) or math.isnan(y):
            if equal_nan and math.isnan(x) and math.isnan(y):
                continue
            return False
        if math.isinf(x) or math.isinf(y):
            if x != y:
                return False
            continue
        if abs(x - y) > atol + rtol * abs(y):
            return False
    return True


def isclose(a, b, rtol=1e-05, atol=1e-08):
    return allclose(a, b, rtol=rtol, atol=atol)


def array(x, *a, **k):
    return x


class testing:
    @staticmethod
    def assert_allclose(actual, desired, rtol=1e-07, atol=0, equal_nan=True, err_msg="", verbose=True):
        if not allclose(actual, desired, rtol=rtol, atol=atol, equal_nan=equal_nan):
            raise AssertionError(f"Not equal to tolerance rtol={rtol}, atol={atol}: {actual!r} vs {desired!r} {err_msg}")
'''

_DRIVER_SRC = r'''import json, os, sys
mode, mem, cpu = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
except Exception:
    pass
sys.path.insert(0, os.getcwd())
sys.setrecursionlimit(20000)
with open("sol.py", encoding="utf-8") as f:
    src = f.read()
code = compile(src, "sol.py", "exec")
DONE = "_evalsuite_done"
if mode == "stdin":
    exec(code, {"__name__": "__main__", "__builtins__": __builtins__})
elif mode == "evalplus":
    exec(code, {"__name__": "__evalsuite__", "__builtins__": __builtins__})
    open(DONE, "w").close()      # proof the hidden tests actually ran (sys.exit/os._exit cannot forge it)
elif mode == "functional":
    idx, func = int(sys.argv[4]), sys.argv[5]
    with open("tests.json", encoding="utf-8") as f:
        t = json.load(f)[idx]
    g = {"__name__": "__evalsuite__", "__builtins__": __builtins__}
    exec(code, g)
    Solution = g["Solution"]
    args = [json.loads(line) for line in t["input"].split("\n")] if t["input"].strip() else []
    got = getattr(Solution(), func)(*args)
    exp = json.loads(t["output"])

    def norm(o):
        if isinstance(o, (list, tuple)):
            return [norm(v) for v in o]
        if isinstance(o, dict):
            return {str(k): norm(v) for k, v in o.items()}
        return o

    def eq(a, b):
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return len(a) == len(b) and all(eq(x, y) for x, y in zip(a, b))
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(eq(a[k], b[k]) for k in a)
        if isinstance(a, bool) or isinstance(b, bool):
            return a == b
        if isinstance(a, int) and isinstance(b, int):
            return a == b          # integer answers are exact: no relative tolerance on big ints
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a == b or abs(a - b) <= 1e-6 + 1e-6 * abs(b)
        return a == b

    got_n, exp_n = norm(got), norm(exp)
    if not eq(got_n, exp_n):
        sys.stderr.write("MISMATCH got=%r exp=%r\n" % (repr(got_n)[:400], repr(exp_n)[:400]))
        sys.exit(3)
    open(DONE, "w").close()
'''


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _python() -> str:
    return sys.executable or "python3"


def _run_process(tmpdir: str, argv: list[str], stdin_bytes: Optional[bytes], timeout: float,
                 capture_stdout: bool) -> dict:
    """One sandboxed execution: python3 -I -X utf8 _run.py <argv...> in tmpdir; stdout to a file when asked."""
    out_path = os.path.join(tmpdir, "stdout.txt")
    err_path = os.path.join(tmpdir, "stderr.txt")
    done_path = os.path.join(tmpdir, DONE_SENTINEL)
    try:
        os.remove(done_path)
    except OSError:
        pass
    env = {"PATH": os.environ.get("PATH", ""), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "HOME": tmpdir, "TMPDIR": tmpdir}
    t0 = time.monotonic()
    timed_out = False
    with open(out_path, "wb") as out_f, open(err_path, "wb") as err_f:
        popen_kw = dict(cwd=tmpdir, env=env, stdin=subprocess.PIPE, stdout=out_f if capture_stdout else subprocess.DEVNULL,
                        stderr=err_f)
        if os.name == "posix":
            popen_kw["start_new_session"] = True
        proc = subprocess.Popen([_python(), "-I", "-X", "utf8", "_run.py"] + argv, **popen_kw)
        try:
            proc.communicate(input=stdin_bytes if stdin_bytes is not None else b"", timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                if os.name == "posix":
                    os.killpg(proc.pid, 9)
                else:
                    proc.kill()
            except OSError:
                pass
            try:
                proc.communicate(timeout=5)
            except Exception:
                pass
        except (BrokenPipeError, OSError):
            # the child exited before reading all of stdin
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.kill()
    elapsed = time.monotonic() - t0
    stdout = ""
    if capture_stdout:
        with open(out_path, "rb") as f:
            stdout = f.read(STDOUT_READ_LIMIT).decode("utf-8", "replace")
    with open(err_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - STDERR_TAIL * 2))
        stderr = f.read().decode("utf-8", "replace")[-STDERR_TAIL:]
    return {"rc": proc.returncode, "timed_out": timed_out, "stdout": stdout, "stderr": stderr, "elapsed": round(elapsed, 3)}


def _err_kind(res: dict) -> str:
    if res["timed_out"]:
        return "timeout"
    err = res["stderr"]
    if "AssertionError" in err:
        return "assertion"
    if res["rc"] == 3 and "MISMATCH" in err:
        return "wrong_output"
    if res["rc"] and res["rc"] < 0 or res["rc"] in (137, -9):
        return "killed"
    m = re.findall(r"^(\w+(?:Error|Exception|Exit|Interrupt))\b", err, re.M)
    return m[-1] if m else ("exception" if res["rc"] else "ok")


def _stdout_matches(got: str, exp: str) -> bool:
    g = [ln.rstrip() for ln in got.strip().splitlines()]
    e = [ln.rstrip() for ln in exp.strip().splitlines()]
    if g == e:
        return True
    gt, et = got.split(), exp.split()
    if gt == et:
        return True
    if len(gt) != len(et):
        return False
    for a, b in zip(gt, et):
        if a == b:
            continue
        try:
            fa, fb = float(a), float(b)
        except ValueError:
            return False
        if math.isnan(fa) or math.isnan(fb) or not math.isclose(fa, fb, rel_tol=1e-6, abs_tol=1e-6):
            return False
    return True


def _assemble_evalplus(item: dict, code: str) -> str:
    """Candidate + hidden tests as one module.  HumanEval+: the prompt (imports, helper functions, a
    docstring-only stub of the entry point) is prepended - a body-only completion becomes the completed
    function, a full function simply overrides the stub.  MBPP+: the test imports are prepended."""
    sub = item.get("subfamily")
    parts: list[str] = []
    if sub == "humanevalplus":
        parts.append(item.get("prompt", "").rstrip("\n"))
        first = next((ln for ln in code.split("\n") if ln.strip()), "")
        parts.append(code if first[:1].isspace() else "\n" + code)
    else:
        parts.extend(item.get("test_imports") or [])
        parts.append(code)
    test = item["test"]
    test = re.sub(r"^\s*import numpy as np\s*$", "import _evalplus_np as np", test, count=1, flags=re.M)
    parts.append("\n# ---- hidden tests ----")
    parts.append(test.rstrip("\n"))
    if sub == "humanevalplus":
        parts.append(f"\ncheck({item['entry_point']})")
    return "\n".join(parts) + "\n"


def _prepare_tmpdir() -> str:
    tmpdir = tempfile.mkdtemp(prefix="evalsuite_code_")
    _write(os.path.join(tmpdir, "_run.py"), _DRIVER_SRC)
    _write(os.path.join(tmpdir, "_evalplus_np.py"), _NP_SHIM)
    return tmpdir


def run_candidate(item: dict, code: str) -> dict:
    """Execute the candidate against the item's tests. -> {passed, n_tests, n_passed, kind, stderr, elapsed, test_index}"""
    sub = item.get("subfamily")
    cpu = int(TEST_TIMEOUT_S) + 1
    base = [str(MEM_LIMIT_BYTES), str(cpu)]
    tmpdir = _prepare_tmpdir()
    t_item = time.monotonic()
    try:
        if sub in ("humanevalplus", "mbppplus"):
            _write(os.path.join(tmpdir, "sol.py"), _assemble_evalplus(item, code))
            res = _run_process(tmpdir, ["evalplus"] + base, b"", TEST_TIMEOUT_S, capture_stdout=False)
            ok = res["rc"] == 0 and not res["timed_out"]
            return {"passed": ok, "n_tests": 1, "n_passed": int(ok), "kind": "ok" if ok else _err_kind(res),
                    "stderr": res["stderr"], "elapsed": res["elapsed"], "test_index": None if ok else 0}
        # ---- LiveCodeBench
        tests = item.get("tests") or []
        mode = item.get("mode") or "stdin"
        prog = LCB_PRELUDE + "\n" + code + "\n"
        _write(os.path.join(tmpdir, "sol.py"), prog)
        if mode == "functional":
            _write(os.path.join(tmpdir, "tests.json"), json.dumps(tests))
        n_passed = 0
        elapsed_total = 0.0
        for i, t in enumerate(tests):
            if time.monotonic() - t_item > ITEM_BUDGET_S:
                return {"passed": False, "n_tests": len(tests), "n_passed": n_passed, "kind": "budget_exceeded",
                        "stderr": "", "elapsed": round(elapsed_total, 3), "test_index": i}
            if mode == "functional":
                res = _run_process(tmpdir, ["functional"] + base + [str(i), str(item.get("func_name"))], b"",
                                   TEST_TIMEOUT_S, capture_stdout=False)
                ok = res["rc"] == 0 and not res["timed_out"]
            else:
                res = _run_process(tmpdir, ["stdin"] + base, t["input"].encode("utf-8"), TEST_TIMEOUT_S, capture_stdout=True)
                ok = res["rc"] == 0 and not res["timed_out"] and _stdout_matches(res["stdout"], t["output"])
                if not ok and res["rc"] == 0 and not res["timed_out"]:
                    res["stderr"] = (res["stderr"] + f"\nMISMATCH got={res['stdout'][:300]!r} exp={t['output'][:300]!r}").strip()
            elapsed_total += res["elapsed"]
            if not ok:
                kind = _err_kind(res)
                if kind == "ok":
                    kind = "wrong_output"
                return {"passed": False, "n_tests": len(tests), "n_passed": n_passed, "kind": kind,
                        "stderr": res["stderr"], "elapsed": round(elapsed_total, 3), "test_index": i}
            n_passed += 1
        return {"passed": True, "n_tests": len(tests), "n_passed": n_passed, "kind": "ok", "stderr": "",
                "elapsed": round(elapsed_total, 3), "test_index": None}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ======================================================================================
# score()
# ======================================================================================

def _expected(item: dict) -> str:
    sub = item.get("subfamily")
    if sub == "lcb_easy_medium":
        return f"pass {len(item.get('tests') or [])} tests ({item.get('mode')})"
    return f"{item.get('entry_point')} passes base+plus tests"


def score(item: dict, response_text: str, meta: Optional[dict] = None) -> Verdict:
    expected = _expected(item)
    code, method = extract_code(response_text, item)
    if code is None:
        return Verdict.unparsed(expected, {"method": method}, ["no_code"])
    prefix = item.get("prompt", "") if _is_body_only(item, code) else ""
    code, syntax_err = _trim_to_compilable(code, prefix)
    extracted = f"{len(code)} chars via {method}" + (" (body only)" if prefix else "")
    if syntax_err is not None:
        return Verdict(False, "wrong", 0.0, extracted, expected, {"method": method, "kind": "syntax_error", "error": syntax_err},
                       ["syntax_error"])
    res = run_candidate(item, code)
    detail = {"method": method, "mode": item.get("mode", "evalplus"), "kind": res["kind"], "n_tests": res["n_tests"],
              "n_passed": res["n_passed"], "elapsed_s": res["elapsed"]}
    flags: list[str] = []
    if res["passed"]:
        return Verdict(True, "correct", 1.0, extracted, expected, detail, flags)
    detail["test_index"] = res["test_index"]
    detail["stderr_tail"] = (res["stderr"] or "")[-600:]
    if res["kind"] == "timeout":
        flags.append("timeout")
    elif res["kind"] == "budget_exceeded":
        flags.append("budget_exceeded")
    partial = res["n_passed"] / res["n_tests"] if res["n_tests"] else 0.0
    return Verdict(False, "wrong", round(partial, 4) if item.get("subfamily") == "lcb_easy_medium" else 0.0,
                   extracted, expected, detail, flags)


# ======================================================================================
# mock oracle / statistics
# ======================================================================================

def mock_response(item: dict):
    """The canonical solution in a python fence (HumanEval+/MBPP+ ship one); LCB ships no solutions, so the
    oracle is a lookup program that answers each of the item's tests (pipeline verification only)."""
    sub = item.get("subfamily")
    if sub == "humanevalplus":
        return ("Here is the completed function.\n\n```python\n" + item["prompt"].rstrip("\n") + "\n"
                + item["canonical"].rstrip("\n") + "\n```\n")
    if sub == "mbppplus":
        return "```python\n" + "\n".join(item.get("test_imports") or []) + "\n" + item["canonical"].strip("\n") + "\n```\n"
    if sub == "lcb_easy_medium":
        table = repr([(t["input"], t["output"]) for t in item.get("tests") or []])
        if item.get("mode") == "functional":
            prog = ("import json\n_T = " + table + "\n\nclass Solution:\n"
                    f"    def {item['func_name']}(self, *args):\n"
                    "        for _i, _o in _T:\n"
                    "            _a = [json.loads(l) for l in _i.split(\"\\n\")] if _i.strip() else []\n"
                    "            if _a == list(args):\n"
                    "                return json.loads(_o)\n"
                    "        return None\n")
        else:
            prog = ("import sys\n_T = " + table + "\n_s = sys.stdin.read()\n"
                    "for _i, _o in _T:\n"
                    "    if _i == _s or _i.rstrip() == _s.rstrip() or _i.split() == _s.split():\n"
                    "        sys.stdout.write(_o)\n        break\n")
        return "```python\n" + prog + "```\n"
    return None


def aggregate(records: list[dict]) -> dict:
    scored = [r for r in records if r.get("status") not in ("error", "cancelled", "skipped")]
    if not scored:
        return {}
    kinds = collections.Counter((r.get("detail") or {}).get("kind", r.get("status")) for r in scored if not r.get("correct"))
    out = {"unparsed_rate": round(sum(1 for r in scored if r.get("status") == "unparsed") / len(scored), 4),
           "timeout_rate": round(sum(1 for r in scored if "timeout" in (r.get("flags") or [])) / len(scored), 4),
           "syntax_error_rate": round(sum(1 for r in scored if "syntax_error" in (r.get("flags") or [])) / len(scored), 4),
           "fail_kinds": dict(sorted(kinds.items()))}
    lcb = [r for r in scored if r.get("sub") == "lcb_easy_medium"]
    if lcb:
        out["lcb_mean_test_pass_frac"] = round(sum(float(r.get("score") or 0.0) for r in lcb) / len(lcb), 4)
    return out
