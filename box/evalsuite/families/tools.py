"""
families/tools.py - agentic tool calling, BFCL v3 single-turn AST categories (Berkeley Function Calling
Leaderboard, ShishirPatil/gorilla, pinned commit) scored BFCL-AST-style without any model in the loop.

Sub-families (default profile / full profile):
  simple             20 / 40   one function offered, exactly one call expected
  multiple           15 / 30   2-4 functions offered, one call expected (pick the right one)
  parallel           15 / 30   one function, 2-8 calls expected (order-insensitive)
  parallel_multiple  10 / 20   2-4 functions, 2-5 calls expected (order-insensitive)
  irrelevance        10 / 20   the offered function does NOT fit: the model must not call anything

Item selection is concentrated on the hard end of each pool: a deterministic difficulty score (number of
expected arguments, "must be omitted" optional parameters that trap default-hallucination, nested
array/dict arguments, number of calls, number of similarly named functions, question length; for
irrelevance the lexical overlap between question and the tempting function) ranks the pool, the top
2xK form the candidate pool and K are drawn with the seeded RNG.

Modes (--family-opt tools.mode=prompt|native, default prompt):
  prompt   the JSON schemas go into the system prompt; the model answers with a JSON array of
           {"name", "arguments"} calls (BFCL-style python calls `[f(a=1)]` are accepted as a fallback);
           `[]` / plain text = no call.  Works on any endpoint.
  native   the schemas are sent as OpenAI `tools` (tool_choice auto) and message.tool_calls is scored;
           text is NOT scored (a parseable call list left in the text is flagged text_calls_ignored so a
           misconfigured tool parser is visible in aggregate()).

Scoring = BFCL AST checker semantics: function name equality (dotted names are sanitised for the API and
mapped back), every expected parameter present unless its possible-answer list contains "" (optional),
no parameter outside the possible-answer dict, each value equal to ANY listed possible value with type
coercion (int/float/numeric strings, bool/"true", BFCL string standardisation: spaces , . / - _ * ^ removed,
lower-cased), lists compared element-wise, nested dicts recursively; parallel categories need a perfect
one-to-one matching between the model's calls and the ground-truth calls (order-insensitive, count must
match).  score = matched calls / expected calls, correct only when everything matches.

aggregate() reports acc_official (BFCL "non-live AST" style unweighted mean of the four AST sub-families),
per-sub-family accuracies, irrelevance accuracy, an error-code histogram and parse statistics.
"""
from __future__ import annotations

import ast
import asyncio
import json
import os
import re
from typing import Any, Callable, Optional

import common
from common import DEFAULT_SEED, ItemOutcome, Verdict
from families import _base

NAME = "tools"
DESCRIPTION = "BFCL v3 single-turn tool calling (simple/multiple/parallel/parallel_multiple/irrelevance), AST scoring"
SUBFAMILIES = ["parallel_multiple", "parallel", "multiple", "simple", "irrelevance"]
PRIORITY = 10
HIDDEN = False
DEFAULT_MAX_TOKENS = {"default": 1024, "reasoning": 2048}
ITEM_TIME_FALLBACK_S = 30.0
NOTES = [
    "tools: BFCL v3 single-turn AST categories, concentrated hard subsets (not the whole benchmark); "
    "acc_official = unweighted mean over simple/multiple/parallel/parallel_multiple (irrelevance separate)",
    "tools: irrelevance counts any non-call response as correct (BFCL semantics), so a model that never "
    "calls tools scores 1.0 on that sub-family and 0 on the others",
]

# ---- source -----------------------------------------------------------------------------------
# Last commit where the v3 files live at berkeley-function-call-leaderboard/data/ (the next commit,
# c15b2a15 "Packagerize for PyPI Distribution", moved them under bfcl_eval/).
BFCL_COMMIT = "791f7aca0f174da17edd40d795daad4254970db9"
BFCL_BASE = "https://raw.githubusercontent.com/ShishirPatil/gorilla/{commit}/berkeley-function-call-leaderboard/data"
AST_SUBS = ["simple", "multiple", "parallel", "parallel_multiple"]
SIZES = {"default": {"simple": 20, "multiple": 15, "parallel": 15, "parallel_multiple": 10, "irrelevance": 10},
         "full": {"simple": 40, "multiple": 30, "parallel": 30, "parallel_multiple": 20, "irrelevance": 20}}
POOL_FACTOR = 2  # candidate pool = top POOL_FACTOR x K by difficulty, K drawn with the seeded RNG

SYSTEM_PROMPT = (
    "You are a function-calling assistant. You are given a user request and a set of available functions, "
    "described below as JSON schemas.\n\n"
    "Decide whether the request can be fulfilled by calling one or more of the functions.\n"
    "- If it can, respond with ONLY a JSON array of function calls, one object per call, in the form\n"
    '  [{"name": "<function name>", "arguments": {"<parameter>": <value>, ...}}]\n'
    "  Use the exact function and parameter names from the schemas, provide every required parameter, use "
    "JSON values of the declared types, and do not add parameters or values that the request does not "
    "specify. If the request needs several calls, put all of them in the one array.\n"
    "- If none of the functions can be used for the request, or the request lacks information a function "
    "requires, respond with the empty array [] followed by a one-sentence explanation. Never call a "
    "function that does not fit the request.\n\n"
    "Available functions:\n{tools}"
)

# ---- BFCL -> OpenAPI schema conversion ---------------------------------------------------------
_TYPE_MAP = {"dict": "object", "object": "object", "float": "number", "number": "number", "integer": "integer",
             "int": "integer", "string": "string", "str": "string", "boolean": "boolean", "bool": "boolean",
             "array": "array", "list": "array", "tuple": "array", "any": None, "null": "null"}
_NAME_RX = re.compile(r"[^A-Za-z0-9_-]")


def _to_openapi(schema: Any) -> Any:
    """Recursively map BFCL types (dict/float/tuple/any) onto JSON-schema types; keep everything else."""
    if not isinstance(schema, dict):
        return schema
    out: dict = {}
    for k, v in schema.items():
        if k == "type":
            t = _TYPE_MAP.get(str(v).lower(), v) if isinstance(v, str) else v
            if t is not None:
                out["type"] = t
        elif k == "properties" and isinstance(v, dict):
            out["properties"] = {pk: _to_openapi(pv) for pk, pv in v.items()}
        elif k == "items":
            out["items"] = _to_openapi(v)
        else:
            out[k] = v
    if out.get("type") == "object" and "properties" not in out and "properties" in schema:
        out["properties"] = {}
    return out


def sanitize_name(name: str) -> str:
    return (_NAME_RX.sub("_", name.strip()) or "fn")[:64]


def build_tools(functions: list[dict]) -> tuple[list[dict], dict]:
    """(OpenAI tools list, {api name -> original BFCL name})."""
    tools: list[dict] = []
    names: dict[str, str] = {}
    for fn in functions:
        orig = fn["name"]
        api = sanitize_name(orig)
        base, k = api, 2
        while api in names and names[api] != orig:
            api = f"{base[:60]}_{k}"
            k += 1
        names[api] = orig
        params = _to_openapi(fn.get("parameters") or {"type": "dict", "properties": {}})
        if params.get("type") != "object":
            params["type"] = "object"
        params.setdefault("properties", {})
        tools.append({"type": "function", "function": {"name": api, "description": fn.get("description", ""),
                                                       "parameters": params}})
    return tools, names


# ---- data ---------------------------------------------------------------------------------------
def _read_jsonl_or_json(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    stripped = text.strip()
    if stripped.startswith("["):
        data = json.loads(stripped)
        return [r for r in data if isinstance(r, dict)]
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


_STOP = set("""a an the of to in on for and or with by from at as is are be this that these those it its into your my
our their his her you we they i me us them what which who how when where why please find get give show tell calculate
compute using use want need would like can could should also all any some about between within over under than then""".split())


def _words(s: str) -> set:
    return {w for w in re.findall(r"[a-z]+", s.lower()) if len(w) > 2 and w not in _STOP}


def _param_alts(alts: Any) -> list:
    return alts if isinstance(alts, list) else [alts]


def difficulty(sub: str, question: str, functions: list[dict], ground_truth: list[dict]) -> float:
    """Deterministic hardness proxy used to concentrate the item set (see module docstring)."""
    qlen = len(question) / 300.0
    if sub == "irrelevance":
        qw = _words(question)
        best = 0.0
        n_params = 0
        for fn in functions:
            fw = _words(fn.get("name", "").replace(".", " ").replace("_", " ") + " " + fn.get("description", ""))
            for p, ps in ((fn.get("parameters") or {}).get("properties") or {}).items():
                fw |= _words(p.replace("_", " ") + " " + str((ps or {}).get("description", "")))
            n_params += len(((fn.get("parameters") or {}).get("properties") or {}))
            if qw and fw:
                best = max(best, len(qw & fw) / len(qw | fw))
        return round(6.0 * best + 0.3 * n_params + qlen, 4)
    n_params = traps = nested = 0
    for call in ground_truth:
        for _fname, args in call.items():
            for _p, alts in (args or {}).items():
                alts = _param_alts(alts)
                n_params += 1
                if all(a == "" for a in alts):
                    traps += 1
                if any(isinstance(a, (list, dict)) for a in alts):
                    nested += 1
    n_calls = len(ground_truth)
    n_funcs = len(functions)
    toks = [set(re.split(r"[._]", fn["name"].lower())) - {""} for fn in functions]
    shared = 0
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            if toks[i] & toks[j]:
                shared += 1
    return round(1.0 * n_params + 1.5 * traps + 2.0 * nested + 1.5 * (n_calls - 1) + 0.75 * (n_funcs - 1)
                 + 0.5 * min(shared, 4) + qlen, 4)


def prepare(data_dir: str, seed: int = DEFAULT_SEED, profile: str = "default", refresh: bool = False,
            log: Callable[[str], None] = print, allow_short: bool = False, commit: Optional[str] = None,
            sizes: Optional[dict] = None, pool_factor: Optional[int] = None, **opts) -> dict:
    commit = str(commit or BFCL_COMMIT)
    want = dict(SIZES["full" if profile == "full" else "default"])
    if isinstance(sizes, dict):
        want.update({k: int(v) for k, v in sizes.items()})
    pf = int(pool_factor or POOL_FACTOR)
    base = BFCL_BASE.format(commit=commit)
    raw_dir = os.path.join(data_dir, "raw", "bfcl_v3", commit[:12])
    sources: dict = {}
    pools: dict = {}
    items: list[dict] = []
    seen_markers: set = set()

    def get(rel: str) -> list[dict]:
        url = f"{base}/{rel}"
        dest = os.path.join(raw_dir, rel.replace("/", os.sep))
        info = common.fetch(url, dest, refresh=refresh, log=log)
        sources[f"bfcl_v3/{rel[:-5] if rel.endswith('.json') else rel}"] = {
            "url": url, "sha256": info["sha256"], "bytes": info["bytes"], "commit": commit}
        return _read_jsonl_or_json(dest)

    for sub in SUBFAMILIES:
        rows = get(f"BFCL_v3_{sub}.json")
        answers: dict[str, list] = {}
        if sub != "irrelevance":
            answers = {r["id"]: r["ground_truth"] for r in get(f"possible_answer/BFCL_v3_{sub}.json")}
        cands: list[tuple[float, str, dict]] = []
        for r in rows:
            q = r.get("question") or []
            turns = q[0] if q and isinstance(q[0], list) else q
            msgs = [m for m in turns if isinstance(m, dict) and m.get("role") in ("system", "user", "assistant")]
            if not msgs or msgs[-1].get("role") != "user" or not isinstance(msgs[-1].get("content"), str):
                continue
            if sub != "irrelevance" and r["id"] not in answers:
                continue
            gt = answers.get(r["id"], [])
            if sub != "irrelevance" and not all(isinstance(c, dict) and len(c) == 1 for c in gt):
                continue
            marker = " ".join(msgs[-1]["content"].split())[:240]
            if marker in seen_markers:
                continue  # two items with the same prompt prefix would collide in mock_server's marker index
            seen_markers.add(marker)
            d = difficulty(sub, msgs[-1]["content"], r["function"], gt)
            cands.append((d, r["id"], {"msgs": msgs, "functions": r["function"], "gt": gt}))
        pools[sub] = len(cands)
        k = want[sub]
        if len(cands) < k:
            if not allow_short:
                raise common.ShortPool(f"{NAME}/{sub}: pool {len(cands)} < requested {k}")
            k = len(cands)
        cands.sort(key=lambda t: (-t[0], t[1]))
        pool = cands[: min(len(cands), max(k, pf * k))]
        rng = common.seeded_rng(seed, NAME, sub)
        chosen = rng.sample(pool, k)
        for order, (d, sid, c) in enumerate(chosen):
            tools, names = build_tools(c["functions"])
            items.append({
                "id": f"bfcl3-{sid}", "family": NAME, "subfamily": sub, "order": order,
                "messages": c["msgs"], "tools": tools, "tool_names": names,
                "answer": c["gt"], "n_expected_calls": len(c["gt"]),
                "meta": {"source": "ShishirPatil/gorilla BFCL_v3", "source_id": sid, "commit": commit,
                         "difficulty": d, "n_functions": len(c["functions"])},
            })
        log(f"   {NAME}/{sub}: pool {len(cands)}, candidate pool {len(pool)} (difficulty >= {pool[-1][0] if pool else None}), "
            f"chose {k}")

    items = _base.order_items(items, SUBFAMILIES)
    common.write_jsonl(_base.items_path(NAME, data_dir), items)
    counts = {sub: sum(1 for it in items if it["subfamily"] == sub) for sub in SUBFAMILIES}
    return {"file": f"items/{NAME}.jsonl", "counts": counts, "pools": pools, "sources": sources, "notes": NOTES,
            "manifest_extra": {"bfcl_v3_commit": commit}}


# ---- prompt -------------------------------------------------------------------------------------
def _mode(ctx: Any) -> str:
    m = str((ctx.opt("mode", "prompt") if ctx is not None else "prompt") or "prompt").lower()
    return "native" if m.startswith("nat") else "prompt"


def build_messages(item: dict, ctx: Any) -> list[dict]:
    msgs = [dict(m) for m in item["messages"]]
    if _mode(ctx) == "native":
        return msgs
    schemas = "\n".join(json.dumps(t["function"], ensure_ascii=False, sort_keys=True) for t in item["tools"])
    sys_text = SYSTEM_PROMPT.replace("{tools}", schemas)  # not .format(): the template shows literal JSON braces
    if msgs and msgs[0].get("role") == "system":
        msgs[0]["content"] = sys_text + "\n\n" + common.message_text(msgs[0].get("content"))
        return msgs
    return [{"role": "system", "content": sys_text}] + msgs


# ---- call extraction ----------------------------------------------------------------------------
_FENCE_RX = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n?(.*?)```", re.S)
_TAG_RX = re.compile(r"</?(?:tool_call|tool_calls|function_call|function_calls|tool|json|output|answer)>", re.I)
_PY_CALL_RX = re.compile(r"[A-Za-z_][\w.]*\s*\(")
# Character steps one extract_calls() may spend in bracket scanning.  A degenerate output such as "f( f( f( ..."
# (4.5k chars, well inside the 1024-token cap) otherwise costs O(n^3) - measured 74 s - and score() runs on the
# event loop.  3M steps is ~0.15 s and still allows hundreds of unbalanced brackets before a real call list.
_SCAN_BUDGET = 3_000_000
_MAX_BARE_CALLS = 200
_NESTED_DEPTH = 2


def _scan_from(text: str, i: int, budget: list) -> Optional[int]:
    """End (exclusive) of the balanced bracket span opening at text[i] (string-aware; [] {} () all counted),
    None when it never closes or the step budget runs out.  budget[0] is decremented per character."""
    depth = 0
    j, n = i, len(text)
    in_str: Optional[str] = None
    esc = False
    while j < n and budget[0] > 0:
        budget[0] -= 1
        c = text[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == in_str:
                in_str = None
        elif c in "\"'":
            in_str = c
        elif c in "[{(":
            depth += 1
        elif c in "]})":
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return None


def _balanced_spans(text: str, opens: str = "[{", budget: Optional[list] = None) -> list[tuple[int, int]]:
    """Top-level balanced bracket spans, left to right, non-overlapping, starting at any character in `opens`."""
    budget = budget if budget is not None else [_SCAN_BUDGET]
    spans: list[tuple[int, int]] = []
    i, n = 0, len(text)
    while i < n and budget[0] > 0:
        if text[i] in opens:
            end = _scan_from(text, i, budget)
            if end is not None:
                spans.append((i, end))
                i = end
                continue
        i += 1
    return spans


def _looks_like_schema(obj: dict) -> bool:
    """A function schema echoed from the prompt ({"name", "description", "parameters": {"properties": ..}})
    is not a call."""
    params = obj.get("parameters")
    return "description" in obj and isinstance(params, dict) and ("properties" in params or params.get("type") == "object")


def _as_call(obj: Any) -> Optional[dict]:
    """{"name","arguments"} from the common call encodings; None when it is not a call object."""
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("function"), dict) and "name" in obj["function"]:
        obj = obj["function"]
    if _looks_like_schema(obj):
        return None
    name = obj.get("name") or obj.get("function") or obj.get("function_name") or obj.get("tool") or obj.get("tool_name")
    if not isinstance(name, str) or not name.strip():
        return None
    args: Any = None
    for key in ("arguments", "parameters", "args", "params", "input", "kwargs"):
        if key in obj:
            args = obj[key]
            break
    if args is None:
        args = {k: v for k, v in obj.items() if k not in ("name", "function", "function_name", "tool", "tool_name",
                                                            "type", "id", "index")} if len(obj) > 1 else {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:  # ValueError, RecursionError on absurd nesting
            return {"name": name.strip(), "arguments": args, "_bad_args": True}
    if args is None:
        args = {}
    return {"name": name.strip(), "arguments": args}


def _calls_from_json(obj: Any) -> Optional[list[dict]]:
    """A JSON value -> list of calls, [] for an explicit 'no call', None when it is not a call encoding."""
    if isinstance(obj, dict):
        for key in ("tool_calls", "function_calls", "calls", "functions"):
            if isinstance(obj.get(key), list):
                inner = _calls_from_json(obj[key])
                if inner is not None:
                    return inner
        c = _as_call(obj)
        return [c] if c else None
    if isinstance(obj, list):
        if not obj:
            return []
        calls = [_as_call(o) for o in obj]
        if all(c is not None for c in calls):
            return calls  # type: ignore[return-value]
        return None
    return None


def _py_value(node: ast.AST, src: str) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return ast.get_source_segment(src, node)


def _calls_from_python(text: str) -> Optional[list[dict]]:
    """BFCL-style `[func(a=1, b='x'), pkg.f()]` (or a bare call) -> calls; None when not parseable."""
    src = text.strip().rstrip(".;")
    try:
        tree = ast.parse(src, mode="eval")
    except Exception:  # SyntaxError, ValueError (null bytes), RecursionError / MemoryError on absurd nesting
        return None
    body = tree.body
    nodes = list(body.elts) if isinstance(body, (ast.List, ast.Tuple)) else [body]
    calls: list[dict] = []
    for nd in nodes:
        if not isinstance(nd, ast.Call):
            return None
        fn = nd.func
        parts: list[str] = []
        while isinstance(fn, ast.Attribute):
            parts.append(fn.attr)
            fn = fn.value
        if not isinstance(fn, ast.Name):
            return None
        parts.append(fn.id)
        args = {kw.arg: _py_value(kw.value, src) for kw in nd.keywords if kw.arg}
        if nd.args:
            args["_positional"] = [_py_value(a, src) for a in nd.args]
        calls.append({"name": ".".join(reversed(parts)), "arguments": args})
    return calls


def extract_calls(text: str) -> tuple[Optional[list[dict]], str]:
    """(calls, method): calls is a list of {"name","arguments"} ([] for an explicit no-call), None when the text
    contains no call encoding.  method in {json_list, json_objects, python, none}."""
    if not text or not text.strip():
        return None, "none"
    t = _TAG_RX.sub(" ", text)
    fenced = [m.group(1) for m in _FENCE_RX.finditer(t)]
    # sources of candidate strings in text order: the whole text de-fenced, plus each fenced body
    t_plain = _FENCE_RX.sub(lambda m: " " + m.group(1) + " ", t)
    lists: list[tuple[list[dict], str]] = []
    objects: list[dict] = []
    budget = [_SCAN_BUDGET]

    def collect(segment: str, depth: int) -> None:
        """Every balanced [..] / {..} span of `segment` in text order; a span that is not a call encoding
        (e.g. \\boxed{[...]}, a schema echo, a set literal) is searched inside, `_NESTED_DEPTH` levels deep."""
        for start, end in _balanced_spans(segment, budget=budget):
            frag = segment[start:end]
            obj: Any = None
            try:
                obj = json.loads(frag)
            except Exception:  # ValueError, RecursionError on absurd nesting
                try:
                    obj = ast.literal_eval(frag)  # single quotes / True / None
                except Exception:
                    obj = None
                if obj is None or not isinstance(obj, (list, dict)):
                    py = _calls_from_python(frag) if frag.startswith("[") and _PY_CALL_RX.search(frag) else None
                    if py is not None:
                        lists.append((py, "python"))
                    elif depth < _NESTED_DEPTH:
                        collect(frag[1:-1], depth + 1)
                    continue
            calls = _calls_from_json(obj)
            if calls is None:
                if depth < _NESTED_DEPTH:
                    collect(frag[1:-1], depth + 1)
                continue
            if isinstance(obj, list):
                lists.append((calls, "json_list"))
            else:
                objects.extend(calls)

    collect(t_plain, 0)
    if lists:
        return lists[-1]                       # the last complete list wins (draft -> final answer)
    if objects:
        return objects, "json_objects"
    for frag in [t_plain] + fenced:
        py = _calls_from_python(frag)
        if py is not None:
            return py, "python"
    n_tried = 0
    for m in _PY_CALL_RX.finditer(t_plain):  # a bare `name(args)` somewhere in prose: one scan per candidate
        n_tried += 1
        if n_tried > _MAX_BARE_CALLS or budget[0] <= 0:
            break
        end = _scan_from(t_plain, m.end() - 1, budget)
        if end is not None:
            py = _calls_from_python(t_plain[m.start():end])
            if py is not None:
                return py, "python"
    return None, "none"


def calls_from_message(message: Optional[dict]) -> tuple[list[dict], list[str]]:
    """OpenAI message.tool_calls -> calls (+ flags for unparsable argument strings)."""
    calls: list[dict] = []
    flags: list[str] = []
    for tc in (message or {}).get("tool_calls") or []:
        fn = tc.get("function") if isinstance(tc, dict) else None
        if not isinstance(fn, dict):
            continue
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except Exception:  # ValueError, RecursionError on absurd nesting
                flags.append("bad_arguments_json")
                args = {"_raw": args}
        calls.append({"name": str(fn.get("name") or ""), "arguments": args if args is not None else {}})
    return calls, flags


# ---- BFCL AST-style comparison -------------------------------------------------------------------
_STD_RX = re.compile(r"[ ,./\-_*^]")
_NAME_SEP_RX = re.compile(r"[ ._\-]")


def _std(s: str) -> str:
    return _STD_RX.sub("", s).lower().replace("'", '"')


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _num(v: Any) -> Optional[float]:
    if _is_num(v):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace(",", "")
        try:
            return float(s)
        except ValueError:
            return None
    return None


def value_matches(v: Any, alt: Any) -> bool:
    """Model value v vs ONE possible answer alt, with type coercion (BFCL checker semantics)."""
    if isinstance(alt, dict):
        return isinstance(v, dict) and dict_matches(v, alt)
    if isinstance(alt, (list, tuple)):
        if not isinstance(v, (list, tuple)) or len(v) != len(alt):
            return False
        return all(value_matches(a, b) for a, b in zip(v, alt))
    if alt is None:
        return v is None or (isinstance(v, str) and v.strip().lower() in ("", "none", "null"))
    if isinstance(alt, bool):
        if isinstance(v, bool):
            return v == alt
        return isinstance(v, str) and v.strip().lower() == str(alt).lower()
    if _is_num(alt):
        n = _num(v)
        return n is not None and abs(n - float(alt)) <= 1e-6 * max(1.0, abs(float(alt)))
    if isinstance(alt, str):
        na = _num(alt)
        if isinstance(v, str):
            nv = _num(v)
            if na is not None and nv is not None:
                # both numeric-looking: compare as numbers ONLY - the BFCL string standardisation would also
                # strip the '.' and accept "15" for "1.5"
                return abs(na - nv) <= 1e-6 * max(1.0, abs(na))
            return _std(v) == _std(alt)
        if isinstance(v, bool):
            return _std(alt) == str(v).lower()
        if _is_num(v):
            if na is not None:
                return abs(na - float(v)) <= 1e-6 * max(1.0, abs(na))
            return _std(str(v)) == _std(alt)
        return False
    return v == alt


def dict_matches(v: dict, alt: dict) -> bool:
    """Nested possible-answer dict: every key's value is itself a list of alternatives ('' = may be omitted)."""
    for k in v:
        if k not in alt:
            return False
    for k, alts in alt.items():
        alts = _param_alts(alts)
        if k not in v:
            if "" not in alts:
                return False
            continue
        if not any(value_matches(v[k], a) for a in alts):
            return False
    return True


def canon_name(name: str, item: dict) -> str:
    names = item.get("tool_names") or {}
    n = (name or "").strip().strip("`'\"")
    if n in names:
        return names[n]
    if n in names.values():
        return n
    s = sanitize_name(n) if n else n
    if s in names:
        return names[s]
    key = _NAME_SEP_RX.sub("", n)   # separator-insensitive but CASE-SENSITIVE (BFCL: names must match exactly)
    for api, orig in names.items():
        if key in (_NAME_SEP_RX.sub("", api), _NAME_SEP_RX.sub("", orig)):
            return orig
    return n


def match_call(call: dict, gt_call: dict, item: dict) -> list[str]:
    """Error codes for one model call against one ground-truth call ([] = match)."""
    (gt_name, gt_args), = gt_call.items()
    gt_args = gt_args or {}
    errors: list[str] = []
    if canon_name(call.get("name", ""), item) != gt_name:
        return [f"wrong_function:{call.get('name')}"]
    args = call.get("arguments")
    if call.get("_bad_args") or not isinstance(args, dict):
        return ["arguments_not_object"]
    for k in args:
        if k not in gt_args:
            errors.append(f"unexpected_param:{k}")
    for k, alts in gt_args.items():
        alts = _param_alts(alts)
        if k not in args:
            if "" not in alts:
                errors.append(f"missing_param:{k}")
            continue
        if not any(value_matches(args[k], a) for a in alts):
            errors.append(f"value_mismatch:{k}")
    return errors


def _max_matching(ok: list[list[bool]]) -> int:
    """Size of a maximum bipartite matching (model calls x ground-truth calls), augmenting paths."""
    n_m = len(ok)
    n_g = len(ok[0]) if ok else 0
    match_g = [-1] * n_g

    def try_m(i: int, seen: list[bool]) -> bool:
        for j in range(n_g):
            if ok[i][j] and not seen[j]:
                seen[j] = True
                if match_g[j] < 0 or try_m(match_g[j], seen):
                    match_g[j] = i
                    return True
        return False

    return sum(1 for i in range(n_m) if try_m(i, [False] * n_g))


def score_calls(calls: list[dict], item: dict) -> Verdict:
    gt: list[dict] = item.get("answer") or []
    expected = gt
    sub = item.get("subfamily")
    extracted = [{"name": c.get("name"), "arguments": c.get("arguments")} for c in calls]
    if sub == "irrelevance":
        if calls:
            return Verdict(False, extracted=extracted, expected="no call",
                           detail={"errors": [f"spurious_call:{c.get('name')}" for c in calls]}, flags=["spurious_call"])
        return Verdict(True, extracted=extracted, expected="no call", detail={"errors": []})
    n_gt = len(gt)
    if not calls:
        return Verdict(False, "wrong", 0.0, extracted, expected, {"errors": ["declined"]}, ["declined"])
    ok = [[not match_call(c, g, item) for g in gt] for c in calls]
    matched = _max_matching(ok)
    errors: list[str] = []
    if len(calls) != n_gt:
        errors.append(f"call_count:{len(calls)}!={n_gt}")
    if matched < n_gt:
        # diagnosis: for each unmatched ground-truth call, the errors of the model call with the fewest errors
        for j, g in enumerate(gt):
            if any(ok[i][j] for i in range(len(calls))):
                continue
            best = min((match_call(c, g, item) for c in calls), key=len, default=["no_call"])
            gname = next(iter(g))
            errors.extend(f"{gname}:{e}" if n_gt > 1 else e for e in best)
    correct = matched == n_gt and len(calls) == n_gt
    partial = matched / max(n_gt, len(calls)) if n_gt else 0.0   # surplus calls dilute the credit
    return Verdict(correct, "correct" if correct else "wrong", partial, extracted, expected,
                   {"errors": errors, "matched": matched, "n_calls": len(calls)}, [] if correct else ["mismatch"])


def score(item: dict, response_text: str, meta: Optional[dict] = None) -> Verdict:
    """prompt mode: calls parsed from the visible text (message.tool_calls as a fallback when the text has none);
    native mode: message.tool_calls only, text calls flagged text_calls_ignored."""
    meta = meta or {}
    message = meta.get("message") or {}
    mode = meta.get("mode")
    if mode is None:
        mode = "native" if message.get("tool_calls") else "prompt"
    tc_calls, tc_flags = calls_from_message(message)
    text_calls, method = extract_calls(response_text or "")
    flags = list(tc_flags)
    if mode == "native":
        if tc_calls:
            calls: Optional[list[dict]] = tc_calls
            method = "tool_calls"
        else:
            calls = None
            if text_calls:
                flags.append("text_calls_ignored")
    else:
        calls = text_calls
        method = method
        if calls is None and tc_calls:
            calls, method = tc_calls, "tool_calls"
            flags.append("tool_calls_in_prompt_mode")
    if calls is None:
        if item.get("subfamily") == "irrelevance":
            v = Verdict(True, extracted=None, expected="no call", detail={"errors": [], "method": method})
        else:
            v = Verdict.unparsed(item.get("answer"), {"errors": ["no_calls"], "method": method}, ["no_calls"])
        v.flags = list(dict.fromkeys(list(v.flags) + flags))
        return v
    v = score_calls(calls, item)
    v.detail["method"] = method
    v.detail["mode"] = mode
    v.flags = list(dict.fromkeys(list(v.flags) + flags))
    return v


# ---- the request (one call per item, tools sent natively when asked) --------------------------
async def run_item(item: dict, ctx: Any) -> ItemOutcome:
    mode = _mode(ctx)
    messages = build_messages(item, ctx)
    tools = item["tools"] if mode == "native" else None
    max_tokens = min(ctx.max_tokens, int(item.get("max_tokens", ctx.max_tokens)))
    res = await ctx.client.chat(messages, route_key=int(item.get("_index", 0)), max_tokens=max_tokens,
                                temperature=ctx.temperature, top_p=ctx.top_p, seed=ctx.seed, tools=tools,
                                extra_body=ctx.extra_body or None)
    out = ItemOutcome(verdict=Verdict(False, "wrong"), prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens,
                      requests=1, retries=res.retries, finish_reason=res.finish_reason, base_url=res.base_url,
                      latency_s=res.latency_s, extra={"tools_mode": mode})
    if not res.ok:
        out.error = res.error or "request failed"
        out.extra["error_kind"] = res.error_kind
        return out
    norm = common.normalize_response(res.message, res.finish_reason)
    tool_calls = (res.message or {}).get("tool_calls") or []
    out.content = res.content or (json.dumps(tool_calls, ensure_ascii=False) if tool_calls else "")
    out.reasoning = norm.reasoning
    out.flags = list(norm.flags)
    out.extra["n_tool_calls"] = len(tool_calls)
    expected = "no call" if item.get("subfamily") == "irrelevance" else item.get("answer")
    if not tool_calls and norm.status == "truncated":
        out.verdict = Verdict(False, "truncated", 0.0, None, expected, {"errors": ["truncated"]})
        return out
    if not tool_calls and norm.status == "empty":
        out.verdict = Verdict(False, "empty", 0.0, None, expected, {"errors": ["empty"]})
        return out
    meta = {"finish_reason": res.finish_reason, "reasoning": norm.reasoning, "flags": norm.flags,
            "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens, "message": res.message,
            "mode": mode}
    # score off the event loop, as run_eval's default path does (a degenerate output can cost real parse time)
    out.verdict = await asyncio.get_running_loop().run_in_executor(None, score, item, norm.visible, meta)
    return out


# ---- mock + statistics ---------------------------------------------------------------------------
def _first_answer(v: Any) -> Any:
    """First concrete value of a possible-answer structure (alt lists inside dicts resolved recursively)."""
    if isinstance(v, dict) and v and all(isinstance(x, list) for x in v.values()):
        out = {}
        for k, alts in v.items():
            picked = next((a for a in alts if a != ""), None)
            if picked is not None:
                out[k] = _first_answer(picked)
        return out
    if isinstance(v, list):
        return [_first_answer(x) for x in v]
    return v


def oracle_calls(item: dict) -> list[dict]:
    """Calls matching the first possible answer of every ground-truth call (API names)."""
    names = {orig: api for api, orig in (item.get("tool_names") or {}).items()}
    calls = []
    for g in item.get("answer") or []:
        (fname, args), = g.items()
        chosen = {}
        for p, alts in (args or {}).items():
            alts = _param_alts(alts)
            picked = next((a for a in alts if a != ""), None)
            if picked is not None:
                chosen[p] = _first_answer(picked)
        calls.append({"name": names.get(fname, sanitize_name(fname)), "arguments": chosen})
    return calls


def mock_response(item: dict):
    """Oracle for mock_server.py: tool_calls (native mode) AND the same calls as a JSON list in the content
    (prompt mode); irrelevance answers with an explicit empty list and no tool call."""
    if item.get("subfamily") == "irrelevance":
        return "[] None of the available functions applies to this request."
    calls = oracle_calls(item)
    return {"content": json.dumps(calls, ensure_ascii=False), "tool_calls": calls}


_ERROR_CODES = ("missing_param", "unexpected_param", "value_mismatch", "wrong_function", "call_count",
                "arguments_not_object", "declined", "no_calls", "spurious_call", "no_call", "truncated", "empty")


def aggregate(records: list[dict]) -> dict:
    scored = [r for r in records if r.get("status") not in ("error", "cancelled", "skipped")]
    by_sub: dict[str, list[dict]] = {}
    for r in scored:
        by_sub.setdefault(r.get("sub") or "", []).append(r)
    acc_sub = {s: round(sum(1 for r in rs if r.get("correct")) / len(rs), 4) for s, rs in by_sub.items() if rs}
    ast_accs = [acc_sub[s] for s in AST_SUBS if s in acc_sub]
    ast_recs = [r for s in AST_SUBS for r in by_sub.get(s, [])]
    errors: dict[str, int] = {}
    for r in scored:
        for e in (r.get("detail") or {}).get("errors") or []:
            code = next((k for k in _ERROR_CODES if k in e), e.split(":")[0])  # drop function/param suffixes
            errors[code] = errors.get(code, 0) + 1
    methods: dict[str, int] = {}
    for r in scored:
        m = (r.get("detail") or {}).get("method") or "-"
        methods[m] = methods.get(m, 0) + 1
    modes = sorted({r.get("tools_mode") for r in scored if r.get("tools_mode")})
    return {
        "acc_official": round(sum(ast_accs) / len(ast_accs), 4) if ast_accs else None,
        "acc_ast_micro": round(sum(1 for r in ast_recs if r.get("correct")) / len(ast_recs), 4) if ast_recs else None,
        "acc_irrelevance": acc_sub.get("irrelevance"),
        "acc_by_sub": acc_sub,
        "mean_score_ast": round(sum(float(r.get("score") or 0.0) for r in ast_recs) / len(ast_recs), 4) if ast_recs else None,
        "unparsed_rate": round(sum(1 for r in scored if r.get("status") == "unparsed") / len(scored), 4) if scored else None,
        "declined_rate_ast": round(sum(1 for r in ast_recs if "declined" in (r.get("flags") or [])) / len(ast_recs), 4) if ast_recs else None,
        "text_calls_ignored": sum(1 for r in scored if "text_calls_ignored" in (r.get("flags") or [])),
        "tool_calls_in_prompt_mode": sum(1 for r in scored if "tool_calls_in_prompt_mode" in (r.get("flags") or [])),
        "error_codes": dict(sorted(errors.items())),
        "parse_methods": dict(sorted(methods.items())),
        "mode": modes[0] if len(modes) == 1 else (modes or None),
    }
