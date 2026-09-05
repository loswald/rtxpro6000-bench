"""
families/_base.py - the family plugin interface: a documented template plus the defaults the
registry (families/__init__.py) copies onto any family module that does not define them.

A family is one Python module `families/<name>.py`.  It is loaded by name from `--families`
(run_eval.py) or `--only` (prepare_data.py) and MUST define:

    NAME                     str, == module name ("math", "code", "tools", "longctx", "knowledge", "ifeval")
    prepare(...)             download + build the concentrated item set once (deterministic, seeded)
    score(item, text)        turn the model's visible text into a Verdict

and MAY define (defaults below are used otherwise):

    SUBFAMILIES              list[str]  sub-family names in dispatch order (slowest first)
    PRIORITY                 int        cross-family dispatch order, lower first (tools 10, code 20, math 30,
                                        longctx 40, knowledge 50, ifeval 60 - see families/__init__.py)
    HIDDEN                   bool       True -> not part of the default --families set (e.g. selftest)
    DEFAULT_MAX_TOKENS       None | {"default": int, "reasoning": int}
                                        None -> the run's --max-tokens; a dict -> the family's own cap
                                        (tools 1024/2048, longctx 512/1024 in the design spec)
    ITEM_TIME_FALLBACK_S     float      p90 item wall-time guess used by the time-budget scheduler until
                                        the family has produced 3 observations
    NOTES                    list[str]  caveats copied into <tag>.json "notes"
    load_items(limit, seed, data_dir=None)
                                        the items to run, IN RUN ORDER (sub-families interleaved, nested
                                        subsets under --limit).  Default: data/items/<NAME>.jsonl via
                                        default_load_items() below.
    build_messages(item, ctx)           the PROMPT builder: OpenAI chat messages for one item.
                                        Default: item["messages"].
    mock_response(item)                 what mock_server.py should answer for this item (str, or a dict
                                        {"content", "reasoning", "finish_reason", "tool_calls"}), None ->
                                        the mock's default oracle ("... \\boxed{answer}.") or echo.
    aggregate(records)                  extra per-family statistics from the item records
                                        (e.g. {"unparsed_rate": ..}), merged into the family's "extra".
    async run_item(item, ctx)           OPTIONAL multi-request driver (BFCL multi-turn, anything agentic).
                                        When present it replaces the default request->normalise->score path
                                        and must return common.ItemOutcome.  Use ctx.client.chat(...) with
                                        route_key=item["_index"] for sticky routing and honour ctx.deadline.
    async prepare_run(items, ctx)       OPTIONAL once-per-run hook before dispatch (e.g. long-context
                                        /tokenize calibration).

Item schema (one JSON object per line in data/items/<NAME>.jsonl, sorted keys):

    {"id": "aime25-I-2", "family": "math", "subfamily": "aime25", "order": 17,
     "messages": [{"role": "user", "content": "..."}], "answer": "588",
     "meta": {"source": "math-ai/aime25", "source_id": "1"}}

  * `order` is the position in the seeded run order of its sub-family; --limit keeps the first N items
    of the interleaved order, so smaller runs are nested subsets of larger ones.
  * `max_tokens` (optional, int) caps this item below the family cap.
  * `mock_marker` (optional) a distinctive substring of the rendered prompt for mock_server.py; the
    default marker is the first 240 whitespace-normalised chars of the last user message.
  * header records (objects without "id", e.g. {"_exemplars": {...}}) are skipped by the default loader
    and available through read_headers().

Scoring contract:

  * score() receives ONLY the visible text: <think> blocks, the reasoning field and harmony leaks are
    removed by common.normalize_response(); a \\boxed{} inside reasoning is never seen.
  * the optional third argument `meta` is {"finish_reason", "reasoning", "flags", "prompt_tokens",
    "completion_tokens", "message"}; declare it only if you need it (score(item, text) is fine).
  * return common.Verdict(correct, status="correct|wrong|unparsed", score=0..1, extracted, expected,
    detail={...}, flags=[...]).  Scorers must be tolerant of \\boxed{} and of leftover markdown.
  * all randomness inside a family goes through common.seeded_rng(seed, NAME, subfamily, ...).

prepare() contract:

    prepare(data_dir, seed=DEFAULT_SEED, profile="default", refresh=False, log=print, allow_short=False,
            **opts) -> {"file": "items/<NAME>.jsonl",            # relative to data_dir
                        "counts": {subfamily: n, ...},
                        "sources": {name: {"url", "sha256", "bytes", "revision"/"commit"}, ...},
                        "pools": {subfamily: pool_size, ...},    # optional, printed
                        "notes": [...],                            # optional
                        "manifest_extra": {...}}                  # optional, merged into manifest.json

  * fetch with common.fetch()/fetch_json() into data_dir/raw/<source>/...; reuse the cache unless refresh.
  * raise common.ShortPool when a pool is smaller than requested and not allow_short.
  * write the item file with common.write_jsonl (sorted keys -> byte-identical rebuilds).
"""
from __future__ import annotations

import functools
import os
from typing import Any, Callable, Optional

import common
from common import DEFAULT_SEED, Verdict  # noqa: F401  (re-exported for family modules)

# ---- module-level attributes with their defaults --------------------------------------------
NAME = "_base"
DESCRIPTION = "abstract family template"
SUBFAMILIES: list[str] = []
PRIORITY = 50
HIDDEN = False
DEFAULT_MAX_TOKENS: Optional[dict] = None
ITEM_TIME_FALLBACK_S = 60.0
NOTES: list[str] = []
run_item = None          # optional: async def run_item(item, ctx) -> common.ItemOutcome
prepare_run = None       # optional: async def prepare_run(items, ctx) -> None


# ---- required hooks (raise in the template) --------------------------------------------------
def prepare(data_dir: str, seed: int = DEFAULT_SEED, profile: str = "default", refresh: bool = False,
            log: Callable[[str], None] = print, allow_short: bool = False, **opts) -> dict:
    raise NotImplementedError(f"family {NAME!r} does not implement prepare()")


def score(item: dict, response_text: str, meta: Optional[dict] = None) -> Verdict:
    raise NotImplementedError(f"family {NAME!r} does not implement score()")


# ---- optional hooks with working defaults ----------------------------------------------------
def build_messages(item: dict, ctx: Any) -> list[dict]:
    """Default PROMPT builder: the messages stored on the item."""
    return item["messages"]


def mock_response(item: dict):
    return None


def aggregate(records: list[dict]) -> dict:
    return {}


# ---- helpers shared by family modules --------------------------------------------------------
def items_path(name: str, data_dir: Optional[str] = None) -> str:
    return os.path.join(data_dir or common.DEFAULT_DATA_DIR, "items", f"{name}.jsonl")


def read_headers(name: str, data_dir: Optional[str] = None) -> dict:
    """Header records (objects without 'id') of an item file, merged into one dict."""
    out: dict = {}
    path = items_path(name, data_dir)
    if os.path.exists(path):
        for row in common.read_jsonl(path):
            if "id" not in row:
                out.update(row)
    return out


def interleave(groups: list[list]) -> list:
    """Round-robin interleave of several ordered lists (keeps each list's order)."""
    out: list = []
    idx = 0
    remaining = [list(g) for g in groups if g]
    while remaining:
        g = remaining[idx % len(remaining)]
        out.append(g.pop(0))
        if not g:
            remaining.remove(g)
        else:
            idx += 1
    return out


def order_items(items: list[dict], subfamilies: Optional[list[str]] = None) -> list[dict]:
    """Group by subfamily (SUBFAMILIES order, then first appearance), sort each group by `order`
    (stable), interleave round-robin."""
    groups: dict[str, list[dict]] = {}
    for sf in subfamilies or []:
        groups[sf] = []
    for it in items:
        groups.setdefault(it.get("subfamily", it.get("family", "")), []).append(it)
    ordered = [sorted(g, key=lambda it: (it.get("order", 0), it.get("id", ""))) for g in groups.values() if g]
    return interleave(ordered)


def default_load_items(mod, limit: Optional[int] = None, seed: int = DEFAULT_SEED,
                       data_dir: Optional[str] = None) -> list[dict]:
    """Read data/items/<NAME>.jsonl, fill family/subfamily, interleave sub-families, apply limit."""
    path = items_path(mod.NAME, data_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run: python3 evalsuite/prepare_data.py --only {mod.NAME}")
    items = [r for r in common.read_jsonl(path) if "id" in r]
    for it in items:
        it.setdefault("family", mod.NAME)
        it.setdefault("subfamily", mod.NAME)
    items = order_items(items, getattr(mod, "SUBFAMILIES", None))
    if limit is not None:
        items = items[: max(0, int(limit))]
    return items


# ---- registry support ------------------------------------------------------------------------
_DEFAULT_ATTRS = ("DESCRIPTION", "SUBFAMILIES", "PRIORITY", "HIDDEN", "DEFAULT_MAX_TOKENS", "ITEM_TIME_FALLBACK_S",
                  "NOTES", "run_item", "prepare_run", "build_messages", "mock_response", "aggregate", "prepare", "score")
REQUIRED_ATTRS = ("NAME",)


def apply_defaults(mod) -> None:
    """Copy missing optional attributes from this template onto a family module (idempotent)."""
    if getattr(mod, "_defaults_applied", False):
        return
    for attr in REQUIRED_ATTRS:
        if not hasattr(mod, attr):
            raise AttributeError(f"family module {mod.__name__} lacks required attribute {attr}")
    this = globals()
    for attr in _DEFAULT_ATTRS:
        if not hasattr(mod, attr):
            setattr(mod, attr, this[attr])
    if not hasattr(mod, "load_items"):
        mod.load_items = functools.partial(default_load_items, mod)
    mod._defaults_applied = True
