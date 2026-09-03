"""
families/selftest.py - synthetic pipeline check.  No downloads, no node needed.

Two sub-families built from a seeded RNG:
  arith - "Compute a x b + c." with the canonical math prompt suffix; scored via \\boxed{} extraction
  mcq   - "Which option equals x x y?" with 4 options; scored via the bounded letter cascade

It exists to verify run_eval.py end-to-end against mock_server.py (oracle -> acc 1.0, canned -> unparsed,
noisy -> Wilson CI covers the configured accuracy) and as a one-minute liveness smoke test on the node.
HIDDEN = True keeps it out of the default --families set.
"""
from __future__ import annotations

from typing import Callable, Optional

import common
from common import DEFAULT_SEED, Verdict
from families import _base

NAME = "selftest"
DESCRIPTION = "synthetic arithmetic + MCQ items for pipeline verification (no downloads)"
SUBFAMILIES = ["arith", "mcq"]
PRIORITY = 0
HIDDEN = True
ITEM_TIME_FALLBACK_S = 10.0
NOTES = ["selftest: synthetic items, not a capability measurement"]

MATH_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
MCQ_SUFFIX = ('\nThink step by step and then output the answer in the format of "The answer is (X)" at the end.')


def prepare(data_dir: str, seed: int = DEFAULT_SEED, profile: str = "default", refresh: bool = False,
            log: Callable[[str], None] = print, allow_short: bool = False, n_arith: Optional[int] = None,
            n_mcq: Optional[int] = None, **opts) -> dict:
    n_arith = int(n_arith) if n_arith is not None else (60 if profile == "full" else 20)
    n_mcq = int(n_mcq) if n_mcq is not None else (30 if profile == "full" else 10)
    items: list[dict] = []

    rng = common.seeded_rng(seed, NAME, "arith")
    for i in range(n_arith):
        a, b, c = rng.randint(12, 99), rng.randint(12, 99), rng.randint(100, 999)
        q = f"Compute {a} × {b} + {c}."
        items.append({"id": f"selftest-arith-{i:03d}", "family": NAME, "subfamily": "arith", "order": i,
                      "messages": [{"role": "user", "content": q + MATH_SUFFIX}],
                      "answer": str(a * b + c), "meta": {"source": "synthetic", "a": a, "b": b, "c": c}})

    rng = common.seeded_rng(seed, NAME, "mcq")
    for i in range(n_mcq):
        x, y = rng.randint(11, 49), rng.randint(11, 49)
        t = x * y
        deltas = rng.sample([-11, -7, -3, -1, 1, 3, 7, 11], 3)
        opts_vals = [t] + [t + d for d in deltas]
        rng.shuffle(opts_vals)
        letters = "ABCD"
        ans = letters[opts_vals.index(t)]
        q = f"Which option equals {x} × {y}?\n" + "\n".join(f"{letters[k]}. {v}" for k, v in enumerate(opts_vals))
        items.append({"id": f"selftest-mcq-{i:03d}", "family": NAME, "subfamily": "mcq", "order": i,
                      "messages": [{"role": "user", "content": q + MCQ_SUFFIX}],
                      "options": [str(v) for v in opts_vals], "answer": ans,
                      "meta": {"source": "synthetic", "x": x, "y": y}})

    rel = f"items/{NAME}.jsonl"
    common.write_jsonl(_base.items_path(NAME, data_dir), items)
    return {"file": rel, "counts": {"arith": n_arith, "mcq": n_mcq}, "pools": {"arith": "synthetic", "mcq": "synthetic"},
            "sources": {"selftest": {"url": None, "note": "synthetic, seeded"}}, "notes": NOTES}


def score(item: dict, response_text: str, meta: Optional[dict] = None) -> Verdict:
    expected = item["answer"]
    if item.get("subfamily") == "mcq":
        letter, method = common.extract_letter(response_text, len(item.get("options", [])) or 4)
        if letter is None:
            return Verdict.unparsed(expected, {"method": method}, ["no_letter"])
        return Verdict(letter == expected, extracted=letter, expected=expected, detail={"method": method})
    cand, method = common.extract_final_answer(response_text, allow_last_integer=False)
    if cand is None:
        return Verdict.unparsed(expected, {"method": method}, ["no_boxed"])
    norm = common.strip_math_delims(cand).replace(",", "").replace(" ", "")
    ok = norm == expected
    try:
        ok = ok or int(float(norm)) == int(expected) and float(norm) == int(expected)
    except ValueError:
        pass
    return Verdict(ok, extracted=norm, expected=expected, detail={"method": method})


def mock_response(item: dict):
    """The mock's oracle answer; arith uses the mock's default '\\boxed{answer}' oracle."""
    if item.get("subfamily") == "mcq":
        return f"Let me compare the options.\nThe answer is ({item['answer']})."
    return None


def aggregate(records: list[dict]) -> dict:
    scored = [r for r in records if r.get("status") not in ("error", "cancelled", "skipped")]
    unparsed = sum(1 for r in scored if r.get("status") == "unparsed")
    return {"unparsed_rate": round(unparsed / len(scored), 4) if scored else None}
