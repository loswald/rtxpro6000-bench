#!/usr/bin/env python3
"""
run_eval.py - run the quality eval suite against OpenAI-compatible (vLLM) endpoints.

  python3 evalsuite/run_eval.py --tag TAG --base-urls URL[,URL...] --model m [--out results/eval]
      [--families math,code,tools,longctx,knowledge,ifeval] [--limit N] [--concurrency 64]
      [--max-tokens 2048] [--max-tokens-family tools=1024,longctx=512] [--reasoning]
      [--temperature T] [--top-p P] [--seed 20260903] [--chat-template-kwargs JSON]
      [--time-budget 900] [--request-timeout 600] [--retries 3]
      [--family-opt tools.mode=native --family-opt knowledge.shots=0 ...]
      [--profile default|full|smoke] [--data-dir evalsuite/data] [--gpus 4] [--notes TEXT]
      [--save-responses tail|full|none] [--dry-run] [--resume] [--list-families]

Outputs: <out>/<tag>.json (per-family accuracy, n, 95% Wilson CI, token means, wall time, the exact items
failed), <out>/<tag>.items.jsonl (one line per attempted item), <out>/<tag>.run.json (arguments and host),
<out>/<tag>.log, and an appended long-format row set in <out>/eval_summary.tsv.

Exit codes: 0 complete and valid; 1 truncated (time budget) or invalid (>5% errors); 2 data missing or
manifest hash mismatch; 3 no live endpoint.

Families are plugins: see families/_base.py for the interface.  The runner owns the request loop,
response normalisation (<think>/reasoning stripping), the concurrency semaphore, retries, the time-budget
scheduler and the reports; a family owns its items, prompt builder and scorer.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import concurrent.futures
import dataclasses
import inspect
import json
import os
import platform
import socket
import sys
import time
import traceback
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402
from common import (DEFAULT_DATA_DIR, DEFAULT_SEED, ChatClient, ItemOutcome, RunContext, Verdict,  # noqa: E402
                    normalize_response, now_iso, wilson)
import families as famreg  # noqa: E402
from families import _base  # noqa: E402

FAMILY_TIME_FALLBACK_S = {"tools": 90.0, "code": 120.0, "math": 120.0, "longctx": 45.0, "knowledge": 40.0, "ifeval": 20.0}
PROGRESS_EVERY_S = 10.0
PARTIAL_EVERY_S = 60.0
TSV_COLUMNS = ["tag", "model", "date", "family", "sub", "n_planned", "n_attempted", "n_scored", "n_correct", "acc",
               "ci_lo", "ci_hi", "acc_strict", "acc_official", "mean_out_tok", "mean_prompt_tok", "trunc_rate",
               "wall_s", "gpu_min", "truncated", "valid", "profile", "reasoning", "max_tokens", "temperature",
               "tools_mode", "lcb_window", "manifest_sha256", "notes"]


class DataError(Exception):
    """Items missing or manifest hash mismatch (exit 2)."""


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sqwish Labs quality eval suite runner",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--tag", help="run tag; names the output files")
    p.add_argument("--base-urls", default="http://127.0.0.1:8000", help="comma-separated OpenAI-compatible base URLs")
    p.add_argument("--model", default="m", help="served model name")
    p.add_argument("--out", default="results/eval", help="output directory")
    p.add_argument("--families", default=None, help="comma-separated family names (default: all non-hidden families)")
    p.add_argument("--limit", type=int, default=None, help="first N items per family (nested subsets)")
    p.add_argument("--concurrency", type=int, default=64, help="in-flight requests across all URLs")
    p.add_argument("--max-tokens", type=int, default=None, help="completion cap (default 2048; 4096 with --reasoning)")
    p.add_argument("--max-tokens-family", default=None, help="per-family caps, e.g. tools=1024,longctx=512")
    p.add_argument("--reasoning", action="store_true", help="reasoning model: T=0.6, top_p=0.95, max_tokens 4096, reasoning caps")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="sampling seed and item-order seed")
    p.add_argument("--chat-template-kwargs", default=None, help='JSON, e.g. {"enable_thinking": false}')
    p.add_argument("--extra-body", default=None, help="JSON merged into every request body")
    p.add_argument("--time-budget", type=float, default=None, help="seconds (default 900; 300 for --profile smoke)")
    p.add_argument("--grace", type=float, default=120.0, help="seconds after the deadline before in-flight items are cancelled")
    p.add_argument("--request-timeout", type=float, default=600.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--retry-backoff", default="2,8,20", help="seconds between retries (+U(0,1))")
    p.add_argument("--profile", choices=["default", "full", "smoke"], default="default")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--family-opt", action="append", default=[], metavar="[FAMILY.]KEY=VALUE",
                   help="family option, e.g. tools.mode=native, knowledge.shots=0, longctx.fixed=true")
    p.add_argument("--gpus", type=int, default=4, help="GPUs behind the endpoints (gpu-minute accounting)")
    p.add_argument("--notes", default="")
    p.add_argument("--save-responses", choices=["tail", "full", "none"], default="tail")
    p.add_argument("--dry-run", action="store_true", help="print the plan and the first rendered request per family")
    p.add_argument("--resume", action="store_true", help="skip ids already in <tag>.items.jsonl")
    p.add_argument("--list-families", action="store_true")
    p.add_argument("--no-manifest-check", action="store_true", help="do not verify item files against data/manifest.json")
    p.add_argument("--quiet", action="store_true", help="no periodic progress lines")
    return p


def parse_family_opts(specs: list[str]) -> dict:
    """['tools.mode=native', 'shots=0'] -> {'tools': {'mode': 'native'}, '*': {'shots': 0}} (values JSON-decoded when possible)."""
    out: dict = collections.defaultdict(dict)
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--family-opt expects [family.]key=value, got {spec!r}")
        k, v = spec.split("=", 1)
        fam, key = (k.split(".", 1) if "." in k else ("*", k))
        out[fam.strip()][key.strip()] = common.json_or_str(v.strip())
    return dict(out)


def resolve_config(args: argparse.Namespace) -> argparse.Namespace:
    cfg = argparse.Namespace(**vars(args))
    cfg.base_urls = [u.strip().rstrip("/") for u in args.base_urls.split(",") if u.strip()]
    cfg.temperature = args.temperature if args.temperature is not None else (0.6 if args.reasoning else 0.0)
    cfg.top_p = args.top_p if args.top_p is not None else (0.95 if args.reasoning else 1.0)
    cfg.max_tokens = args.max_tokens if args.max_tokens is not None else (4096 if args.reasoning else 2048)
    cfg.time_budget = args.time_budget if args.time_budget is not None else (300.0 if args.profile == "smoke" else 900.0)
    cfg.limit = args.limit if args.limit is not None else (3 if args.profile == "smoke" else None)
    cfg.max_tokens_family = common.parse_kv_list(args.max_tokens_family, int)
    cfg.retry_backoff = tuple(float(x) for x in args.retry_backoff.split(",") if x.strip()) or (2.0,)
    cfg.extra_body = {}
    if args.chat_template_kwargs:
        cfg.extra_body["chat_template_kwargs"] = json.loads(args.chat_template_kwargs)
    if args.extra_body:
        cfg.extra_body.update(json.loads(args.extra_body))
    cfg.family_opts = parse_family_opts(args.family_opt)
    cfg.family_names = [f.strip() for f in args.families.split(",")] if args.families else None
    cfg.data_dir = os.path.abspath(args.data_dir)
    return cfg


def family_max_tokens(mod, cfg) -> int:
    if mod.NAME in cfg.max_tokens_family:
        return int(cfg.max_tokens_family[mod.NAME])
    d = getattr(mod, "DEFAULT_MAX_TOKENS", None)
    if isinstance(d, dict):
        return int(d.get("reasoning" if cfg.reasoning else "default", d.get("default", cfg.max_tokens)))
    if isinstance(d, int):
        return d
    return int(cfg.max_tokens)


def family_opts(mod, cfg) -> dict:
    opts = dict(cfg.family_opts.get("*", {}))
    opts.update(cfg.family_opts.get(mod.NAME, {}))
    return opts


def make_context(mod, cfg, client, log) -> RunContext:
    return RunContext(family=mod.NAME, data_dir=cfg.data_dir, cfg=cfg, client=client,
                      max_tokens=family_max_tokens(mod, cfg), reasoning=cfg.reasoning, temperature=cfg.temperature,
                      top_p=cfg.top_p, seed=cfg.seed, profile=cfg.profile, opts=family_opts(mod, cfg),
                      extra_body=dict(cfg.extra_body), log=log)


# --------------------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------------------

@dataclasses.dataclass
class Work:
    index: int
    mod: Any
    item: dict

    @property
    def family(self) -> str:
        return self.mod.NAME

    @property
    def sub(self) -> str:
        return self.item.get("subfamily", self.mod.NAME)

    @property
    def id(self) -> str:
        return self.item["id"]


def check_manifest(mod, cfg, manifest: Optional[dict], log) -> None:
    if manifest is None or cfg.no_manifest_check:
        return
    entry = (manifest.get("items") or {}).get(mod.NAME)
    if not entry:
        return
    path = os.path.join(cfg.data_dir, entry["file"])
    if not os.path.exists(path):
        raise DataError(f"{path} listed in manifest.json but missing - run prepare_data.py --only {mod.NAME}")
    digest = common.sha256_file(path)
    if digest != entry.get("sha256"):
        raise DataError(f"{path}: sha256 {digest[:12]}.. != manifest {str(entry.get('sha256'))[:12]}.. "
                        f"(rebuild with prepare_data.py or pass --no-manifest-check)")


def build_plan(mods: list, cfg, manifest: Optional[dict], log) -> tuple[list[Work], dict[str, list[dict]]]:
    per_family: dict[str, list[dict]] = {}
    for mod in mods:
        check_manifest(mod, cfg, manifest, log)
        try:
            items = mod.load_items(cfg.limit, cfg.seed, data_dir=cfg.data_dir)
        except FileNotFoundError as e:
            raise DataError(str(e))
        seen = set()
        for it in items:
            if "id" not in it:
                raise DataError(f"{mod.NAME}: item without id: {str(it)[:120]}")
            if it["id"] in seen:
                raise DataError(f"{mod.NAME}: duplicate item id {it['id']}")
            seen.add(it["id"])
            it.setdefault("family", mod.NAME)
            it.setdefault("subfamily", mod.NAME)
        per_family[mod.NAME] = items
    interleaved = _base.interleave([[(mod, it) for it in per_family[mod.NAME]] for mod in mods])
    work = [Work(i, mod, it) for i, (mod, it) in enumerate(interleaved)]
    for w in work:
        w.item["_index"] = w.index
    return work, per_family


def print_plan(mods, cfg, work, per_family, log) -> None:
    log(f"plan: {len(work)} items, families {[m.NAME for m in mods]}, limit={cfg.limit}, profile={cfg.profile}, "
        f"time_budget={cfg.time_budget:.0f}s, concurrency={cfg.concurrency}, T={cfg.temperature}, top_p={cfg.top_p}, "
        f"seed={cfg.seed}, reasoning={cfg.reasoning}")
    for mod in mods:
        items = per_family[mod.NAME]
        subs = collections.Counter(it.get("subfamily") for it in items)
        mt = family_max_tokens(mod, cfg)
        est_out = sum(min(mt, it.get("max_tokens", mt)) for it in items) * 0.5
        log(f"  {mod.NAME}: n={len(items)} subs={dict(subs)} max_tokens={mt} est_out_tokens~{int(est_out)} opts={family_opts(mod, cfg)}")
        if items:
            ctx = make_context(mod, cfg, None, log)
            try:
                msgs = mod.build_messages(items[0], ctx)
            except Exception as e:  # a family may need a live client to render (e.g. longctx calibration)
                log(f"    first request: (not renderable without an endpoint: {e})")
                continue
            log(f"    first request ({items[0]['id']}):")
            for m in msgs:
                c = common.message_text(m.get("content"))
                c = c if len(c) <= 400 else c[:200] + f" ...[{len(c) - 400} chars]... " + c[-200:]
                log(f"      [{m.get('role')}] {c!r}")


# --------------------------------------------------------------------------------------
# per-item execution
# --------------------------------------------------------------------------------------

def _call_score(mod, item: dict, text: str, meta: dict) -> Verdict:
    fn = mod.score
    accepts_meta = True
    try:
        params = list(inspect.signature(fn).parameters.values())
        accepts_meta = len(params) >= 3 or any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params)
    except (TypeError, ValueError):
        pass
    v = fn(item, text, meta) if accepts_meta else fn(item, text)
    if isinstance(v, Verdict):
        return v
    if isinstance(v, bool):
        return Verdict(v)
    if isinstance(v, dict):
        return Verdict(**v)
    raise TypeError(f"{mod.NAME}.score returned {type(v).__name__}, expected common.Verdict")


def _tail(s: str, n: int, mode: str) -> str:
    if mode == "none":
        return ""
    if mode == "full" or len(s) <= n:
        return s
    return s[-n:]


class RunState:
    def __init__(self):
        self.records: dict[str, dict] = {}
        self.resumed: set[str] = set()
        self.blocked: set[str] = set()
        self.skipped: list[Work] = []
        self.inflight = 0
        self.item_times: dict[str, list[float]] = collections.defaultdict(list)
        self.t0 = time.monotonic()
        self.started_at = now_iso()
        self.ended_at: Optional[str] = None
        self.deadline: float = float("inf")
        self.interrupted = False


async def run_one(w: Work, ctx: RunContext, client: ChatClient, cfg, state: RunState, pool, items_path: str, log) -> None:
    mod, item = w.mod, w.item
    t_start = time.monotonic()
    rec: dict = {"id": w.id, "family": w.family, "sub": w.sub, "status": "error", "correct": False, "score": 0.0,
                 "expected": None, "extracted": None, "finish_reason": None, "prompt_tokens": 0, "completion_tokens": 0,
                 "requests": 0, "retries": 0, "latency_s": 0.0, "item_s": 0.0, "base_url": None, "error": None,
                 "error_kind": None, "flags": [], "detail": {}, "content": "", "reasoning": "",
                 "t_start": round(t_start - state.t0, 3), "t_end": None}
    state.inflight += 1
    try:
        if getattr(mod, "run_item", None) is not None:
            outcome: ItemOutcome = await mod.run_item(item, ctx)
            rec.update({"prompt_tokens": outcome.prompt_tokens, "completion_tokens": outcome.completion_tokens,
                        "requests": outcome.requests, "retries": outcome.retries, "finish_reason": outcome.finish_reason,
                        "base_url": outcome.base_url, "latency_s": round(outcome.latency_s, 3), "flags": list(outcome.flags),
                        "content": _tail(outcome.content, 4000, cfg.save_responses),
                        "reasoning": _tail(outcome.reasoning, 1000, cfg.save_responses)})
            rec.update(outcome.extra or {})
            if outcome.error:
                rec["status"], rec["error"] = "error", outcome.error[:500]
            else:
                _apply_verdict(rec, outcome.verdict, outcome.finish_reason)
        else:
            messages = mod.build_messages(item, ctx)
            max_tokens = min(ctx.max_tokens, int(item.get("max_tokens", ctx.max_tokens)))
            res = await client.chat(messages, route_key=w.index, max_tokens=max_tokens, temperature=ctx.temperature,
                                    top_p=ctx.top_p, seed=cfg.seed, extra_body=ctx.extra_body or None)
            rec.update({"prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens, "requests": 1,
                        "retries": res.retries, "latency_s": round(res.latency_s, 3), "base_url": res.base_url,
                        "finish_reason": res.finish_reason})
            if not res.ok:
                rec["status"], rec["error"], rec["error_kind"] = "error", (res.error or "request failed")[:500], res.error_kind
            else:
                norm = normalize_response(res.message, res.finish_reason)
                rec["flags"] = list(norm.flags)
                rec["content"] = _tail(res.content, 4000, cfg.save_responses)
                rec["reasoning"] = _tail(norm.reasoning, 1000, cfg.save_responses)
                if norm.status == "truncated":
                    rec["status"], rec["expected"] = "truncated", item.get("answer")
                elif norm.status == "empty":
                    rec["status"], rec["expected"] = "empty", item.get("answer")
                else:
                    meta = {"finish_reason": res.finish_reason, "reasoning": norm.reasoning, "flags": norm.flags,
                            "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
                            "message": res.message}
                    loop = asyncio.get_running_loop()
                    verdict = await loop.run_in_executor(pool, _call_score, mod, item, norm.visible, meta)
                    _apply_verdict(rec, verdict, res.finish_reason)
    except asyncio.CancelledError:
        rec["status"], rec["error"], rec["correct"] = "cancelled", "cancelled at deadline + grace", False
        _finish(rec, w, t_start, state, items_path)
        raise
    except Exception as e:
        rec["status"], rec["correct"] = "error", False
        rec["error"] = f"{type(e).__name__}: {e}"[:500]
        rec["error_kind"] = "exception"
        log(f"[{w.id}] exception: {traceback.format_exc(limit=3).strip().splitlines()[-1]}")
    _finish(rec, w, t_start, state, items_path)


def _apply_verdict(rec: dict, v: Verdict, finish_reason: Optional[str]) -> None:
    rec["correct"] = bool(v.correct)
    rec["score"] = float(v.score if v.score is not None else (1.0 if v.correct else 0.0))
    rec["status"] = v.status or ("correct" if v.correct else "wrong")
    rec["extracted"], rec["expected"], rec["detail"] = v.extracted, v.expected, dict(v.detail or {})
    rec["flags"] = list(dict.fromkeys(list(rec.get("flags", [])) + list(v.flags or [])))
    if "degenerate" in rec["flags"]:
        rec["correct"], rec["score"] = False, 0.0
        rec["status"] = "truncated" if finish_reason == "length" else "wrong"
    elif finish_reason == "length" and not rec["correct"]:
        rec["status"] = "truncated"


def _finish(rec: dict, w: Work, t_start: float, state: RunState, items_path: str) -> None:
    t_end = time.monotonic()
    rec["item_s"] = round(t_end - t_start, 3)
    rec["t_end"] = round(t_end - state.t0, 3)
    state.inflight -= 1
    state.records[w.id] = rec
    state.item_times[w.family].append(rec["item_s"])
    common.append_jsonl(items_path, rec)


# --------------------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------------------

def estimate_item_s(mod, state: RunState, time_budget: float) -> float:
    """p90 of the family's observed item wall times; before three observations a fallback prior,
    capped at a quarter of the budget so a stale prior can never block a family from being sampled."""
    obs = state.item_times.get(mod.NAME, [])
    if len(obs) >= 3:
        return common.percentile(obs, 90) or 0.0
    prior = float(getattr(mod, "ITEM_TIME_FALLBACK_S", None) or FAMILY_TIME_FALLBACK_S.get(mod.NAME, 60.0))
    return min(prior, max(0.0, time_budget) / 4.0)


async def run_async(cfg, mods, work, per_family, manifest, paths, log) -> tuple[Optional[dict], int]:
    state = RunState()
    client = ChatClient(cfg.base_urls, cfg.model, concurrency=cfg.concurrency, request_timeout=cfg.request_timeout,
                        retries=cfg.retries, backoff=cfg.retry_backoff, log=log, rng_seed=cfg.seed)
    await client.open()
    try:
        probe = await client.probe()
        for u in probe["dead"]:
            log(f"warning: endpoint {u} is not answering /v1/models - dropped")
        if not probe["alive"]:
            log("error: no live endpoint")
            return None, 3
        if probe["served_models"] and cfg.model not in probe["served_models"]:
            log(f"warning: --model {cfg.model!r} not in served models {probe['served_models']}")
        await client.tokenize(prompt="probe")  # learns max_model_len when /tokenize exists
        contexts = {mod.NAME: make_context(mod, cfg, client, log) for mod in mods}
        for mod in mods:
            hook = getattr(mod, "prepare_run", None)
            if hook is not None:
                log(f"{mod.NAME}: prepare_run ...")
                await hook(per_family[mod.NAME], contexts[mod.NAME])

        run_info = {"args": {k: (v if isinstance(v, (str, int, float, bool, list, dict, type(None))) else str(v))
                             for k, v in vars(cfg).items()},
                    "hostname": socket.gethostname(), "python": platform.python_version(), "platform": platform.platform(),
                    "evalsuite_git_sha": common.evalsuite_git_sha(), "started_at": state.started_at,
                    "endpoint": {"alive": probe["alive"], "dead": probe["dead"], "served_models": probe["served_models"],
                                 "versions": probe["versions"], "max_model_len": client.max_model_len}}
        common.atomic_write_text(paths["run"], json.dumps(run_info, indent=1, sort_keys=True))

        if cfg.resume and os.path.exists(paths["items"]):
            for rec in common.read_jsonl(paths["items"]):
                if rec.get("status") in ("error", "cancelled"):
                    continue  # re-run failures
                state.records[rec["id"]] = rec
                state.resumed.add(rec["id"])
            log(f"resume: {len(state.resumed)} items already done")
            keep = [r for r in state.records.values()]
            common.write_jsonl(paths["items"], keep)
        else:
            common.atomic_write_text(paths["items"], "")

        state.t0 = time.monotonic()
        state.deadline = state.t0 + cfg.time_budget
        for ctx in contexts.values():
            ctx.deadline = state.deadline
        slots = asyncio.Semaphore(cfg.concurrency)
        tasks: set[asyncio.Task] = set()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        planned = {mod.NAME: len(per_family[mod.NAME]) for mod in mods}

        def _release(_t):
            tasks.discard(_t)
            slots.release()

        async def dispatcher():
            for w in work:
                if w.id in state.resumed:
                    continue
                if w.family in state.blocked:
                    state.skipped.append(w)
                    continue
                remaining = state.deadline - time.monotonic()
                if remaining <= 0:
                    state.blocked.add(w.family)
                    state.skipped.append(w)
                    continue
                try:
                    await asyncio.wait_for(slots.acquire(), timeout=remaining)
                except asyncio.TimeoutError:      # the deadline passed while waiting for a slot
                    state.blocked.add(w.family)
                    state.skipped.append(w)
                    continue
                now = time.monotonic()
                est = estimate_item_s(w.mod, state, cfg.time_budget)
                if now + est > state.deadline:
                    state.blocked.add(w.family)
                    state.skipped.append(w)
                    slots.release()
                    log(f"time budget: {w.family} blocked at t={now - state.t0:.0f}s (p90 est {est:.1f}s)")
                    continue
                t = asyncio.create_task(run_one(w, contexts[w.family], client, cfg, state, pool, paths["items"], log))
                tasks.add(t)
                t.add_done_callback(_release)

        async def progress():
            last_partial = time.monotonic()
            while True:
                await asyncio.sleep(PROGRESS_EVERY_S)
                now = time.monotonic()
                el = now - state.t0
                done = collections.Counter(r["family"] for r in state.records.values())
                per = " ".join(f"{f}={done.get(f, 0)}/{planned[f]}" for f in planned)
                ct = client.totals["completion_tokens"]
                if not cfg.quiet:
                    log(f"[{el:6.0f}s] {per} inflight={state.inflight} req={client.totals['requests']} "
                        f"retries={client.totals['retries']} err={client.totals['errors']} "
                        f"out_tok/s={ct / max(el, 1e-6):.0f} budget_left={state.deadline - now:.0f}s")
                if now - last_partial >= PARTIAL_EVERY_S:
                    last_partial = now
                    result = build_result(cfg, mods, work, per_family, state, client, manifest, probe, partial=True)
                    common.atomic_write_text(paths["json"], json.dumps(result, indent=1, sort_keys=True))

        prog = asyncio.create_task(progress())
        try:
            await dispatcher()
            while tasks:
                remaining = state.deadline + cfg.grace - time.monotonic()
                if remaining <= 0:
                    break
                await asyncio.wait(set(tasks), timeout=min(remaining, 5.0), return_when=asyncio.FIRST_COMPLETED)
            if tasks:
                log(f"grace period over: cancelling {len(tasks)} in-flight items")
                for t in list(tasks):
                    t.cancel()
                await asyncio.gather(*list(tasks), return_exceptions=True)
        except (KeyboardInterrupt, asyncio.CancelledError):
            state.interrupted = True
            log("interrupted: cancelling in-flight items and writing partial results")
            for t in list(tasks):
                t.cancel()
            await asyncio.gather(*list(tasks), return_exceptions=True)
        finally:
            prog.cancel()
            try:
                await prog
            except (asyncio.CancelledError, Exception):
                pass
            pool.shutdown(wait=False)
        state.ended_at = now_iso()
        result = build_result(cfg, mods, work, per_family, state, client, manifest, probe, partial=state.interrupted)
        return result, (0 if result["valid"] and not result["truncated"] else 1)
    finally:
        await client.close()


# --------------------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------------------

def _r(x: Optional[float], nd: int = 4) -> Optional[float]:
    return None if x is None else round(x, nd)


def block_stats(recs: list[dict], n_planned: int) -> dict:
    n_attempted = len(recs)
    n_error = sum(1 for r in recs if r["status"] == "error")
    n_cancelled = sum(1 for r in recs if r["status"] == "cancelled")
    scored = [r for r in recs if r["status"] not in ("error", "cancelled")]
    n_scored = len(scored)
    n_correct = sum(1 for r in scored if r.get("correct"))
    lo, hi = wilson(n_correct, n_scored)
    out_toks = [r.get("completion_tokens", 0) for r in scored]
    in_toks = [r.get("prompt_tokens", 0) for r in scored]
    item_s = [r.get("item_s", 0.0) for r in recs if r.get("item_s") is not None]
    starts = [r["t_start"] for r in recs if r.get("t_start") is not None]
    ends = [r["t_end"] for r in recs if r.get("t_end") is not None]
    return {
        "n_planned": n_planned, "n_attempted": n_attempted, "n_scored": n_scored, "n_error": n_error,
        "n_cancelled": n_cancelled, "n_skipped": max(0, n_planned - n_attempted), "n_correct": n_correct,
        "acc": _r(n_correct / n_scored) if n_scored else None, "ci95": [_r(lo), _r(hi)],
        "acc_strict": _r(n_correct / n_attempted) if n_attempted else None,
        "mean_score": _r(common.mean([float(r.get("score") or 0.0) for r in scored])),
        "status_counts": dict(sorted(collections.Counter(r["status"] for r in recs).items())),
        "mean_out_tokens": _r(common.mean(out_toks), 1), "p50_out_tokens": common.percentile(out_toks, 50),
        "mean_prompt_tokens": _r(common.mean(in_toks), 1),
        "total_out_tokens": sum(out_toks), "total_prompt_tokens": sum(in_toks),
        "trunc_rate": _r(sum(1 for r in scored if r.get("finish_reason") == "length") / n_scored) if n_scored else None,
        "degenerate_rate": _r(sum(1 for r in scored if "degenerate" in r.get("flags", [])) / n_scored) if n_scored else None,
        "answer_from_reasoning": sum(1 for r in scored if "answer_from_reasoning" in r.get("flags", [])),
        "p50_item_s": _r(common.percentile(item_s, 50), 2), "p90_item_s": _r(common.percentile(item_s, 90), 2),
        "wall_s": _r(max(ends) - min(starts), 1) if starts and ends else 0.0,
        "failed_ids": sorted(r["id"] for r in scored if not r.get("correct")),
        "error_ids": sorted(r["id"] for r in recs if r["status"] == "error"),
        "degenerate_ids": sorted(r["id"] for r in scored if "degenerate" in r.get("flags", [])),
    }


def build_result(cfg, mods, work, per_family, state: RunState, client: ChatClient, manifest, probe, partial: bool) -> dict:
    t_end = time.monotonic()
    wall_s = t_end - state.t0
    gpu_minutes_total = wall_s / 60.0 * max(1, cfg.gpus)
    recs_by_family: dict[str, list[dict]] = collections.defaultdict(list)
    for r in state.records.values():
        recs_by_family[r["family"]].append(r)
    weights = {m.NAME: sum(r.get("completion_tokens", 0) + r.get("prompt_tokens", 0) / 8.0 for r in recs_by_family[m.NAME])
               for m in mods}
    wsum = sum(weights.values()) or 1.0

    families_out: dict[str, dict] = {}
    notes: list[str] = []
    for mod in mods:
        recs = recs_by_family[mod.NAME]
        items = per_family[mod.NAME]
        fam = block_stats(recs, len(items))
        subs_planned = collections.Counter(it.get("subfamily") for it in items)
        sub_order = list(getattr(mod, "SUBFAMILIES", None) or []) + [s for s in subs_planned if s not in (getattr(mod, "SUBFAMILIES", None) or [])]
        fam["sub"] = {}
        for sf in sub_order:
            if sf not in subs_planned:
                continue
            sub_recs = [r for r in recs if r["sub"] == sf]
            sb = block_stats(sub_recs, subs_planned[sf])
            sb.pop("failed_ids", None); sb.pop("error_ids", None); sb.pop("degenerate_ids", None)
            fam["sub"][sf] = sb
        fam["gpu_minutes"] = _r(gpu_minutes_total * weights[mod.NAME] / wsum, 3)
        p = fam["acc"]
        fam["info_per_gpu_min"] = _r(p * (1 - p) * fam["n_scored"] / fam["gpu_minutes"], 3) if (p is not None and fam["gpu_minutes"]) else None
        fam["failures"] = [{"id": r["id"], "sub": r["sub"], "status": r["status"], "expected": r.get("expected"),
                            "extracted": r.get("extracted"), "finish_reason": r.get("finish_reason"),
                            "out_tokens": r.get("completion_tokens"), "error": r.get("error"),
                            "flags": r.get("flags", []), "response_tail": (r.get("content") or "")[-300:]}
                           for r in sorted(recs, key=lambda r: r["id"]) if not r.get("correct")]
        fam["max_tokens"] = family_max_tokens(mod, cfg)
        fam["opts"] = family_opts(mod, cfg)
        extra: dict = {}
        try:
            extra = dict(mod.aggregate(recs) or {})
        except Exception as e:  # never let a family's statistics kill the report
            extra = {"aggregate_error": f"{type(e).__name__}: {e}"}
        fam["extra"] = extra
        families_out[mod.NAME] = fam
        notes.extend(getattr(mod, "NOTES", None) or [])

    all_recs = list(state.records.values())
    agg = block_stats(all_recs, len(work))
    accs = [f["acc"] for f in families_out.values() if f["acc"] is not None]
    agg["acc_micro"] = agg.pop("acc")
    agg["acc_macro"] = _r(sum(accs) / len(accs)) if accs else None
    for k in ("failed_ids", "error_ids", "degenerate_ids"):
        agg.pop(k, None)
    agg.update({"requests": client.totals["requests"], "retries": client.totals["retries"],
                "request_errors": client.totals["errors"], "gpu_minutes": _r(gpu_minutes_total, 3),
                "out_tokens_per_s": _r(client.totals["completion_tokens"] / wall_s, 1) if wall_s > 0 else None,
                "per_url": {u: dict(c) for u, c in client.per_url.items()}})

    n_att = agg["n_attempted"]
    invalid: list[str] = []
    if not probe["alive"]:
        invalid.append("no live endpoint")
    if n_att and agg["n_error"] / n_att > 0.05:
        invalid.append(f"error rate {agg['n_error']}/{n_att} > 5%")
    if not n_att:
        invalid.append("no items attempted")
    truncated = bool(state.skipped) or agg["n_cancelled"] > 0 or state.interrupted

    return {
        "schema_version": common.SCHEMA_VERSION, "tag": cfg.tag, "model": cfg.model, "base_urls": cfg.base_urls,
        "started_at": state.started_at, "ended_at": state.ended_at or now_iso(), "wall_s": _r(wall_s, 1),
        "partial": partial, "truncated": truncated, "valid": not invalid, "invalid_reasons": invalid,
        "config": {"seed": cfg.seed, "concurrency": cfg.concurrency, "max_tokens": cfg.max_tokens,
                   "max_tokens_family": {m.NAME: family_max_tokens(m, cfg) for m in mods if family_max_tokens(m, cfg) != cfg.max_tokens},
                   "reasoning": cfg.reasoning, "temperature": cfg.temperature, "top_p": cfg.top_p,
                   "time_budget_s": cfg.time_budget, "grace_s": cfg.grace, "request_timeout_s": cfg.request_timeout,
                   "retries": cfg.retries, "families": [m.NAME for m in mods], "limit": cfg.limit, "profile": cfg.profile,
                   "family_opts": cfg.family_opts, "chat_template_kwargs": cfg.extra_body.get("chat_template_kwargs"),
                   "extra_body": {k: v for k, v in cfg.extra_body.items() if k != "chat_template_kwargs"} or None,
                   "gpus": cfg.gpus, "notes": cfg.notes, "resume": cfg.resume},
        "data_manifest_sha256": (manifest or {}).get("manifest_sha256"),
        "lcb_window": (manifest or {}).get("lcb_window"),
        "evalsuite_git_sha": common.evalsuite_git_sha(),
        "endpoint": {"served_models": probe["served_models"], "vllm_version": next(iter(probe["versions"].values()), None),
                     "urls_alive": len(probe["alive"]), "urls_dead": probe["dead"], "max_model_len": client.max_model_len},
        "families": families_out,
        "aggregate": agg,
        "skipped_ids": sorted(w.id for w in state.skipped),
        "notes": list(dict.fromkeys(notes)),
    }


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v).replace("\t", " ").replace("\n", " ")


def append_tsv(path: str, result: dict) -> int:
    new = not os.path.exists(path)
    date = result["started_at"][:10]
    cfg = result["config"]
    tools_mode = (result["families"].get("tools", {}).get("opts", {}) or {}).get("mode", "-")
    lcb = result.get("lcb_window")
    lcb_s = f"{lcb.get('after')}:{lcb.get('hard_available')}" if isinstance(lcb, dict) else "-"
    rows: list[list[str]] = []

    def row(family: str, sub: str, b: dict, acc_key: str = "acc", official=None, max_tokens=None):
        return [result["tag"], result["model"], date, family, sub, b["n_planned"], b["n_attempted"], b["n_scored"],
                b["n_correct"], b.get(acc_key), b["ci95"][0], b["ci95"][1], b.get("acc_strict"), official,
                b.get("mean_out_tokens"), b.get("mean_prompt_tokens"), b.get("trunc_rate"), b.get("wall_s"),
                b.get("gpu_minutes"), result["truncated"], result["valid"], cfg["profile"], cfg["reasoning"],
                max_tokens if max_tokens is not None else cfg["max_tokens"], cfg["temperature"], tools_mode, lcb_s,
                result.get("data_manifest_sha256"), cfg.get("notes") or ""]

    for fam, b in result["families"].items():
        official = b.get("extra", {}).get("acc_official")
        rows.append(row(fam, "-", b, official=official, max_tokens=b.get("max_tokens")))
        for sf, sb in b.get("sub", {}).items():
            rows.append(row(fam, sf, sb, official=(sb.get("extra") or {}).get("acc_official"), max_tokens=b.get("max_tokens")))
    rows.append(row("aggregate", "-", result["aggregate"], acc_key="acc_micro"))
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        if new:
            f.write("\t".join(TSV_COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join(_fmt(v) for v in r) + "\n")
    return len(rows)


def print_summary(result: dict, log) -> None:
    log(f"== {result['tag']} model={result['model']} wall={result['wall_s']}s truncated={result['truncated']} "
        f"valid={result['valid']} {result['invalid_reasons'] or ''}")
    for fam, b in result["families"].items():
        acc = "-" if b["acc"] is None else f"{b['acc']:.3f}"
        log(f"  {fam:10s} acc={acc} n={b['n_scored']}/{b['n_planned']} ci95=[{b['ci95'][0]:.3f},{b['ci95'][1]:.3f}] "
            f"out_tok={b['mean_out_tokens']} trunc={b['trunc_rate']} err={b['n_error']} "
            f"{'skipped=' + str(b['n_skipped']) if b['n_skipped'] else ''}")
        for sf, sb in b.get("sub", {}).items():
            acc = "-" if sb["acc"] is None else f"{sb['acc']:.3f}"
            log(f"    {sf:16s} acc={acc} n={sb['n_scored']}/{sb['n_planned']} ci95=[{sb['ci95'][0]:.3f},{sb['ci95'][1]:.3f}]")
    a = result["aggregate"]
    acc = "-" if a["acc_micro"] is None else f"{a['acc_micro']:.3f}"
    log(f"  aggregate  acc_micro={acc} acc_macro={a['acc_macro']} n={a['n_scored']}/{a['n_planned']} "
        f"ci95={a['ci95']} requests={a['requests']} retries={a['retries']} out_tok/s={a['out_tokens_per_s']}")


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

class Logger:
    def __init__(self, path: Optional[str]):
        self.path = path
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"{now_iso()} {msg}"
        print(line, file=sys.stderr, flush=True)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_families:
        for name in famreg.discover():
            mod = famreg.load(name)
            print(f"{name:12s} priority={famreg.priority(mod):3d} hidden={mod.HIDDEN} subs={mod.SUBFAMILIES} - {mod.DESCRIPTION}")
        return 0
    if not args.tag:
        print("error: --tag is required", file=sys.stderr)
        return 2
    cfg = resolve_config(args)
    out_dir = os.path.abspath(cfg.out)
    paths = {"json": os.path.join(out_dir, f"{cfg.tag}.json"), "items": os.path.join(out_dir, f"{cfg.tag}.items.jsonl"),
             "run": os.path.join(out_dir, f"{cfg.tag}.run.json"), "log": os.path.join(out_dir, f"{cfg.tag}.log"),
             "tsv": os.path.join(out_dir, "eval_summary.tsv")}
    log = Logger(None if cfg.dry_run else paths["log"])

    try:
        mods = famreg.resolve(cfg.family_names, require=("score",))
    except Exception as e:
        log(f"error: cannot load families {cfg.family_names}: {e}")
        return 2
    manifest = common.load_manifest(cfg.data_dir)
    if manifest is None:
        log(f"warning: {os.path.join(cfg.data_dir, 'manifest.json')} not found - item files are not hash-verified")
    try:
        work, per_family = build_plan(mods, cfg, manifest, log)
    except DataError as e:
        log(f"error: {e}")
        return 2
    if not work:
        log("error: no items to run")
        return 2
    if cfg.dry_run:
        print_plan(mods, cfg, work, per_family, log)
        return 0

    log(f"run {cfg.tag}: {len(work)} items, families {[m.NAME for m in mods]}, urls {cfg.base_urls}, "
        f"concurrency {cfg.concurrency}, budget {cfg.time_budget:.0f}s")
    try:
        result, code = asyncio.run(run_async(cfg, mods, work, per_family, manifest, paths, log))
    except KeyboardInterrupt:
        log("interrupted before results could be written")
        return 1
    if result is None:
        return code
    common.atomic_write_text(paths["json"], json.dumps(result, indent=1, sort_keys=True))
    n_rows = append_tsv(paths["tsv"], result)
    print_summary(result, log)
    log(f"wrote {paths['json']}, {paths['items']}, +{n_rows} rows in {paths['tsv']} -> exit {code}")
    return code


if __name__ == "__main__":
    sys.exit(main())
