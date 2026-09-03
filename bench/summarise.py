#!/usr/bin/env python3
"""
summarise.py — aggregate `vllm bench serve` result JSONs and nvidia-smi dmon CSVs into
results/summary.md (+ summary.csv, summary.json).

Inputs, per cell directory results/<cell>[__tag]/ :
  <run_id>.meta.json          written by sweep.sh (shape, concurrency, ports, engine, hw flags ...)
  <run_id>[__pPORT].json      `vllm bench serve --save-result` output (one per replica port);
                              keys read: request_throughput, output_throughput,
                              total_token_throughput, completed, num_prompts, duration,
                              total_input_tokens, total_output_tokens, mean_/median_/p50_/p99_
                              ttft|tpot|itl|e2el _ms (verified against vllm main 2026-09-02)
  <run_id>.dmon.csv           `nvidia-smi dmon -s pucm -o DT` capture for the run
  <run_id>.skipped.json       runs sweep.sh skipped for the time budget (listed, not aggregated)
  launch.json / loadtest.json from launch.sh (engine version, time-to-ready, load errors,
                              hardware decisions)
A results/<dir> counts as a cell only if it holds one of those marker files; results/hw,
results/cotenancy (lora_cotenant.sh serve_*.json), results/probe etc. are skipped.

Hardware decision flags (bench/env.sh <- results/hw/decisions.env) travel in every meta /
bench JSON: p2p_ok, p2p_disabled, custom_allreduce, acs_suspected, pessimistic_tp.
Rows with pessimistic_tp == 1 (TP>1 on a box where PCIe ACS is suspected) get a dagger (†):
they are a LOWER BOUND for a node without ACS / with NVLink.  TP1 replica rows never do.
TP>1 rows whose flags are null / "unknown" (no decisions.env when the run was made) get '?'.

Per cell you get: a compact concurrency x shape table of output tok/s, then one detailed
table per shape (req/s, output tok/s, total tok/s, TTFT p50/p99, TPOT p50/p99, mean GPU
power, tok/J, peak GPU memory, $/1M output tokens).  x4 replica runs are summed across
ports (throughput), p50s are completion-weighted means, p99s are the max over ports.

Usage:
  python3 bench/summarise.py [--results-dir DIR] [--out FILE] [--cost-per-hour USD] [--quiet]
Cost column: --cost-per-hour or $COST_PER_HOUR = USD per hour for the WHOLE machine.
Python 3.8+ stdlib only.
"""
import argparse
import csv
import datetime as dt
import glob
import json
import os
import re
import sys

SHAPE_ORDER = ["router", "judge", "agent"]
RUN_RE = re.compile(r"^(?P<run>.+?__c\d+__\d{8}T\d{6})(?:__p(?P<port>\d+))?$")
SKIP_JSON = {"launch.json", "loadtest.json", "smoke.json"}
DAGGER = "†"
UNKNOWN = "?"      # TP>1 row but acs_suspected / pessimistic_tp unknown (no decisions.env at run time)


# ----------------------------------------------------------------------------- io helpers
def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def first(d, *keys):
    for k in keys:
        if isinstance(d, dict) and d.get(k) not in (None, ""):
            return d[k]
    return None


def flag01(v, default=0):
    """0/1 from '1', 1, 'true', True, ... (strings as written by --metadata)."""
    if v is None or v == "":
        return default
    if isinstance(v, bool):
        return int(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return 1
    if s in ("0", "false", "no", "off"):
        return 0
    return default


def flag01n(v):
    """0/1, or None for unknown ('', None, 'unknown', '?', 'null' -- what sweep.sh/launch.sh write without decisions.env)."""
    if v is None or str(v).strip().lower() in ("", "unknown", "?", "null", "none"):
        return None
    return flag01(v, None)


# ----------------------------------------------------------------------------- dmon parsing
def parse_dmon(path):
    """Parse `nvidia-smi dmon -s pucm -o DT` output. Returns None if no samples."""
    cols, rows = None, []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    parts = line[1:].split()
                    if cols is None and "gpu" in parts:
                        cols = parts  # first header row = names; second (#Idx W C ...) = units
                    continue
                if cols is None:
                    continue
                vals = line.split()
                if len(vals) != len(cols):
                    continue
                rows.append(dict(zip(cols, vals)))
    except OSError:
        return None
    if not rows:
        return None
    # group rows into per-timestamp samples
    groups, last = [], None
    for r in rows:
        g = fnum(r.get("gpu"))
        if "Time" in r:
            key = (r.get("Date", ""), r.get("Time", ""))
            if not groups or groups[-1][0] != key:
                groups.append((key, []))
        else:
            if not groups or (g is not None and last is not None and g <= last):
                groups.append((None, []))
        groups[-1][1].append(r)
        last = g
    total_w, sm_util, mem_util = [], [], []
    peak_fb = {}
    for _, rs in groups:
        pw = [fnum(r.get("pwr")) for r in rs]
        pw = [p for p in pw if p is not None]
        if pw:
            total_w.append(sum(pw))
        for r in rs:
            s = fnum(r.get("sm"))
            if s is not None:
                sm_util.append(s)
            m = fnum(r.get("mem"))
            if m is not None:
                mem_util.append(m)
            fb = fnum(r.get("fb"))
            gid = r.get("gpu")
            if fb is not None and gid is not None:
                peak_fb[gid] = max(peak_fb.get(gid, 0.0), fb)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None  # noqa: E731
    return {
        "samples": len(rows),
        "timestamps": len(groups),
        "mean_total_w": mean(total_w),
        "peak_total_w": max(total_w) if total_w else None,
        "mean_sm_util_pct": mean(sm_util),
        "mean_membw_util_pct": mean(mem_util),
        "peak_fb_mib_per_gpu": peak_fb,
        "peak_fb_mib": max(peak_fb.values()) if peak_fb else None,
    }


# ----------------------------------------------------------------------------- discovery
def discover(results_dir):
    cells = {}
    for cell_dir in sorted(p for p in glob.glob(os.path.join(results_dir, "*")) if os.path.isdir(p)):
        name = os.path.basename(cell_dir)
        # not cells: hardware truth, gate/train scratch, lora_cotenant.sh's serve_*.json (they carry
        # request_throughput but belong to results/cotenancy.json), the pre-existing probe/ directory
        if name in ("hw", "hardware", "gates", "train", "instance-logs", "cotenancy", "probe", "warmup") or name.startswith("_"):
            continue
        # a cell directory is one that launch.sh / sweep.sh wrote into (launch.json, loadtest.json,
        # *.meta.json or *.skipped.json); stray directories holding bench-like JSONs are ignored
        if not (os.path.exists(os.path.join(cell_dir, "launch.json")) or os.path.exists(os.path.join(cell_dir, "loadtest.json"))
                or glob.glob(os.path.join(cell_dir, "*.meta.json")) or glob.glob(os.path.join(cell_dir, "*.skipped.json"))):
            continue
        info = {
            "name": name,
            "dir": cell_dir,
            "runs": {},
            "skipped": [],
            "launch": load_json(os.path.join(cell_dir, "launch.json")),
            "loadtest": load_json(os.path.join(cell_dir, "loadtest.json")),
        }
        for p in sorted(glob.glob(os.path.join(cell_dir, "*.meta.json"))):
            m = load_json(p) or {}
            rid = m.get("run_id") or os.path.basename(p)[: -len(".meta.json")]
            info["runs"][rid] = {"meta": m, "bench": [], "dmon": None}
        for p in sorted(glob.glob(os.path.join(cell_dir, "*.skipped.json"))):
            s = load_json(p)
            if isinstance(s, dict):
                info["skipped"].append(s)
        for p in sorted(glob.glob(os.path.join(cell_dir, "*.json"))):
            b = os.path.basename(p)
            if b.endswith(".meta.json") or b.endswith(".skipped.json") or b in SKIP_JSON:
                continue
            j = load_json(p)
            if not isinstance(j, dict) or "request_throughput" not in j:
                continue
            stem = b[:-5]
            mm = RUN_RE.match(stem)
            rid = j.get("run_id") or (mm.group("run") if mm else stem)
            run = info["runs"].setdefault(rid, {"meta": {}, "bench": [], "dmon": None})
            j["_file"] = b
            run["bench"].append(j)
        for p in glob.glob(os.path.join(cell_dir, "*.dmon.csv")):
            rid = os.path.basename(p)[: -len(".dmon.csv")]
            if rid in info["runs"]:
                info["runs"][rid]["dmon"] = parse_dmon(p)
        if info["runs"] or info["launch"] or info["loadtest"] or info["skipped"]:
            cells[name] = info
    return cells


# ----------------------------------------------------------------------------- aggregation
def aggregate(cell, rid, run, cost_per_hour, launch=None):
    bench, meta = run["bench"], run["meta"] or {}
    launch = launch or {}
    if not bench:
        return None
    b0 = bench[0]

    def tot(*keys):
        return sum(fnum(first(j, *keys)) or 0.0 for j in bench)

    completed = int(tot("completed"))
    num_prompts = int(tot("num_prompts"))
    weights = [fnum(first(j, "completed")) or 0.0 for j in bench]

    def wmean(*keys):
        pairs = [(fnum(first(j, *keys)), w) for j, w in zip(bench, weights)]
        pairs = [(v, w) for v, w in pairs if v is not None and w > 0]
        if not pairs:
            return None
        return sum(v * w for v, w in pairs) / sum(w for _, w in pairs)

    def wmax(*keys):
        vals = [fnum(first(j, *keys)) for j in bench]
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None

    def pick(key, default=None):
        v = first(meta, key)
        if v is None:
            v = first(b0, key)
        if v is None:
            v = launch.get(key)
        return default if v is None else v

    shape = first(meta, "shape") or first(b0, "shape") or "unknown"
    conc = first(meta, "concurrency") or first(b0, "concurrency") or first(b0, "max_concurrency")
    conc = int(fnum(conc) or 0)
    in_len = first(meta, "in_len") or first(b0, "in_len")
    out_len = first(meta, "out_len") or first(b0, "out_len")
    if in_len in (None, "dataset") and completed:
        ti = tot("total_input_tokens")
        in_len = f"~{ti / completed:.0f}" if ti else in_len
    if out_len is None and completed:
        to = tot("total_output_tokens")
        out_len = f"~{to / completed:.0f}" if to else None

    out_tps = tot("output_throughput")
    tot_tps = tot("total_token_throughput")
    req_s = tot("request_throughput")
    dmon = run.get("dmon") or {}
    mean_w = dmon.get("mean_total_w")
    peak_mib = dmon.get("peak_fb_mib")
    cost_out = cost_tot = None
    if cost_per_hour and out_tps:
        cost_out = cost_per_hour / 3600.0 / out_tps * 1e6
    if cost_per_hour and tot_tps:
        cost_tot = cost_per_hour / 3600.0 / tot_tps * 1e6

    tp = pick("tp")
    tp_i = int(fnum(tp) or 1)
    acs = flag01n(pick("acs_suspected"))                 # 1 / 0 / None (unknown)
    # pessimistic_tp as written by sweep.sh/launch.sh is already per-cell (TP>1 and box flag);
    # old files without it fall back to acs_suspected; TP1 rows are never pessimistic; unknown stays None.
    pess_raw = flag01n(pick("pessimistic_tp"))
    if tp_i <= 1:
        pessimistic = 0
    elif pess_raw is not None:
        pessimistic = pess_raw
    else:
        pessimistic = acs                                 # 1 / 0 / None
    return {
        "cell": cell,
        "run_id": rid,
        "run_tag": first(meta, "run_tag") or first(b0, "run_tag") or "",
        "engine": pick("engine", ""),
        "engine_version": pick("engine_version", ""),
        "model": first(meta, "model") or first(b0, "model") or first(b0, "model_id") or launch.get("model") or "",
        "model_path": pick("model_path", ""),
        "tp": tp,
        "dp": pick("dp"),
        "replicas": pick("replicas"),
        "kv_cache_dtype": pick("kv_cache_dtype", ""),
        "max_num_batched_tokens": pick("max_num_batched_tokens"),
        "mode": first(meta, "mode") or first(b0, "mode") or "",
        "ports": first(meta, "ports") or "",
        "p2p_ok": flag01(pick("p2p_ok"), 1),
        "p2p_disabled": flag01(pick("p2p_disabled"), 0),
        "custom_allreduce": flag01(pick("custom_allreduce"), 0),
        "acs_suspected": acs,
        "pessimistic_tp": pessimistic,
        "spec_decoding": pick("spec_decoding", "") or "",
        "shape": shape,
        "concurrency": conc,
        "in_len": in_len,
        "out_len": out_len,
        "num_prompts": num_prompts,
        "completed": completed,
        "duration_s": wmax("duration"),
        "req_s": req_s,
        "out_tok_s": out_tps,
        "total_tok_s": tot_tps,
        "ttft_p50_ms": wmean("p50_ttft_ms", "median_ttft_ms"),
        "ttft_p99_ms": wmax("p99_ttft_ms"),
        "ttft_mean_ms": wmean("mean_ttft_ms"),
        "tpot_p50_ms": wmean("p50_tpot_ms", "median_tpot_ms"),
        "tpot_p99_ms": wmax("p99_tpot_ms"),
        "tpot_mean_ms": wmean("mean_tpot_ms"),
        "itl_p99_ms": wmax("p99_itl_ms"),
        "e2el_p50_ms": wmean("p50_e2el_ms", "median_e2el_ms"),
        "e2el_p99_ms": wmax("p99_e2el_ms"),
        "mean_gpu_w": mean_w,
        "peak_gpu_w": dmon.get("peak_total_w"),
        "mean_sm_util_pct": dmon.get("mean_sm_util_pct"),
        "tok_per_joule": (out_tps / mean_w) if (mean_w and out_tps) else None,
        "peak_mem_gib": (peak_mib / 1024.0) if peak_mib else None,
        "peak_mem_gib_per_gpu": {k: v / 1024.0 for k, v in (dmon.get("peak_fb_mib_per_gpu") or {}).items()},
        "cost_per_1m_out_usd": cost_out,
        "cost_per_1m_total_usd": cost_tot,
        "bench_files": len(bench),
        "bench_exit_code": first(meta, "bench_exit_code"),
    }


# ----------------------------------------------------------------------------- formatting
def f0(v):
    return "n/a" if v is None else f"{v:,.0f}"


def f1(v):
    return "n/a" if v is None else f"{v:,.1f}"


def f2(v):
    return "n/a" if v is None else f"{v:,.2f}"


def f3(v):
    return "n/a" if v is None else f"{v:,.3f}"


def dag(r):
    """† for pessimistic TP>1 rows, ? for TP>1 rows whose flag is unknown, '' otherwise (TP1 never marked)."""
    p = r.get("pessimistic_tp")
    if p == 1:
        return DAGGER
    if p is None and int(fnum(r.get("tp")) or 1) > 1:
        return UNKNOWN
    return ""


def shape_sort_key(s):
    return (SHAPE_ORDER.index(s) if s in SHAPE_ORDER else len(SHAPE_ORDER), s)


def shape_label(rows_for_shape, shape):
    r = rows_for_shape[0]
    if r["in_len"] is not None and r["out_len"] is not None:
        return f"{shape} ({r['in_len']} in / {r['out_len']} out)"
    return shape


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def cell_header_lines(info, rows):
    launch = info.get("launch") or {}
    r0 = rows[0] if rows else {}
    bits = []
    eng = r0.get("engine") or launch.get("engine") or ""
    ver = r0.get("engine_version") or launch.get("engine_version") or ""
    if eng or ver:
        bits.append(f"engine `{eng} {ver}`".strip())
    model = r0.get("model") or launch.get("model")
    if model:
        bits.append(f"model `{model}`")
    mp = r0.get("model_path") or launch.get("model_path")
    if mp and mp != model:
        bits.append(f"weights `{mp}`")
    tp, dp, rep = r0.get("tp") or launch.get("tp"), r0.get("dp") or launch.get("dp"), r0.get("replicas") or launch.get("replicas")
    if tp or dp or rep:
        bits.append(f"TP{tp} x DP{dp} x {rep} replica(s)")
    gpus = launch.get("gpu_ids")
    if gpus:
        bits.append(f"GPUs {gpus}")
    kv = r0.get("kv_cache_dtype") or launch.get("kv_cache_dtype")
    if kv:
        bits.append(f"kv `{kv}`")
    mnbt = r0.get("max_num_batched_tokens") or launch.get("max_num_batched_tokens")
    if mnbt:
        bits.append(f"max-num-batched-tokens {mnbt}")
    if launch.get("max_num_seqs"):
        bits.append(f"max-num-seqs {launch['max_num_seqs']}")
    if launch.get("max_model_len"):
        bits.append(f"max-model-len {launch['max_model_len']}")
    lines = ["- " + "; ".join(bits)] if bits else []
    if launch.get("seconds_to_ready") is not None:
        lines.append(f"- time to /health: {launch['seconds_to_ready']} s (status `{launch.get('status')}`)")
    if launch.get("kv_cache_line") or launch.get("max_concurrency_line"):
        lines.append(f"- server KV capacity: {launch.get('kv_cache_line', '')} {launch.get('max_concurrency_line', '')}".rstrip())
    # hardware decision flags (launch.json first, then the rows)
    flags = []
    p2p_dis = flag01(launch.get("nccl_p2p_disable", launch.get("p2p_disabled", r0.get("p2p_disabled"))), 0)
    flags.append("NCCL_P2P_DISABLE=1 (explicit human decision / A/B)" if p2p_dis else "P2P on")
    car = flag01(launch.get("custom_allreduce", r0.get("custom_allreduce")), 0)
    flags.append("custom all-reduce ON (A/B)" if car else "custom all-reduce off (default)")
    if launch.get("enable_ep") in (1, "1"):
        flags.append("expert-parallel ON")
    spec = launch.get("spec_decoding") or r0.get("spec_decoding") or ""
    if spec == "on" or launch.get("spec_config"):
        flags.append(f"speculative decoding ON ({launch.get('spec_config', '')})")
    else:
        flags.append("spec decoding off")
    acs = flag01n(launch.get("acs_suspected", r0.get("acs_suspected")))
    pess = flag01n(launch.get("pessimistic_tp", r0.get("pessimistic_tp")))
    tp_i = int(fnum(tp) or 1)
    if pess == 1 and tp_i > 1:
        flags.append(f"{DAGGER} pessimistic: TP{tp_i} on a host with PCIe ACS suspected (lower bound)")
    elif tp_i > 1 and pess is None and acs is None:
        flags.append(f"{UNKNOWN} TP{tp_i} but acs_suspected/pessimistic_tp unknown (no results/hw/decisions.env at launch)")
    elif acs == 1 and tp_i <= 1:
        flags.append("TP1 replicas: unaffected by ACS")
    lines.append("- flags: " + ", ".join(flags))
    if launch.get("ram_warning"):
        lines.append(f"- host RAM warning at launch: {launch['ram_warning']}")
    if info.get("skipped"):
        sk = ", ".join(f"{s.get('shape')} C{s.get('concurrency')} (~{s.get('est_minutes')} min)" for s in info["skipped"])
        lines.append(f"- skipped for the time budget (MAX_RUN_MINUTES): {sk}")
    return lines


# ----------------------------------------------------------------------------- main
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=os.environ.get("RESULTS_ROOT", os.path.join(os.path.dirname(here), "results")))
    ap.add_argument("--out", default=None, help="markdown output (default <results-dir>/summary.md)")
    ap.add_argument("--cost-per-hour", type=float, default=fnum(os.environ.get("COST_PER_HOUR")) or None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    results_dir = os.path.abspath(a.results_dir)
    out_md = a.out or os.path.join(results_dir, "summary.md")

    cells = discover(results_dir)
    rows_by_cell = {}
    all_rows = []
    for name, info in cells.items():
        rows = []
        for rid, run in info["runs"].items():
            r = aggregate(name, rid, run, a.cost_per_hour, info.get("launch"))
            if r:
                rows.append(r)
        rows.sort(key=lambda r: (shape_sort_key(r["shape"]), r["concurrency"], r["run_id"]))
        rows_by_cell[name] = rows
        all_rows.extend(rows)
    any_dagger = any(r["pessimistic_tp"] == 1 for r in all_rows)
    any_unknown = any(dag(r) == UNKNOWN for r in all_rows)
    any_acs = any(r["acs_suspected"] == 1 for r in all_rows) or any(
        flag01n((i.get("launch") or {}).get("acs_suspected")) == 1 for i in cells.values())

    has_cost = a.cost_per_hour is not None
    md = [f"# 4x RTX PRO 6000 Blackwell (sm_120) - serving throughput summary", ""]
    md.append(f"Generated {dt.datetime.now().isoformat(timespec='seconds')} from `{results_dir}`.  ")
    md.append("Throughput columns are at saturation (`--request-rate inf`, fixed `--max-concurrency`); TTFT/TPOT are logged only.  ")
    if has_cost:
        md.append(f"Cost assumes **${a.cost_per_hour:.2f}/hr** for the whole machine (COST_PER_HOUR).  ")
    else:
        md.append("Set `COST_PER_HOUR=<usd>` (or `--cost-per-hour`) to add the $/1M-token column.  ")
    md.append("x4 replica cells: throughput summed over ports; TTFT/TPOT p50 = completion-weighted mean, p99 = max over ports.  ")
    if any_acs or any_dagger:
        md.append(f"{DAGGER} = `pessimistic_tp`: TP>1 on a host where PCIe ACS is suspected (switch-local P2P redirected through "
                  "the root complex; all_reduce ~19-38 GB/s). These rows are a **lower bound** for a Scan node without ACS / with NVLink. "
                  "TP1 replica cells are unaffected. P2P stays enabled and custom all-reduce is off unless the flags line says otherwise.")
    if any_unknown:
        md.append(f"{UNKNOWN} = TP>1 row recorded without `results/hw/decisions.env` (acs_suspected / pessimistic_tp unknown): "
                  "run `vast/hardware_truth.sh` and re-read these rows as pessimistic if it reports ACS_SUSPECTED=1.")
    md.append("")

    # ---- headline: best output tok/s per cell x shape
    shapes_all = sorted({r["shape"] for r in all_rows}, key=shape_sort_key)
    if all_rows:
        md.append("## Peak output tok/s per cell (best concurrency per shape)")
        md.append("")
        hdr = ["cell", "layout", "engine"] + shapes_all
        trs = []
        for name, rows in rows_by_cell.items():
            if not rows:
                continue
            best = {}
            for r in rows:
                if r["out_tok_s"] and (r["shape"] not in best or r["out_tok_s"] > best[r["shape"]]["out_tok_s"]):
                    best[r["shape"]] = r
            eng = f"{rows[0]['engine']} {rows[0]['engine_version']}".strip()
            r0 = rows[0]
            layout = f"TP{r0.get('tp')} x DP{r0.get('dp')} x {r0.get('replicas')}"
            line = [f"`{name}`{dag(r0)}", layout, eng]
            for s in shapes_all:
                b = best.get(s)
                line.append(f"{f0(b['out_tok_s'])} @c{b['concurrency']}{dag(b)}" if b else "-")
            trs.append(line)
        md.append(md_table(hdr, trs))
        md.append("")

    # ---- load-test-only cells
    lt = [(n, i["loadtest"]) for n, i in cells.items() if i.get("loadtest")]
    if lt:
        md.append("## Attempt-to-load cells")
        md.append("")
        trs = []
        for n, l in lt:
            err = (l.get("error_excerpt") or "").replace("|", "\\|")
            trs.append([f"`{n}`", f"`{l.get('model', '')}`", l.get("status", ""), l.get("seconds_to_ready", ""),
                        (l.get("kv_cache_line") or "")[:60], err[:300] or "-"])
        md.append(md_table(["cell", "model", "status", "seconds", "kv cache", "error excerpt"], trs))
        md.append("")

    # ---- per cell
    for name, rows in rows_by_cell.items():
        info = cells[name]
        if not rows and not info.get("launch"):
            continue
        md.append(f"## {name}{dag(rows[0]) if rows else ''}")
        md.append("")
        md.extend(cell_header_lines(info, rows))
        md.append("")
        if not rows:
            md.append("_no sweep results yet_")
            md.append("")
            continue
        shapes = sorted({r["shape"] for r in rows}, key=shape_sort_key)
        concs = sorted({r["concurrency"] for r in rows})
        # compact: concurrency x shape -> output tok/s (latest run wins for duplicates)
        grid = {}
        for r in rows:
            grid[(r["concurrency"], r["shape"])] = r
        md.append("### Output tok/s by concurrency")
        md.append("")
        trs = []
        for c in concs:
            line = [c]
            for s in shapes:
                r = grid.get((c, s))
                line.append(f"{f0(r['out_tok_s'])}{dag(r)}" if r else "-")
            trs.append(line)
        md.append(md_table(["C"] + shapes, trs))
        md.append("")
        # detailed per shape
        for s in shapes:
            srows = [r for r in rows if r["shape"] == s]
            md.append(f"### {shape_label(srows, s)}")
            md.append("")
            hdr = ["C", "req/s", "out tok/s", "total tok/s", "TTFT p50/p99 ms", "TPOT p50/p99 ms",
                   "mean W", "tok/J", "peak mem GiB"]
            if has_cost:
                hdr.append("$/1M out tok")
            hdr.append("done")
            trs = []
            for r in srows:
                done = f"{r['completed']}/{r['num_prompts']}"
                if r["bench_exit_code"] not in (None, 0, "0"):
                    done += " (client rc=%s)" % r["bench_exit_code"]
                if r["p2p_disabled"] == 1:
                    done += " P2P-off"
                if r["custom_allreduce"] == 1:
                    done += " CAR-on"
                line = [f"{r['concurrency']}{dag(r)}", f2(r["req_s"]), f0(r["out_tok_s"]), f0(r["total_tok_s"]),
                        f"{f0(r['ttft_p50_ms'])} / {f0(r['ttft_p99_ms'])}",
                        f"{f1(r['tpot_p50_ms'])} / {f1(r['tpot_p99_ms'])}",
                        f0(r["mean_gpu_w"]), f1(r["tok_per_joule"]), f1(r["peak_mem_gib"])]
                if has_cost:
                    line.append(f3(r["cost_per_1m_out_usd"]))
                line.append(done)
                trs.append(line)
            md.append(md_table(hdr, trs))
            md.append("")

    if any_dagger or any_unknown:
        md.append("---")
        md.append(f"{DAGGER} pessimistic_tp = 1: TP>1 on a host with PCIe ACS suspected -> lower bound. "
                  f"{UNKNOWN} = TP>1 with the flag unknown (no decisions.env). "
                  "`P2P-off` = NCCL_P2P_DISABLE=1 was set explicitly for that run (human decision / A/B); `CAR-on` = custom all-reduce A/B.")
        md.append("")

    text = "\n".join(md) + "\n"
    os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(text)

    # flat csv + json next to the markdown
    base = os.path.splitext(out_md)[0]
    cols = ["cell", "run_tag", "engine", "engine_version", "model", "model_path", "tp", "dp", "replicas", "kv_cache_dtype",
            "max_num_batched_tokens", "mode", "shape", "in_len", "out_len", "concurrency", "num_prompts", "completed",
            "duration_s", "req_s", "out_tok_s", "total_tok_s", "ttft_p50_ms", "ttft_p99_ms", "ttft_mean_ms",
            "tpot_p50_ms", "tpot_p99_ms", "tpot_mean_ms", "itl_p99_ms", "e2el_p50_ms", "e2el_p99_ms", "mean_gpu_w", "peak_gpu_w",
            "mean_sm_util_pct", "tok_per_joule", "peak_mem_gib", "cost_per_1m_out_usd", "cost_per_1m_total_usd",
            "p2p_ok", "p2p_disabled", "custom_allreduce", "acs_suspected", "pessimistic_tp", "spec_decoding",
            "bench_exit_code", "run_id"]
    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump({"generated": dt.datetime.now().isoformat(timespec="seconds"), "cost_per_hour": a.cost_per_hour,
                   "dagger": f"{DAGGER} = pessimistic_tp (TP>1 with PCIe ACS suspected; lower bound); {UNKNOWN} = TP>1, flag unknown (no decisions.env); pessimistic_tp null = unknown",
                   "cells": {n: {"launch": i.get("launch"), "loadtest": i.get("loadtest"), "skipped": i.get("skipped")}
                             for n, i in cells.items()},
                   "rows": all_rows}, f, indent=2, default=str)
    if not a.quiet:
        sys.stdout.write(text)
    sys.stderr.write(f"wrote {out_md}, {base}.csv, {base}.json ({len(all_rows)} runs, {len(cells)} cells)\n")


if __name__ == "__main__":
    main()
