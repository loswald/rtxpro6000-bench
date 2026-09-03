#!/usr/bin/env python3
"""
gates/gates_summary.py [--results-dir DIR] [--out FILE]

Markdown summary of the correctness gates and the co-tenancy / sleep-wake cells,
complementing bench/summarise.py (throughput tables):

  results/hw/decisions.env               -> hardware decision line (ACS_SUSPECTED, P2P_OK, ...)
  results/<cell>[__tag]/gsm8k.json       -> GSM8K strict / flexible exact match
  results/<cell>[__tag]/kv_diff.json     -> pass, exact-match rate, mean NED, corruption counts
                                            (or status launch_failed / not_applicable with the reason)
  results/<cell>[__tag]/loadtest.json    -> attempt-to-load cells (status + error excerpt)
  results/cotenancy.json                 -> serving tok/s before/during, delta %, training tok/s
  results/sleep_wake.json                -> level-1 sleep/wake timings and memory freed

Dagger (†) = pessimistic_tp: PESSIMISTIC_TP=1 in decisions.env AND the cell is TP>1
(TP2/TP4, DP-over-TP).  TP1 replica cells never get one.  Older JSONs without the
field are re-derived from launch.json's tp + decisions.env.
Python 3 stdlib only.
"""
import argparse
import datetime as dt
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hwdecisions import hw_decisions, pessimistic_flags, to_flag  # noqa: E402


def load(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def fmt(v, nd=3, pct=False):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "NO"
    if isinstance(v, (int, float)):
        return f"{v * 100:.1f}%" if pct else (f"{v:.{nd}f}" if isinstance(v, float) else f"{v:,}")
    return str(v)


def md_escape(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def layout_of(*docs):
    """'TP2 x DP1 x 1' from the first doc that knows tp/dp/replicas (gate JSONs carry them, launch.json too)."""
    for d in docs:
        if not isinstance(d, dict):
            continue
        for src in (d, d.get("launch") or {}, d.get("launch_a") or {}, d.get("launch_json") or {}):
            if isinstance(src, dict) and src.get("tp") not in (None, ""):
                return f"TP{src.get('tp')} x DP{src.get('dp') or 1} x {src.get('replicas') or 1}", src.get("tp")
    return "-", None


def dagger_for(dec, *docs):
    """True/False/None from the JSONs' own pessimistic_tp, else re-derived from tp + decisions.env."""
    for d in docs:
        if isinstance(d, dict) and "pessimistic_tp" in d and d["pessimistic_tp"] is not None and "acs_suspected" in d:
            # launch.json / loadtest.json carry the flag as int 0/1 (launch.sh num()), gate JSONs as bool:
            # normalise, otherwise `1 is True` is False and the dagger is silently dropped.
            return to_flag(d["pessimistic_tp"])
    _, tp = layout_of(*docs)
    return pessimistic_flags(dec, tp)["pessimistic_tp"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--results-dir", default=os.environ.get("RESULTS_ROOT", os.path.join(os.path.dirname(here), "results")))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = os.path.abspath(a.results_dir)
    out = a.out or os.path.join(res, "gates_summary.md")
    dec = hw_decisions(results_root=res)
    base_flags = pessimistic_flags(dec)

    md = [f"# Correctness gates and co-tenancy - {dt.datetime.now().isoformat(timespec='seconds')}", ""]
    if dec.get("_present"):
        md += [f"Hardware decisions (`{dec.get('_source')}`): ACS_SUSPECTED={dec.get('ACS_SUSPECTED', '?')}, "
               f"P2P_OK={dec.get('P2P_OK', '?')}, CUSTOM_ALLREDUCE={dec.get('CUSTOM_ALLREDUCE', '?')}, "
               f"NCCL_P2P_DISABLE={dec.get('NCCL_P2P_DISABLE', '?')}, PESSIMISTIC_TP={dec.get('PESSIMISTIC_TP', '?')}, "
               f"HOST_RAM_GB={dec.get('HOST_RAM_GB', '?')}." + (f" Notes: {md_escape(dec['NOTES'])}" if dec.get("NOTES") else ""), ""]
    else:
        md += ["_results/hw/decisions.env not found (vast/hardware_truth.sh writes it); TP>1 rows cannot be flagged._", ""]

    cells = sorted(p for p in glob.glob(os.path.join(res, "*")) if os.path.isdir(p))
    rows = []
    for cd in cells:
        name = os.path.basename(cd)
        g, k, lt = load(os.path.join(cd, "gsm8k.json")), load(os.path.join(cd, "kv_diff.json")), load(os.path.join(cd, "loadtest.json"))
        if not (g or k or lt):
            continue
        launch = load(os.path.join(cd, "launch.json"))
        layout, _ = layout_of(g, k, launch, lt)
        pess = dagger_for(dec, g, k, launch, lt)
        dagger = " †" if pess is True else (" ?" if pess is None and layout not in ("-",) and not layout.startswith("TP1 ") else "")
        if k and k.get("status") == "launch_failed":
            kv_sides = f"{k.get('kv_a')} vs {k.get('kv_b')}: launch failed ({k.get('kv_dtype_failed')})"
        elif k and k.get("status") == "not_applicable":
            # gates/run_kv_diff.sh: e.g. DeepSeek-V4 on sm_120 has no non-fp8 KV side (fp8 layouts only)
            kv_sides = f"n/a by design: {md_escape((k.get('reason') or '')[:90])}"
        elif k and k.get("side_a"):
            kv_sides = f"{k['side_a'].get('label')} vs {k['side_b'].get('label')}"
        else:
            kv_sides = "-"
        rows.append("| " + " | ".join([
            f"`{name}`{dagger}", layout,
            fmt((g or {}).get("exact_match_strict"), pct=True), fmt((g or {}).get("exact_match_flexible"), pct=True),
            (fmt(g.get("pass")) if g and g.get("pass") is not None else ("run" if g and g.get("status") == "ok" else (g.get("status") if g else "-"))),
            kv_sides,
            fmt((k or {}).get("pass")) if k else "-", fmt((k or {}).get("exact_match_rate"), pct=True),
            fmt((k or {}).get("mean_norm_edit_distance"), 4),
            (f"{k['side_a'].get('corrupt_count', 0)}/{k['side_b'].get('corrupt_count', 0)}" if k and k.get("side_a") else "-"),
            (f"{lt.get('status')} ({md_escape((lt.get('error_excerpt') or '')[:80])})" if lt else "-"),
        ]) + " |")
    md += ["## Per-cell gates", "",
           "| cell | layout | GSM8K strict | GSM8K flexible | GSM8K pass (vs REF) | kv_diff sides | kv_diff pass | exact match | mean NED | corrupt A/B | load test |",
           "|---|---|---|---|---|---|---|---|---|---|---|"] + (rows or ["| _no gate results yet_ | | | | | | | | | | |"]) + [""]

    c = load(os.path.join(res, "cotenancy.json"))
    md += ["## Training co-tenancy", ""]
    if c:
        s, d = c.get("serving", {}), c.get("serving_delta_during_vs_before_pct") or {}
        b, du = s.get("before") or {}, s.get("during") or {}
        tr = c.get("training") or {}
        pess = to_flag(c.get("pessimistic_tp"))
        if pess is None:
            pess = pessimistic_flags(dec, c.get("serving_tp"))["pessimistic_tp"]
        md += [f"Serving cell `{c.get('serving_cell')}`{' †' if pess is True else ''} ({c.get('serving_model')}, TP{c.get('serving_tp')}, GPUs {c.get('serving_gpus')}), "
               f"shape **{c.get('shape')}** at C={c.get('concurrency')}; training on GPU {c.get('training_gpu')} "
               f"({tr.get('model')}, LoRA r={((tr.get('lora') or {}).get('r'))}, {tr.get('batch')}x{tr.get('seq_len')}).", "",
               "| phase | req/s | output tok/s | total tok/s | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | TPOT p99 ms |", "|---|---|---|---|---|---|---|---|"]
        for tag, r in (("before", b), ("during", du), ("after", s.get("after") or {})):
            if r:
                md.append(f"| {tag} | {fmt(r.get('request_throughput'), 2)} | {fmt(r.get('output_throughput'), 0)} | {fmt(r.get('total_token_throughput'), 0)} | "
                          f"{fmt(r.get('median_ttft_ms') or r.get('p50_ttft_ms'), 1)} | {fmt(r.get('p99_ttft_ms'), 1)} | {fmt(r.get('median_tpot_ms') or r.get('p50_tpot_ms'), 2)} | {fmt(r.get('p99_tpot_ms'), 2)} |")
        md += ["", f"- output tok/s during vs before: **{fmt(d.get('output_throughput'), 2)} %**; p99 TTFT {fmt(d.get('p99_ttft_ms'), 1)} %, p99 TPOT {fmt(d.get('p99_tpot_ms'), 1)} %",
               f"- training: {fmt(c.get('training_tok_s'), 0)} tok/s overall, {fmt(c.get('training_tok_s_steady'), 0)} tok/s steady-state, "
               f"{tr.get('steps')} steps, peak {fmt(tr.get('peak_mem_alloc_gb'), 1)} GB, status {tr.get('status')}",
               f"- training overlapped {fmt(c.get('training_overlap_fraction_of_during_window'), pct=True)} of the DURING window", ""]
    else:
        md += ["_results/cotenancy.json not found_", ""]

    sw = load(os.path.join(res, "sleep_wake.json"))
    md += ["## Sleep / wake (vLLM sleep mode)", ""]
    if sw and sw.get("summary_level1"):
        s1 = sw["summary_level1"]
        md += [f"Cell `{sw.get('cell')}` ({sw.get('model')}), {s1['n']} level-1 cycles: sleep {fmt(s1.get('sleep_call_s_mean'), 2)} s, "
               f"wake {fmt(s1.get('wake_call_s_mean'), 2)} s, first request after wake {fmt(s1.get('first_request_after_wake_s_mean'), 2)} s "
               f"(awake probe {fmt(s1.get('probe_before_s_mean'), 2)} s); memory awake {s1.get('mem_awake_before_gb_per_gpu')} GB -> asleep {s1.get('mem_asleep_gb_per_gpu')} GB per GPU."
               + (f" Level-2: {json.dumps(sw.get('summary_level2'))}" if sw.get("summary_level2") else ""), ""]
    elif sw:
        md += [f"_status: {sw.get('status')} - {sw.get('fix') or sw.get('error')}_", ""]
    else:
        md += ["_results/sleep_wake.json not found_", ""]

    md += ["---",
           "† pessimistic_tp: PCIe ACS is suspected on this host (switch-local P2P redirected through the root complex; "
           "same-switch pair all_reduce ~21 GB/s vs cross-switch ~38 GB/s, 4-GPU ring ~19 GB/s). TP2/TP4 and DP-over-TP "
           "numbers are a pessimistic lower bound; TP1 replica cells are unaffected. P2P itself is ON "
           f"(P2P_OK={dec.get('P2P_OK', '?')}, NCCL_P2P_DISABLE={dec.get('NCCL_P2P_DISABLE', '?')}); custom all-reduce "
           f"{'ON' if base_flags['custom_allreduce'] else 'off'} by default. '?' = TP>1 cell but no decisions.env."]
    text = "\n".join(md) + "\n"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
