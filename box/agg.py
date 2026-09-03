#!/usr/bin/env python3
"""Aggregate the 4 per-port bench JSONs for one point and print a full latency+throughput line.
Also appends a row to <tag>/summary_full.tsv so nothing is lost.
Usage: agg.py <dir> <filename-prefix> <label> <total_concurrency> <tag>
"""
import glob, json, os, sys

d, pref, label, tot, tag = sys.argv[1:6]
files = sorted(glob.glob(os.path.join(d, pref + "*.json")))

req = out = inp = tot_tp = 0.0
dur = 0.0
comp = 0
ttft_w = tpot_w = itl_w = e2e_w = 0.0
p99_ttft = p99_tpot = p99_e2e = 0.0
p50_ttft = p50_tpot = 0.0

for f in files:
    try:
        j = json.load(open(f))
    except Exception:
        continue
    c = j.get("completed", 0) or 0
    du = j.get("duration") or 1
    comp += c
    dur = max(dur, du)
    req += j.get("request_throughput", 0) or 0
    out += j.get("output_throughput", 0) or 0
    tot_tp += j.get("total_token_throughput", 0) or 0
    inp += (j.get("total_input_tokens", 0) or 0) / du
    # completion-weighted means, max of p99s (worst port is what a user feels)
    ttft_w += (j.get("mean_ttft_ms", 0) or 0) * c
    tpot_w += (j.get("mean_tpot_ms", 0) or 0) * c
    itl_w += (j.get("mean_itl_ms", 0) or 0) * c
    e2e_w += (j.get("mean_e2el_ms", 0) or 0) * c
    p99_ttft = max(p99_ttft, j.get("p99_ttft_ms", 0) or 0)
    p99_tpot = max(p99_tpot, j.get("p99_tpot_ms", 0) or 0)
    p99_e2e = max(p99_e2e, j.get("p99_e2el_ms", 0) or 0)
    p50_ttft = max(p50_ttft, j.get("median_ttft_ms", 0) or j.get("p50_ttft_ms", 0) or 0)
    p50_tpot = max(p50_tpot, j.get("median_tpot_ms", 0) or j.get("p50_tpot_ms", 0) or 0)

if not comp:
    print(f"  {label:10s} C{tot:<5} FAILED ({len(files)} files)")
    sys.exit(0)

n = max(comp, 1)
ttft, tpot, itl, e2e = ttft_w / n, tpot_w / n, itl_w / n, e2e_w / n

print(
    f"  {label:10s} C{tot:<5} {dur:6.1f}s "
    f"{req*60:7.0f} req/min  {out:7.0f} out/s  {inp:8.0f} in/s  {tot_tp:8.0f} tot/s  "
    f"TTFT {ttft:6.0f}/{p99_ttft:7.0f} ms  TPOT {tpot:5.1f}/{p99_tpot:6.1f} ms  "
    f"ITL {itl:5.1f}  E2E {e2e/1000:5.1f}/{p99_e2e/1000:5.1f} s"
)

tsv = os.path.join(d, "summary_full.tsv")
new = not os.path.exists(tsv)
with open(tsv, "a") as fh:
    if new:
        fh.write("tag\tlabel\tC\tdur_s\treq_s\tout_tps\tin_tps\ttotal_tps\t"
                 "ttft_mean_ms\tttft_p50_ms\tttft_p99_ms\ttpot_mean_ms\ttpot_p50_ms\ttpot_p99_ms\t"
                 "itl_mean_ms\te2e_mean_ms\te2e_p99_ms\tcompleted\n")
    fh.write("\t".join(str(x) for x in [
        tag, label, tot, round(dur, 1), round(req, 3), round(out), round(inp), round(tot_tp),
        round(ttft), round(p50_ttft), round(p99_ttft),
        round(tpot, 2), round(p50_tpot, 2), round(p99_tpot, 2),
        round(itl, 2), round(e2e), round(p99_e2e), comp]) + "\n")
