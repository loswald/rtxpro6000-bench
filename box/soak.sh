#!/usr/bin/env bash
# Soak: is the configuration this node would actually run stable for hours, not for a forty-minute sweep?
#
# Every throughput number in this repository comes from a few minutes of steady load. A node bought for two
# years serves for days at a time, and the failures that matter there are the ones a sweep never sees: a
# replica that dies on the fourth hour, a preemption count that climbs as fragmentation sets in, thermal
# throttling under sustained 600 W, latency tails that drift. This runs the same replicas for MINUTES minutes
# of back-to-back rounds on a mixed shape (a shared 512-token prefix, 1,024 in, 256 out - closer to an agent
# harness than any single shape) and records, per round, output tok/s and TTFT p99, and every 30 s the
# engine's preemption counter, running requests, KV usage, and the card's power, temperature and SM clock.
# The tripwire runs at the end as well as the start: a model that serves fast but has started to babble is a
# failure this catches and a throughput number does not.
#
# Reuses ksweep.sh's launcher and bench helper: with no list argument, sourcing it defines the functions and
# returns (its trailing kill_all finds nothing to kill).
set -u
source /workspace/bench/ksweep.sh
TAG=${TAG:-soak_q27_nvfp4}
DIR=${DIR:-/workspace/models/Qwen27B-NVFP4-RTX5090}
LIN=${LIN:-b12x}; MOE=${MOE:--}
MINUTES=${MINUTES:-120}
ROUND_C=${ROUND_C:-64}          # concurrency per replica
log "===== soak: $TAG, $MINUTES min, ${ROUND_C}/replica on 512-prefix + 1024/256 ====="
serve "$TAG" "$DIR" 1 "$LIN" "$MOE" || { log "soak: server never became healthy"; kill_all; exit 1; }
n=$NGPU
mkdir -p "$P/$TAG"
echo "ts,preemptions,running,kv_usage,power_w,temp_c,sm_mhz,unhealthy" > "$P/$TAG/metrics.csv"
(
  while :; do
    m=$(curl -fsS -m 3 http://127.0.0.1:8000/metrics 2>/dev/null)
    pre=$(printf '%s\n' "$m" | grep -m1 -E '^vllm:num_preemptions_total' | awk '{print $2}')
    run=$(printf '%s\n' "$m" | grep -m1 -E '^vllm:num_requests_running' | awk '{print $2}')
    kv=$(printf '%s\n' "$m" | grep -m1 -E '^vllm:(gpu_cache_usage_perc|kv_cache_usage_perc)' | awk '{print $2}')
    g=$(nvidia-smi --query-gpu=power.draw,temperature.gpu,clocks.sm --format=csv,noheader,nounits | head -1 | tr -d ' ')
    bad=0; for i in $(seq 0 $((n-1))); do curl -fsS -m 3 "http://127.0.0.1:$((8000+i))/health" >/dev/null 2>&1 || bad=$((bad+1)); done
    echo "$(date +%s),${pre:-},${run:-},${kv:-},$g,$bad" >> "$P/$TAG/metrics.csv"
    sleep 30
  done
) & SAMPLER=$!
t_end=$(( $(date +%s) + MINUTES * 60 )); r=0
while [ "$(date +%s)" -lt "$t_end" ]; do
  r=$((r+1))
  pt "$TAG" "round$(printf '%02d' "$r")" 1024 256 512 "$ROUND_C" "$n" "$DIR" 2>&1 | tail -1
  bad=0; for i in $(seq 0 $((n-1))); do curl -fsS -m 3 "http://127.0.0.1:$((8000+i))/health" >/dev/null 2>&1 || bad=$((bad+1)); done
  [ "$bad" = 0 ] || log "  after round $r: $bad replica(s) UNHEALTHY"
done
kill "$SAMPLER" 2>/dev/null
$CLEAN python3 "$B/quality20.py" m http://127.0.0.1:8000 "$P/${TAG}_quality20_end.json" --mode chat --max-tokens 2048 2>&1 | tail -1 | sed 's/^/  tripwire at the end: /'
python3 - "$P/$TAG" <<'PY'
import csv, sys, glob, os
d = sys.argv[1]; rows = []
for f in glob.glob(os.path.join(d, "summary*.tsv")):
    rows += list(csv.DictReader(open(f, encoding="utf-8"), delimiter="\t"))
rounds = sorted((r for r in rows if r.get("label", "").startswith("round")), key=lambda r: r["label"])
if not rounds:
    print("  no rounds summarised"); sys.exit()
tps = [float(r["out_tps"]) for r in rounds]; p99 = [float(r.get("ttft_p99_ms") or 0) for r in rounds]
print("  rounds=%d  out tok/s first=%.0f last=%.0f min=%.0f max=%.0f (spread %.1f%%)" % (
    len(tps), tps[0], tps[-1], min(tps), max(tps), 100 * (max(tps) - min(tps)) / max(tps)))
print("  TTFT p99 ms first=%.0f last=%.0f max=%.0f" % (p99[0], p99[-1], max(p99)))
PY
python3 - "$P/$TAG/metrics.csv" <<'PY'
import csv, sys
rows = [r for r in list(csv.reader(open(sys.argv[1])))[1:] if len(r) >= 8]
if rows:
    f = lambda i: [float(r[i] or 0) for r in rows]
    pre, pw, tmp, clk, bad = f(1), f(4), f(5), f(6), f(7)
    print("  samples=%d | preemptions first=%.0f last=%.0f | power W mean=%.0f max=%.0f | temp C max=%.0f | "
          "SM MHz min=%.0f mean=%.0f | unhealthy samples=%d" % (
              len(rows), pre[0], pre[-1], sum(pw) / len(pw), max(pw), max(tmp), min(clk), sum(clk) / len(clk),
              sum(1 for b in bad if b > 0)))
PY
log "SOAK DONE"
kill_all
