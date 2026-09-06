#!/usr/bin/env bash
# Original 600 W box to 02:00 UTC 6 Sept: after the DP4 + EP 65k-room run, the TP2 diagnosis (glm_perf5.sh) in place
# of the native-FP8 arm - the DGX Spark community serves this model at TP2 with --enforce-eager and Marlin MoE, so
# each of those is tried alone and together; an arm whose 20-item probe is clean gets a 403-item run. FP8 if time.
R=/workspace/results; B=/workspace/bench; P=$R/probe
DEADLINE=$(( $(date -d "2026-09-06 02:00:00" +%s) ))
left(){ echo $(( (DEADLINE - $(date +%s)) / 60 )); }
step(){ echo "[$(date +%H:%M:%S)] 600W-14 (${1:-}min left): ${*:2}"; }
J=$R/eval/glm53f_dp4long.json
for i in $(seq 1 900); do
  python3 -c "import json,sys; d=json.load(open('$J')); sys.exit(0 if d.get('partial') is False else 1)" 2>/dev/null && break
  grep -qE "600W-13.*(native FP8|noise floor|CHAIN600W7 DONE)" $R/chain600w.log 2>/dev/null && break
  sleep 15
done
tmux kill-session -t =q600r 2>/dev/null && step "$(left)" "chain600w13 stopped after the 65k-room run; chain600w14 diagnoses TP2"
sleep 3; source $B/hardkill.sh; kill_all >/dev/null 2>&1
for i in $(seq 1 40); do ss -lnt 2>/dev/null | grep -q ":8000 " || break; sleep 3; done
L=$(left)
if [ "$L" -gt 60 ]; then
  step "$L" "GLM at TP2: eager, Marlin MoE, both, and TP2 without EP - probe each"
  bash $B/glm_perf5.sh > $R/glm_perf5.log 2>&1
fi
# the fastest arm whose probe has no more than one degenerate item
best=""; besttps=0
for d in $P/glm53f_dp2_eager $P/glm53f_dp2_marlin $P/glm53f_dp2_eagermarlin $P/glm53f_tp2x2_noep; do
  t=$(basename $d); q=$P/${t}_quality20.json; [ -f "$q" ] || continue
  deg=$(python3 -c "import json,collections; d=json.load(open('$q')); it=d if isinstance(d,list) else d.get('items',[]); print(sum(1 for x in it if x.get('status')=='degenerate'))" 2>/dev/null || echo 99)
  tps=$(python3 $B/pick_best.py "$d" router 1024 2>/dev/null >/dev/null; python3 - <<PY 2>/dev/null
import json,glob
v=0
for f in glob.glob("$d/*__router__c1024__*.json"):
    try:
        j=json.load(open(f)); v=max(v, float(j.get("output_throughput",0)))
    except Exception: pass
print(int(v))
PY
)
  step "$(left)" "  $t: degenerate $deg/20, router C1024 ${tps:-0} out tok/s"
  if [ "${deg:-99}" -le 1 ] && [ "${tps:-0}" -gt "$besttps" ]; then best=$t; besttps=$tps; fi
done
L=$(left)
if [ -n "$best" ] && [ "$L" -gt 60 ]; then
  case "$best" in
    glm53f_dp2_eager)       flags="--tensor-parallel-size 2 --data-parallel-size 2 --enable-expert-parallel --max-num-seqs 384 --enforce-eager";;
    glm53f_dp2_marlin)      flags="--tensor-parallel-size 2 --data-parallel-size 2 --enable-expert-parallel --max-num-seqs 384 --moe-backend marlin";;
    glm53f_dp2_eagermarlin) flags="--tensor-parallel-size 2 --data-parallel-size 2 --enable-expert-parallel --max-num-seqs 384 --enforce-eager --moe-backend marlin";;
    glm53f_tp2x2_noep)      flags="--tensor-parallel-size 2 --data-parallel-size 2 --max-num-seqs 384";;
  esac
  b=$(( (L - 20) * 60 )); [ "$b" -gt 4200 ] && b=4200
  step "$L" "403-item quality on the clean TP2 arm $best ($besttps out tok/s): $flags, budget ${b}s"
  BEST_FLAGS="$flags --max-num-batched-tokens 16384" BEST_BUDGET=$b ARMS=best bash $B/glm_eval.sh > $R/glm_eval_tp2fix.log 2>&1
else
  step "$L" "no TP2 arm probed clean (best: ${best:-none})"
fi
L=$(left)
if [ "$L" -gt 110 ]; then
  step "$L" "GLM-5.3-Flash at native FP8, 330 GB, TP4, 32 sequences: quality (budget $(( (L - 45) * 60 ))s)"
  FP8_BUDGET=$(( (L - 45) * 60 )) ARMS=fp8 bash $B/glm_eval.sh > $R/glm_eval_fp8.log 2>&1
fi
step "$(left)" "CHAIN600W7 DONE"
kill_all
