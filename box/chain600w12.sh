#!/usr/bin/env bash
# Original 600 W box to 02:00 UTC 6 Sept (replaces chain600w11). The DP2 x TP2 + EP layout scored 0.643 on 403 items
# against 0.794 at TP4 (truncation 16% vs 8%, 2.7% degenerate): the fast layout is not quality-safe on this build.
# Isolate the cause - TP4 + EP quality (expert parallelism alone), two TP2 replicas without EP (data parallelism
# alone, throughput then quality) - then run the throughput levers and the 65k-room quality run on the fastest
# layout that holds quality. Prints CHAIN600W7 DONE.
R=/workspace/results; B=/workspace/bench; P=$R/probe
DEADLINE=$(( $(date -d "2026-09-06 02:00:00" +%s) ))
left(){ echo $(( (DEADLINE - $(date +%s)) / 60 )); }
step(){ echo "[$(date +%H:%M:%S)] 600W-12 (${1:-}min left): ${*:2}"; }
acc(){ python3 -c "import json; d=json.load(open('$R/eval/$1.json')); print(d['aggregate']['acc_micro'])" 2>/dev/null || echo 0; }
tps(){ python3 $B/pick_best.py "$P/$1" router 1024 2>&1 >/dev/null | tr -d '\n'; }
for s in q600n q600m q600k; do tmux kill-session -t =$s 2>/dev/null && step "$(left)" "$s stopped; chain600w12 isolates the GLM layout regression"; done
sleep 3; source $B/hardkill.sh; kill_all >/dev/null 2>&1

step "$(left)" "two TP2 replicas without expert parallelism: throughput"
bash $B/glm_perf4.sh > $R/glm_perf4.log 2>&1
step "$(left)" "TP4 + EP at 512 sequences: 403-item quality (isolates expert parallelism)"
ISO_BUDGET=3300 ARMS=tp4ep bash $B/glm_eval.sh > $R/glm_eval_tp4ep.log 2>&1
step "$(left)" "two TP2 replicas without EP: 403-item quality (isolates data parallelism)"
ISO_BUDGET=3300 ARMS=dp2 bash $B/glm_eval.sh > $R/glm_eval_dp2.log 2>&1
a_ep=$(acc glm53f_tp4ep); a_dp=$(acc glm53f_dp2); t_dp=$(tps "glm53f_dp2noep_s384")
step "$(left)" "isolation: TP4+EP acc $a_ep (931 out/s) | DP2 no-EP acc $a_dp (${t_dp:-?} out/s) | TP4 base 0.794 (911)"
# the fastest layout that holds quality (within 0.03 of the TP4 base); TP4 at 512 sequences is the fallback
flags="--max-num-seqs 512 --max-num-batched-tokens 16384"; lay="TP4, 512 seqs"
python3 -c "import sys; sys.exit(0 if float('$a_ep' or 0) >= 0.764 else 1)" && { flags="--enable-expert-parallel --max-num-seqs 512 --max-num-batched-tokens 16384"; lay="TP4 + EP, 512 seqs"; }
python3 -c "import sys; sys.exit(0 if float('$a_dp' or 0) >= 0.764 and float('${t_dp:-0}' or 0) > 931 else 1)" && { flags="--tensor-parallel-size 2 --data-parallel-size 2 --max-num-seqs 384 --max-num-batched-tokens 16384"; lay="DP2 x TP2 without EP, 384 seqs"; }
echo "$flags" > $R/glm_safe.flags
step "$(left)" "quality-safe layout for the rest: $lay -> $flags"
L=$(left)
if [ "$L" -gt 100 ]; then
  step "$L" "throughput levers on that layout: CUDA graphs, 512/768 sequences with 16-bit SSM state, 32k prefill chunk, FP8 KV"
  SAFE_FLAGS="$flags" bash $B/glm_perf5.sh > $R/glm_perf5.log 2>&1
fi
L=$(left)
if [ "$L" -gt 75 ]; then
  b=$(( (L - 20) * 60 )); [ "$b" -gt 5400 ] && b=5400
  step "$L" "GLM on the quality-safe layout with 65k tokens of output room, budget ${b}s"
  BEST_FLAGS="$flags" BEST_BUDGET=$b LONG_CONC=64 ARMS=bestlong bash $B/glm_eval.sh > $R/glm_eval_bestlong.log 2>&1
fi
step "$(left)" "CHAIN600W7 DONE"
kill_all
