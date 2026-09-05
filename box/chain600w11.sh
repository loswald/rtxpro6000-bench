#!/usr/bin/env bash
# Original 600 W box to 02:00 UTC 6 Sept (replaces chain600w10): after the GLM quality run on the fastest layout,
# the remaining GLM throughput levers on that layout (glm_perf3.sh), then GLM with 65k tokens of output room on
# the best of them, then GLM at native FP8 if time, then the control arms. Prints CHAIN600W7 DONE.
R=/workspace/results; B=/workspace/bench; P=$R/probe
DEADLINE=$(( $(date -d "2026-09-06 02:00:00" +%s) ))
left(){ echo $(( (DEADLINE - $(date +%s)) / 60 )); }
step(){ echo "[$(date +%H:%M:%S)] 600W-11 (${1:-}min left): ${*:2}"; }
J=$R/eval/glm53f_best.json
for i in $(seq 1 720); do
  python3 -c "import json,sys; d=json.load(open('$J')); sys.exit(0 if d.get('partial') is False else 1)" 2>/dev/null && break
  grep -q "no time for the quality run on the fastest GLM layout\|CHAIN600W7 DONE" $R/chain600w.log 2>/dev/null && break
  sleep 20
done
for s in q600k q600m; do tmux kill-session -t =$s 2>/dev/null && step "$(left)" "$s stopped; chain600w11 runs the extended window"; done
sleep 3; source $B/hardkill.sh; kill_all >/dev/null 2>&1
python3 -c "import json; d=json.load(open('$J')); a=d['aggregate']; print('  GLM best-layout quality:', a.get('n_scored'), 'items, acc', a.get('acc_micro'), 'trunc', a.get('trunc_rate'))" 2>/dev/null
step "$(left)" "GLM-5.3-Flash on DP2 x TP2 + EP: CUDA graphs, 512 and 768 sequences (16-bit SSM state), 32k prefill chunk, FP8 KV"
bash $B/glm_perf3.sh > $R/glm_perf3.log 2>&1
best=$(python3 $B/pick_best.py "$P/glm53f_dp2*" router 1024 2>$R/glm_best3.tps)
flags="--tensor-parallel-size 2 --data-parallel-size 2 --enable-expert-parallel --max-num-seqs 384"
case "$best" in
  glm53f_dp2_cg)    flags="$flags --compilation-config {\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"}";;
  glm53f_dp2_s512)  flags="${flags/384/512}";;
  glm53f_dp2_ssm16) flags="${flags/384/768} --mamba-ssm-cache-dtype bfloat16";;
  glm53f_dp2_mb32k) flags="$flags --max-num-batched-tokens 32768";;
  glm53f_dp2_kvfp8) flags="$flags --kv-cache-dtype fp8";;
esac
step "$(left)" "fastest GLM arm now: ${best:-dp2tp2ep2_s384} ($(cat $R/glm_best3.tps 2>/dev/null) out tok/s) -> $flags"
L=$(left)
if [ "$L" -gt 90 ]; then
  b=$(( (L - 150) * 60 )); [ "$b" -lt 3600 ] && b=3600; [ "$b" -gt 5400 ] && b=5400
  step "$L" "GLM on that layout with 65k tokens of output room, budget ${b}s"
  BEST_FLAGS="$flags" BEST_BUDGET=$b LONG_CONC=64 ARMS=bestlong bash $B/glm_eval.sh > $R/glm_eval_bestlong.log 2>&1
fi
L=$(left)
if [ "$L" -gt 110 ]; then
  step "$L" "GLM-5.3-Flash at native FP8, 330 GB, TP4, 32 sequences: quality (budget $(( (L - 45) * 60 ))s)"
  FP8_BUDGET=$(( (L - 45) * 60 )) ARMS=fp8 bash $B/glm_eval.sh > $R/glm_eval_fp8.log 2>&1
fi
L=$(left)
if [ "$L" -gt 40 ]; then
  step "$L" "noise floor (Qwen NVFP4 a second time) and the gittensor weights under the official chat template"
  MODE=eval EVAL_BUDGET=$(( (L - 10) * 60 / 2 )) bash $B/ksweep.sh $B/lists/control600w.txt > $R/keval_control.log 2>&1
fi
step "$(left)" "CHAIN600W7 DONE"
kill_all
