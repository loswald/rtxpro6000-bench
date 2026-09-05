#!/usr/bin/env bash
# Original 600 W box, extended window (Nish, 5 Sept 20:25 UTC: four more hours). Takes over from chain600w8 once
# the 403-item quality run on the fastest GLM layout is complete, then: GLM on that layout with 65k tokens of
# output room (what the 32k cap cost the 29 items that hit it), GLM at its native FP8 (330 GB - is NVFP4 losing
# anything?), then the noise-floor/template arms and RedHat under W4A4. Prints CHAIN600W7 DONE at the end so the
# finish-watcher needs no change. Deadline 02:00 UTC 6 Sept.
R=/workspace/results; B=/workspace/bench; P=$R/probe
DEADLINE=$(( $(date -d "2026-09-06 02:00:00" +%s) ))
left(){ echo $(( (DEADLINE - $(date +%s)) / 60 )); }
step(){ echo "[$(date +%H:%M:%S)] 600W-10 (${1:-}min left): ${*:2}"; }
J=$R/eval/glm53f_best.json
for i in $(seq 1 720); do
  python3 -c "import json,sys; d=json.load(open('$J')); sys.exit(0 if d.get('partial') is False else 1)" 2>/dev/null && break
  grep -q "no time for the quality run on the fastest GLM layout\|CHAIN600W7 DONE" $R/chain600w.log 2>/dev/null && break
  sleep 20
done
tmux kill-session -t =q600k 2>/dev/null && step "$(left)" "chain600w8 stopped after the GLM quality run; continuing as chain600w10 to 02:00 UTC"
sleep 3; source $B/hardkill.sh; kill_all >/dev/null 2>&1
python3 -c "import json; d=json.load(open('$J')); a=d['aggregate']; print('  GLM best-layout quality:', a.get('n_scored'), 'items, acc', a.get('acc_micro'), 'trunc', a.get('trunc_rate'))" 2>/dev/null
flags=$(cat $R/glm_best.flags 2>/dev/null); [ -z "$flags" ] && flags="--tensor-parallel-size 2 --data-parallel-size 2 --enable-expert-parallel --max-num-seqs 384"
L=$(left)
if [ "$L" -gt 100 ]; then
  step "$L" "GLM on the fastest layout with 65k tokens of output room ($flags), budget $(( (L - 190) > 60 ? (L - 190) * 60 : 4800 ))s"
  BEST_FLAGS="$flags --max-num-batched-tokens 16384" BEST_BUDGET=$(( (L - 190) > 60 ? (L - 190) * 60 : 4800 )) LONG_CONC=64 ARMS=bestlong bash $B/glm_eval.sh > $R/glm_eval_bestlong.log 2>&1
fi
L=$(left)
if [ "$L" -gt 120 ]; then
  step "$L" "GLM-5.3-Flash at native FP8, 330 GB, TP4, 32 sequences: quality (budget $(( (L - 60) * 60 ))s)"
  FP8_BUDGET=$(( (L - 60) * 60 )) ARMS=fp8 bash $B/glm_eval.sh > $R/glm_eval_fp8.log 2>&1
fi
L=$(left)
if [ "$L" -gt 40 ]; then
  step "$L" "noise floor (Qwen NVFP4 a second time) and the gittensor weights under the official chat template"
  MODE=eval EVAL_BUDGET=$(( (L - 10) * 60 / 2 )) bash $B/ksweep.sh $B/lists/control600w.txt > $R/keval_control.log 2>&1
fi
L=$(left)
if [ "$L" -gt 40 ]; then
  step "$L" "RedHat NVFP4 under the W4A4 kernel: quality, then two throughput shapes"
  MODE=eval EVAL_BUDGET=$(( (L - 20) * 60 )) bash $B/ksweep.sh $B/lists/fix600w_b12x.txt > $R/keval_fixb12x.log 2>&1
  SHAPES=fast bash $B/ksweep.sh $B/lists/fix600w_b12x.txt > $R/ksweep_fixb12x.log 2>&1
fi
step "$(left)" "CHAIN600W7 DONE"
kill_all
