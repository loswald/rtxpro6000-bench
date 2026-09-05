#!/usr/bin/env bash
# Third box, after the DeepSeek quality run: Qwen3.8-Flash-Next (index 46) takes the slot chain_c3 gave MiniMax.
# Waits for the DeepSeek eval JSON to be complete, stops chain_c3, downloads the NVFP4 checkpoint (this box pulls
# ~50 GB in two minutes), runs two layouts at two shapes, a 403-item quality run on the faster, then MiniMax if
# time allows. Prints CHAINC3 DONE so the finish-watcher, the hourly check and the destroy step need no change.
R=/workspace/results; B=/workspace/bench; MD=/workspace/models; P=$R/probe
DEADLINE=$(( $(date -d "2026-09-05 22:00:00" +%s) ))
left(){ echo $(( (DEADLINE - $(date +%s)) / 60 )); }
step(){ echo "[$(date +%H:%M:%S)] CHAINC4 (${1:-}min left): ${*:2}"; }
J=$R/eval/ds4flash_dp4ep4_s512_b12x--.json
for i in $(seq 1 240); do
  python3 -c "import json,sys; d=json.load(open('$J')); sys.exit(0 if d.get('partial') is False else 1)" 2>/dev/null && break
  grep -q "MiniMax-M3" $R/chain_c.log 2>/dev/null && break
  sleep 15
done
tmux kill-session -t =chainc3 2>/dev/null && step "$(left)" "chain_c3 stopped after the DeepSeek quality run; continuing as chain_c4"
sleep 3; source $B/hardkill.sh; kill_all >/dev/null 2>&1
python3 -c "import json; d=json.load(open('$J')); a=d['aggregate']; print('  DeepSeek DP4+EP quality:', a.get('n_scored'), 'items, acc', a.get('acc_micro'), 'partial', d.get('partial'))" 2>/dev/null

step "$(left)" "Qwen3.8-Flash-Next: downloading the NVFP4 checkpoint"
source $B/dlget.sh
L=$R/dl_c.log get RadixArk/Qwen3.8-Flash-Next-NVFP4 Qwen3.8-Flash-Next-NVFP4
echo "[$(date +%H:%M:%S)] QFNDL DONE" >> $R/dl_c.log
ls $MD/Qwen3.8-Flash-Next-NVFP4/*.safetensors >/dev/null 2>&1 || { step "$(left)" "Qwen3.8-Flash-Next download FAILED"; }

step "$(left)" "Qwen3.8-Flash-Next throughput: DP4+EP, then 2 x TP2, two shapes each"
SHAPES=fast bash $B/ksweep.sh $B/lists/qwen38fn_c2.txt > $R/ksweep_qfn.log 2>&1
best=$(python3 $B/pick_best.py "$P/qwen38fn_*" router 1024 2>$R/qfn_best.tps)
L=$(left)
if [ -n "$best" ] && [ "$L" -gt 35 ]; then
  line=$(grep -E "^${best%%_-*}\|" $B/lists/qwen38fn_c2.txt | head -1)
  [ -z "$line" ] && line=$(grep -vE "^#|^$" $B/lists/qwen38fn_c2.txt | head -1)
  step "$L" "Qwen3.8-Flash-Next quality on the faster layout: $best ($(cat $R/qfn_best.tps 2>/dev/null) out tok/s at router C1024), budget $(( (L - 15) * 60 ))s"
  printf '%s\n' "$line" > $B/lists/qfn_best.txt
  MODE=eval FIRST_ONLY=1 EVAL_CONC=64 EVAL_BUDGET=$(( (L - 15) * 60 )) bash $B/ksweep.sh $B/lists/qfn_best.txt > $R/keval_qfn.log 2>&1
else
  step "$L" "Qwen3.8-Flash-Next: no layout served, or no time for its quality run (best: ${best:-none})"
fi
L=$(left)
if [ "$L" -gt 45 ]; then
  step "$L" "MiniMax-M3 quality (budget $(( (L - 20) * 60 ))s), then two throughput shapes"
  MODE=eval FIRST_ONLY=1 EVAL_BUDGET=$(( (L - 20) * 60 )) bash $B/ksweep.sh $B/lists/minimax_c.txt > $R/keval_minimax.log 2>&1
  SHAPES=fast bash $B/ksweep.sh $B/lists/minimax_c.txt > $R/ksweep_minimax.log 2>&1
else
  step "$L" "MiniMax-M3 not run (time)"
fi
step "$(left)" "CHAINC3 DONE"
kill_all
