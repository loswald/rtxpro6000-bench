#!/usr/bin/env bash
# Third box: after MiniMax-M3's quality run, a saturation probe for DeepSeek-V4-Flash's DP4 + EP layout at 2,048 streams
# (it has only ever been offered 1,024 - 256 per engine), then chain_c8's remaining steps: MiniMax's two shapes and
# DeepSeek with 65k tokens of output room. CHAINC3 DONE at the end hands over to chain_c10.
R=/workspace/results; B=/workspace/bench; P=$R/probe
DEADLINE=$(( $(date -d "2026-09-06 02:00:00" +%s) ))
left(){ echo $(( (DEADLINE - $(date +%s)) / 60 )); }
step(){ echo "[$(date +%H:%M:%S)] CHAINC11 (${1:-}min left): ${*:2}"; }
for i in $(seq 1 720); do
  grep -q "KSWEEP DONE" $R/keval_minimax.log 2>/dev/null && break
  sleep 15
done
tmux kill-session -t =chainc8 2>/dev/null && step "$(left)" "chain_c8 stopped after MiniMax's quality run; chain_c11 runs the rest"
sleep 3; source $B/hardkill.sh; kill_all >/dev/null 2>&1
for i in $(seq 1 40); do ss -lnt 2>/dev/null | grep -q ":8000 " || break; sleep 3; done
python3 - <<'PY' 2>/dev/null
import json,glob
for f in glob.glob("/workspace/results/eval/minimaxm3_*.json"):
    if f.endswith(".run.json"): continue
    d=json.load(open(f)); a=d["aggregate"]; print("  MiniMax quality:", f.split("/")[-1], a.get("n_scored"), "items, acc", a.get("acc_micro"), "partial", d.get("partial"))
PY
L=$(left)
if [ "$L" -gt 100 ]; then
  step "$L" "DeepSeek-V4-Flash DP4 + EP at 2,048 streams: router and shared-prefix shapes"
  EXTRA_ENV="VLLM_DSV4_OPROJ_SM120_FALLBACK=1" SHAPES=deep bash $B/ksweep.sh $B/lists/ds_deep.txt > $R/ksweep_dsdeep.log 2>&1
fi
L=$(left)
if [ "$L" -gt 80 ] && ls -d $P/minimaxm3_* >/dev/null 2>&1; then
  step "$L" "MiniMax-M3: two throughput shapes"
  SHAPES=fast bash $B/ksweep.sh $B/lists/minimax_c3.txt > $R/ksweep_minimax.log 2>&1
fi
L=$(left)
if [ "$L" -gt 40 ]; then
  step "$L" "DeepSeek-V4-Flash, fastest layout, 65k tokens of output room: quality (budget $(( (L - 20) * 60 ))s)"
  EXTRA_ENV="VLLM_DSV4_OPROJ_SM120_FALLBACK=1" MODE=eval FIRST_ONLY=1 EVAL_CONC=48 EVAL_MAXTOK=65536 \
    EVAL_CAPS="math=65536,code=40960,knowledge=40960,ifeval=32768,tools=16384,longctx=12288" \
    EVAL_BUDGET=$(( (L - 20) * 60 )) bash $B/ksweep.sh $B/lists/ds_long.txt > $R/keval_dslong.log 2>&1
fi
step "$(left)" "CHAINC3 DONE"
kill_all
