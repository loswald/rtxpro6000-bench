#!/usr/bin/env bash
# Third box: the rest of the five-hour window, re-cut at 17:10 UTC. Waits for chain_c2's logit ladder to finish
# (the ladder anchors and kldiff are in flight), then replaces it: DeepSeek-V4-Flash's throughput ceiling - the
# DP4 + expert-parallel layout that was 54% faster in the first campaign, and 512-sequence budgets - then
# MiniMax-M3 quality and throughput. The soak is dropped for time.
R=/workspace/results; B=/workspace/bench; MD=/workspace/models
DEADLINE=$(( $(date -d "2026-09-05 21:20:00" +%s) ))
left(){ echo $(( (DEADLINE - $(date +%s)) / 60 )); }
step(){ echo "[$(date +%H:%M:%S)] CHAINC3 (${1:-}min left): ${*:2}"; }
source $B/dlget.sh
tmux kill-session -t =dl 2>/dev/null
tmux new-session -d -s dl "bash -c 'source $B/dlget.sh; L=$R/dl_c.log get deepseek-ai/DeepSeek-V4-Flash-0731 DeepSeek-V4-Flash; echo \"[\$(date +%H:%M:%S)] DSDL DONE\" >> $R/dl_c.log'"
for i in $(seq 1 600); do grep -q "KLDIFF-BF16 DONE" $R/kldiff.log 2>/dev/null && break; sleep 15; done
tmux kill-session -t =chainc2 2>/dev/null && step "$(left)" "chain_c2 stopped after its logit ladder; continuing as chain_c3"
sleep 3; source $B/hardkill.sh; kill_all >/dev/null 2>&1

step "$(left)" "DeepSeek-V4-Flash: waiting for the download, applying the sm_120 o_proj fallback"
for i in $(seq 1 60); do grep -q "DSDL DONE" $R/dl_c.log 2>/dev/null && break; sleep 30; done
python3 $B/patch_oproj.py 2>&1 | tail -2 | sed 's/^/  /'
step "$(left)" "DeepSeek-V4-Flash throughput ceiling: DP4 + EP at 256 and 512 sequences, TP4 at 512"
EXTRA_ENV="VLLM_DSV4_OPROJ_SM120_FALLBACK=1" bash $B/ksweep.sh $B/lists/ds_perf.txt > $R/ksweep_dsperf.log 2>&1

step "$(left)" "MiniMax-M3: waiting for its download"
for i in $(seq 1 20); do grep -q "MMDL DONE" $R/dl_c.log 2>/dev/null && break; sleep 30; done
L=$(left); BUD=$(( (L - 60) * 60 )); [ "$BUD" -lt 1800 ] && BUD=1800
step "$L" "MiniMax-M3 quality (budget ${BUD}s), then throughput"
MODE=eval FIRST_ONLY=1 EVAL_BUDGET=$BUD bash $B/ksweep.sh $B/lists/minimax_c.txt > $R/keval_minimax.log 2>&1
bash $B/ksweep.sh $B/lists/minimax_c.txt > $R/ksweep_minimax.log 2>&1
step "$(left)" "CHAINC3 DONE"
kill_all
