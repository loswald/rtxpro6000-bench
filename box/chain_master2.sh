#!/usr/bin/env bash
# Replaces chain_master.sh, which deleted the native BF16 parent to make room for GLM's FP8 release BEFORE
# the logit ladder that measures every rung against that parent could run. GLM FP8 moves to the 600 W box
# (where its throughput counts); the ladder runs here right after the quantisation ladder's quality pass.
R=/workspace/results; B=/workspace/bench
step(){ echo "[$(date +%H:%M:%S)] MASTER2: $*"; }
step "quantisation ladder: throughput (resumed; arms already measured are skipped)"
bash $B/ksweep.sh $B/lists/quant6000.txt >> $R/ksweep_quant.log 2>&1
step "quantisation ladder: quality, an hour per request so no item is lost to the timeout"
MODE=eval EVAL_BUDGET=5400 bash $B/ksweep.sh $B/lists/quant6000.txt > $R/keval_quant.log 2>&1
step "logit divergence from the native BF16 parent, and speculation with a drafter that loads"
POS=16 bash $B/kldiff.sh >> $R/kldiff.log 2>&1
step "the tensor-parallel arms the fleet lost, causes fixed"
bash $B/ksweep.sh $B/lists/tp6000.txt > $R/ksweep_tp.log 2>&1
MODE=eval EVAL_BUDGET=3600 bash $B/ksweep.sh $B/lists/tp6000.txt > $R/keval_tp.log 2>&1
step "thinking on vs off, gemma-4 BF16"
MODE=eval EVAL_BUDGET=3600 bash $B/ksweep.sh $B/lists/thinkmode6000.txt > $R/keval_think.log 2>&1
step "Motif-3 on its vendor fork"
MD=/workspace/models/Motif-3-NVFP4 bash $B/motif_vllm.sh > $R/motif2.log 2>&1
step "MASTER DONE"
