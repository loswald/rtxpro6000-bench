#!/usr/bin/env bash
# 400 W box, after the master chain. Replaces chain_laguna.sh, which was armed on the same marker this one
# waits for - two waiters on one marker is the collision that cost Ornith its arms on 4 Sept - so it is
# stopped and its work runs here, first. Everything here is quality (power-independent) or a fix.
R=/workspace/results; B=/workspace/bench
step(){ echo "[$(date +%H:%M:%S)] POST2: $*"; }
for i in $(seq 1 2880); do grep -q "MASTER DONE" $R/chain6000.log 2>/dev/null && break; sleep 60; done
step "Laguna-S-2.1: thinking on (with room to finish) vs off"
MODE=eval EVAL_BUDGET=3600 bash $B/ksweep.sh $B/lists/laguna6000.txt > $R/keval_laguna.log 2>&1
step "Nemotron-3-Super: finish the 24 items the 600 s request timeout cut off"
MODE=eval EVAL_RESUME=1 EVAL_BUDGET=3600 bash $B/ksweep.sh $B/lists/resume6000.txt > $R/keval_resume6000.log 2>&1
step "Ling-3.0-flash quality; gemma-4-31B in both thinking modes"
MODE=eval EVAL_BUDGET=5400 bash $B/ksweep.sh $B/lists/tierb6000.txt > $R/keval_tierb.log 2>&1
step "POST2 DONE"
