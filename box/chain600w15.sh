#!/usr/bin/env bash
# Original 600 W box to 02:00 UTC 6 Sept (replaces chain600w14). GLM with 65k tokens of room on the NVFP4 build gained
# nothing (0.777 vs 0.792; 7.9% still truncated at 65k) while Z.AI's endpoint at 65k scored 0.860 on the same items
# (paired 33 to 7): the four-bit build is what holds GLM back here. So GLM at its native FP8 (330 GB, TP4) comes first -
# the one run that says whether this node can host GLM at the maker's quality - then the TP2 diagnosis probes.
R=/workspace/results; B=/workspace/bench; P=$R/probe
DEADLINE=$(( $(date -d "2026-09-06 02:00:00" +%s) ))
left(){ echo $(( (DEADLINE - $(date +%s)) / 60 )); }
step(){ echo "[$(date +%H:%M:%S)] 600W-15 (${1:-}min left): ${*:2}"; }
for s in q600s q600r; do tmux kill-session -t =$s 2>/dev/null && step "$(left)" "$s stopped; chain600w15 puts native FP8 first"; done
sleep 3; source $B/hardkill.sh; kill_all >/dev/null 2>&1
for i in $(seq 1 40); do ss -lnt 2>/dev/null | grep -q ":8000 " || break; sleep 3; done
L=$(left)
if [ "$L" -gt 90 ]; then
  b=$(( (L - 60) * 60 )); [ "$b" -gt 7200 ] && b=7200
  step "$L" "GLM-5.3-Flash at native FP8, 330 GB, TP4, 32 sequences: quality (budget ${b}s)"
  FP8_BUDGET=$b ARMS=fp8 bash $B/glm_eval.sh > $R/glm_eval_fp8.log 2>&1
  acc=$(python3 -c "import json; d=json.load(open('$R/eval/glm53f_fp8.json')); a=d['aggregate']; print(a.get('n_scored'), round(a['acc_micro'],3), 'partial', d.get('partial'))" 2>/dev/null)
  step "$(left)" "native FP8 quality: ${acc:-none}"
fi
L=$(left)
if [ "$L" -gt 25 ]; then
  step "$L" "GLM at TP2: eager, Marlin MoE, both, and TP2 without EP - probe each (throughput shapes only if time)"
  bash $B/glm_perf5.sh > $R/glm_perf5.log 2>&1
fi
step "$(left)" "CHAIN600W7 DONE"
kill_all
