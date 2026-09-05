#!/usr/bin/env bash
# Original 600 W box, after chain600w15 prints CHAIN600W7 DONE: GLM at native FP8 gets its throughput shapes (router
# and shared-prefix at the 32-sequence budget the 330 GB checkpoint leaves), so the FP8 point can sit on the frontier
# with its quality. Then CHAIN600W7 DONE again; hard stop 02:15 UTC.
R=/workspace/results; B=/workspace/bench; P=$R/probe; S=$R/smoke
HARD=$(( $(date -d "2026-09-06 02:15:00" +%s) ))
left(){ echo $(( (HARD - $(date +%s)) / 60 )); }
step(){ echo "[$(date +%H:%M:%S)] 600W-16 (${1:-}min to 02:15): ${*:2}"; }
n0=$(grep -c "CHAIN600W7 DONE" $R/chain600w.log 2>/dev/null || echo 0)
for i in $(seq 1 1000); do n=$(grep -c "CHAIN600W7 DONE" $R/chain600w.log 2>/dev/null || echo 0); [ "$n" -gt "$n0" ] && break; sleep 15; done
sleep 5; source $B/hardkill.sh; kill_all >/dev/null 2>&1
L=$(left)
if [ "$L" -gt 25 ]; then
  step "$L" "GLM-5.3-Flash at native FP8, TP4, 32 sequences: router and shared-prefix shapes"
  bash $B/glm_fp8_shapes.sh > $R/glm_fp8_shapes.log 2>&1
fi
step "$(left)" "CHAIN600W7 DONE"
kill_all
