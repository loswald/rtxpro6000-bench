#!/usr/bin/env bash
# The untested lever: MoE kernel A/B on gpt-oss-120b x4 replicas (our throughput champion).
# Each backend: launch, confirm which kernel the server ACTUALLY selected, quality-check, 3-point probe.
R=/workspace/results; P=$R/probe; S=$R/smoke; B=/workspace/bench; MD=/workspace/models
mkdir -p $P $S
log(){ echo "[$(date +%H:%M:%S)] $*"; }
kill_all(){ tmux kill-session -t =srv 2>/dev/null
  for pid in $(pgrep -f "vllm serv[e]"); do kill "$pid" 2>/dev/null; done; sleep 8
  for pid in $(pgrep -f "vllm serv[e]"); do kill -9 "$pid" 2>/dev/null; done; sleep 3; }

try_backend(){ # $1 = backend name (or "default"), $2 = extra args
  local be="$1"; shift
  local extra="$*"
  local tag="moe_${be}"
  local args=""
  [ "$be" != "default" ] && args="--moe-backend $be"
  kill_all
  log "=== $be === $args $extra"
  cat > $B/l_moe.sh <<L
#!/usr/bin/env bash
exec bash /workspace/bench/launch_x4.sh /workspace/models/gpt-oss-120b gptoss $args $extra
L
  chmod +x $B/l_moe.sh
  tmux new-session -d -s srv "bash $B/l_moe.sh > $S/$tag.log 2>&1; echo EXIT=\$? >> $S/$tag.log"
  local t=0 ok=0
  while [ $t -lt 900 ]; do
    ok=1; for p in 8000 8001 8002 8003; do curl -fsS -m 3 "http://127.0.0.1:$p/health" >/dev/null 2>&1 || ok=0; done
    [ "$ok" = 1 ] && break
    if grep -q "^EXIT=" "$S/$tag.log" 2>/dev/null; then
      log "$be REJECTED after ${t}s"
      grep -iE "invalid choice|does not support|not supported|Error|error" "$S/$tag.log" | grep -vE "import_utils|deep_ep" | head -2 | cut -c1-230
      return 1
    fi
    sleep 10; t=$((t+10))
  done
  [ "$ok" = 1 ] || { log "$be TIMED OUT"; return 1; }
  local sel
  sel=$(grep -m1 -oE "Using '[A-Z0-9_]+' Mxfp4 MoE backend" "$S/$tag.log" | head -1)
  log "$be healthy in ${t}s — server selected: ${sel:-unknown}"
  python3 $B/quality20.py gptoss http://127.0.0.1:8000 $P/${tag}_quality20.json 2>&1 | tail -1
  bash $B/probe4.sh "$tag" gptoss $MD/gpt-oss-120b auto tune > $P/$tag.log 2>&1
  echo "  -> $(tail -n +2 $P/$tag/summary.tsv 2>/dev/null | awk -F'\t' '{printf "%s C%s: %s out tok/s | ", $2,$6,$10}')"
  return 0
}

log "MoE backend A/B on gpt-oss-120b x4 (baseline: default = MARLIN, 6,860 out tok/s router C256 on host A)"
try_backend default
try_backend marlin
try_backend flashinfer_cutlass
try_backend b12x
try_backend flashinfer_b12x
try_backend triton
try_backend flashinfer_cutedsl
log "MOE-AB DONE"
kill_all
