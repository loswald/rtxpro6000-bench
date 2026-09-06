#!/usr/bin/env bash
# One hour on a fresh 4x RTX PRO 6000 box for one question: does GLM-5.3-Flash's two-engine layout (DP2 x TP2 + EP,
# 1,300 out tok/s on routing traffic) serve clean output with the DGX Spark recipe (eager mode + Marlin MoE)?
# Provision (vendor image lift + NVFP4 weights in parallel), port the vendor tree to sm_120, then glm_perf5.sh
# probe-first with a hard cut-off. Marker: GLMHOUR DONE. All times UTC (the box clock is UTC).
set -u
B=/workspace/bench; R=/workspace/results; M=/workspace/models
STOP_AT="${STOP_AT:-2026-09-06 01:38:00}"          # no new arm after this; the box is destroyed ~01:50
log(){ echo "[$(date -u +%H:%M:%S)] GLMHOUR: $*"; }
mkdir -p "$B" "$R/smoke" "$R/probe" "$R/eval" "$M" /workspace/glmvllm
cd "$B"
log "start; stop-at $STOP_AT UTC; $(nvidia-smi --query-gpu=name,power.limit --format=csv,noheader | head -1)"

# 1. vendor image (8.6 GB, 32 layers) in the background
( python3 "$B/pull_image.py" "vllm/vllm-openai:glm53-flash-x86_64-cu130" /workspace/glmimg > "$R/glmimg.log" 2>&1; echo "IMG_EXIT=$?" >> "$R/glmimg.log" ) &
IMGPID=$!

# 2. weights (198 GB) in the foreground, hf_transfer on
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install -q hf_transfer 2>/dev/null || true
t0=$(date +%s)
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("RedHatAI/GLM-5.3-Flash-NVFP4", local_dir="/workspace/models/GLM-5.3-Flash-NVFP4", max_workers=16)
PY
log "weights: $(du -sh $M/GLM-5.3-Flash-NVFP4 2>/dev/null | cut -f1) in $(( $(date +%s) - t0 ))s"
mkdir -p "$M/glm53f_tok"; cp "$M"/GLM-5.3-Flash-NVFP4/tok* "$M"/GLM-5.3-Flash-NVFP4/*.jinja "$M/glm53f_tok/" 2>/dev/null
ls "$M/glm53f_tok" | paste -sd" " | sed 's/^/  tok: /'

# 3. wait for the image, port the vendor tree
wait "$IMGPID" 2>/dev/null; tail -n 2 "$R/glmimg.log" | sed 's/^/  /'
VEND=/workspace/glmimg/usr/local/lib/python3.12/dist-packages/vllm
[ -d "$VEND" ] || { log "no vendor tree at $VEND - stop"; echo "GLMHOUR DONE"; exit 1; }
[ -e /workspace/glmvllm/vllm ] || ln -s "$VEND" /workspace/glmvllm/vllm
python3 "$B/vllm_sm120_nope.py" "$VEND" 2>&1 | sed 's/^/  port: /'

# 4. the probes, probe-first, cut off by the clock
sed -i "s|^STOP_AT=.*|STOP_AT=\$(date -u -d \"$STOP_AT\" +%s)|" "$B/glm_perf5.sh"
grep -n "^STOP_AT=" "$B/glm_perf5.sh" | sed 's/^/  /'
log "provisioned in $(( $(date +%s) - t0 ))s; starting glm_perf5.sh"
bash "$B/glm_perf5.sh" > "$R/glm_perf5.log" 2>&1
tail -n 12 "$R/glm_perf5.log"
log "GLMHOUR DONE"
