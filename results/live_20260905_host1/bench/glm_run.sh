#!/usr/bin/env bash
# Stand up the vendor vLLM (glm5_next-capable) beside the working one and serve
# GLM-5.3-Flash, the highest-scoring open-weight model that fits 4x96 GB.
# Recipe from vLLM's own GLM-5.3-Flash page and LibertAI's NVFP4 card:
#   TP4, fp8 KV, MTP speculation with 5 draft tokens, and on vLLM the reasoning
#   parser is deepseek_r1 (glm45 is SGLang-only), tool parser glm47.
set -u
IMG=/workspace/glmimg
B=/workspace/bench; R=/workspace/results; P=$R/probe; S=$R/smoke
MD=/workspace/models/GLM-5.3-Flash-NVFP4
mkdir -p "$P" "$S"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
source /workspace/bench/hardkill.sh

SP=$(find "$IMG" -maxdepth 6 -type d -name "vllm" -path "*packages*" 2>/dev/null | head -1)
[ -z "$SP" ] && { log "no vllm found in the extracted image"; exit 1; }
SITE=$(dirname "$SP")
log "vendor site-packages: $SITE"
log "  vendor vllm version: $(cat "$SP/version.py" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+[^\"]*' | head -1)"
log "  glm5 model files: $(ls "$SP/model_executor/models/" 2>/dev/null | grep -ci glm5) present"

log "does the vendor build import against our torch?"
PYTHONPATH="$SITE" python3 -c "
import vllm, sys
print('  imported vllm', vllm.__version__, 'from', vllm.__file__[:60])
from vllm.model_executor.models.registry import ModelRegistry as M
a=[x for x in M.get_supported_archs() if 'Glm5' in x or 'GLM5' in x]
print('  glm5 archs:', a)
" 2>&1 | tail -6

log "launching GLM-5.3-Flash TP4 with MTP speculation"
cat > "$B/l_glm.sh" <<L
#!/usr/bin/env bash
export PYTHONPATH="$SITE"
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_OFFLINE=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export MAX_JOBS=6 NVCC_THREADS=2
exec python3 -m vllm.entrypoints.openai.api_server \\
  --model $MD --served-model-name glm53f \\
  --host 0.0.0.0 --port 8000 \\
  --tensor-parallel-size 4 \\
  --kv-cache-dtype auto --max-model-len 40960 --max-num-seqs 256 \\
  --max-num-batched-tokens 8192 --gpu-memory-utilization 0.92 \\
  --speculative-config '{"method":"glm5_next_mtp","num_speculative_tokens":5}' \\
  --reasoning-parser deepseek_r1 --tool-call-parser glm47 \\
  --enable-prefix-caching --trust-remote-code --disable-custom-all-reduce \\
  --disable-uvicorn-access-log
L
chmod +x "$B/l_glm.sh"
kill_all
tmux new-session -d -s glmsrv "bash $B/l_glm.sh > $S/glm53f.log 2>&1; echo EXIT=\$? >> $S/glm53f.log"
t=0; ok=0
while [ "$t" -lt 2700 ]; do
  curl -fsS -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
  grep -q "^EXIT=" "$S/glm53f.log" 2>/dev/null && break
  sleep 15; t=$((t+15))
done
if [ "$ok" != 1 ]; then
  log "GLM FAILED after ${t}s"
  grep -iE "error|not supported|no kernel|Traceback|ValueError|out of memory|Unknown" "$S/glm53f.log" \
    | grep -vE "import_utils|deep_ep" | tail -8 | cut -c1-200
  exit 1
fi
log "GLM healthy in ${t}s | $(grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$S/glm53f.log")"

pt(){ # label in out prefix conc
  local label=$1 in=$2 out=$3 pre=$4 c=$5
  mkdir -p "$P/glm53f"
  vllm bench serve --backend openai --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
    --model glm53f --tokenizer "$MD" --trust-remote-code \
    --dataset-name random --random-input-len "$in" --random-output-len "$out" \
    --random-prefix-len "$pre" --random-range-ratio 0 \
    --request-rate inf --max-concurrency "$c" --num-prompts $((c*6)) --ignore-eos --seed $((9000+c+in)) \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --disable-tqdm \
    --save-result --result-dir "$P/glm53f" --result-filename "glm53f__${label}__c${c}__p8000.json" \
    > "$P/glm53f/${label}_c${c}.log" 2>&1
  python3 "$B/agg.py" "$P/glm53f" "glm53f__${label}__c${c}__p" "$label" "$c" glm53f
}
for c in 64 256 512; do pt router 1024 128 0 "$c"; done
pt promptopt 512 256 3072 256
pt judge 4096 512 0 128
a=$(curl -fsS -m 5 http://127.0.0.1:8000/metrics 2>/dev/null | grep -E "spec_decode" | head -4)
log "MTP acceptance counters:"; echo "$a" | sed 's/^/    /'
python3 "$B/quality20.py" glm53f http://127.0.0.1:8000 "$P/glm53f_quality20.json" 2>&1 | tail -1
log "GLM DONE"
kill_all