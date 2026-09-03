#!/usr/bin/env bash
# NVFP4 on sm_120, explicit kernel selection. Auto-selection took the W4A16 CuTe-DSL kernel
# (4-bit weights dequantised against 16-bit activations) and lost 39% to FP8 at C256, even
# though the checkpoint is W4A4 and vLLM ships two B12x W4A4 kernels for capability 120.
# Same shapes and session discipline as nvtier.sh, so the FP8 control (2,642 out/s at C256,
# 3,146 at C1024) is directly comparable.
B=/workspace/bench; R=/workspace/results; P=$R/probe; S=$R/smoke; MD=/workspace/models
mkdir -p "$P" "$S"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
source /workspace/bench/hardkill.sh
CLEAN="env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH"
DIR=$MD/Qwen27B-NVFP4-RTX5090

serve_x4(){ # tag [extra...]
  local tag="$1"; shift
  kill_all
  cat > "$B/l_n2.sh" <<L
#!/usr/bin/env bash
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_OFFLINE=1
export MAX_JOBS=6 NVCC_THREADS=2
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=\$i vllm serve $DIR --served-model-name q27 \\
    --host 0.0.0.0 --port \$((8000+i)) \\
    --kv-cache-dtype fp8 --max-model-len 40960 --max-num-seqs 512 \\
    --max-num-batched-tokens 8192 --gpu-memory-utilization 0.96 \\
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \\
    --no-enable-flashinfer-autotune --enable-prefix-caching --trust-remote-code \\
    --disable-uvicorn-access-log $* \\
    > /workspace/results/smoke/${tag}_p\$((8000+i)).log 2>&1 &
  sleep 2
done
wait
L
  chmod +x "$B/l_n2.sh"
  log "  launch $tag :: $*"
  tmux new-session -d -s srv "bash $B/l_n2.sh"
  local t=0 ok=0
  while [ "$t" -lt 1500 ]; do
    ok=1; for p in 8000 8001 8002 8003; do curl -fsS -m 3 "http://127.0.0.1:$p/health" >/dev/null 2>&1 || ok=0; done
    [ "$ok" = 1 ] && break
    grep -qiE "ValueError|not supported|Traceback|Unknown|invalid" "$S/${tag}_p8000.log" 2>/dev/null && { sleep 20; curl -fsS -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1 || break; }
    sleep 10; t=$((t+10))
  done
  if [ "$ok" != 1 ]; then
    log "  $tag FAILED after ${t}s"
    grep -iE "ValueError|not supported|no kernel|out of memory|Error|Unknown|invalid" "$S/${tag}_p8000.log" 2>/dev/null \
      | grep -vE "import_utils|deep_ep|WARNING|min_frames|max_frames" | head -3 | cut -c1-200
    return 1
  fi
  log "  $tag healthy ${t}s | $(grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$S/${tag}_p8000.log")"
  grep -m1 -oE "Using [A-Za-z0-9]+ for NVFP4 GEMM|Selected [A-Za-z0-9]+Kernel" "$S/${tag}_p8000.log" | sed 's/^/    kernel: /'
  return 0
}

pt(){ # tag label in out prefix c_per_port
  local tag=$1 label=$2 in=$3 out=$4 pre=$5 c=$6
  local np=$(( c*8 )) tot=$(( c*4 ))
  mkdir -p "$P/$tag"
  for i in 0 1 2 3; do
    local p=$((8000+i))
    $CLEAN vllm bench serve --backend openai --base-url "http://127.0.0.1:$p" --endpoint /v1/completions \
      --model q27 --tokenizer "$DIR" --trust-remote-code \
      --dataset-name random --random-input-len "$in" --random-output-len "$out" \
      --random-prefix-len "$pre" --random-range-ratio 0 \
      --request-rate inf --max-concurrency "$c" --num-prompts "$np" --ignore-eos --seed $((8300+c+in)) \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --disable-tqdm \
      --save-result --result-dir "$P/$tag" --result-filename "${tag}__${label}__c${tot}__p${p}.json" \
      > "$P/$tag/${label}_c${tot}_p${p}.log" 2>&1 &
  done
  wait
  $CLEAN python3 "$B/agg.py" "$P/$tag" "${tag}__${label}__c${tot}__p" "$label" "$tot" "$tag"
}

sweep(){ # tag [extra...]
  local tag="$1"; shift
  # resumable: an arm with all four shape results already on disk is not repeated
  if [ "$(ls "$P/$tag"/*__judge__*.json 2>/dev/null | wc -l)" -ge 4 ]; then log "  $tag already measured, skipping"; return; fi
  if serve_x4 "$tag" "$@"; then
    $CLEAN python3 "$B/quality20.py" q27 http://127.0.0.1:8000 "$P/${tag}_quality20.json" --mode chat --max-tokens 1024 2>&1 | tail -1
    pt "$tag" router    1024 128 0     64
    pt "$tag" router    1024 128 0    256
    pt "$tag" promptopt  512 256 3072 256
    pt "$tag" judge     4096 512 0    128
  fi
}

log "===== Qwen3.8-27B NVFP4 (W4A4 checkpoint) : explicit kernel backends on sm_120 ====="
for be in b12x flashinfer_b12x flashinfer_cutedsl cutlass; do
  log "--- modelopt build, linear_backend=$be ---"
  sweep "n2_$be" --kernel-config.linear_backend "$be"
done
# the compressed-tensors NVFP4 builds (llm-compressor) go through a different scheme and
# kernel selector; test auto and the sm_120 native kernel on each
for ck in RedHat unsloth; do
  DIR=$MD/Qwen27B-NVFP4-$ck
  [ -f "$DIR/config.json" ] || { log "SKIP $ck (not downloaded)"; continue; }
  log "--- $ck build, auto ---";  sweep "n2_${ck}_auto"
  log "--- $ck build, b12x ---";  sweep "n2_${ck}_b12x" --kernel-config.linear_backend b12x
done
# speculative decoding with the sm_120-native kernel: the DSpark drafter (1.4 GB, modelopt
# fp4). Acceptance counters are the thing to read; at our concurrency it may still lose.
DIR=$MD/Qwen27B-NVFP4-RTX5090
log "--- modelopt build, b12x + DSpark drafter ---"
sweep n2_b12x_dspark --kernel-config.linear_backend b12x \
  --speculative-config "{\"method\":\"draft_model\",\"model\":\"$MD/Qwen27B-DSpark-NVFP4\",\"num_speculative_tokens\":7,\"draft_tensor_parallel_size\":1}"
curl -fsS -m 5 http://127.0.0.1:8000/metrics 2>/dev/null | grep -E "spec_decode.*(accepted|draft|emitted)" | head -4 | sed 's/^/    spec: /'
log "NVTIER2 DONE"
kill_all
