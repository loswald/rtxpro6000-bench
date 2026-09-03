#!/usr/bin/env bash
# Shared launch/measure helpers. Sourced by the round scripts.
# Every point records: req/s, output tok/s, input tok/s, total tok/s,
# TTFT mean + p50 + p99, TPOT mean + p50 + p99, ITL and end-to-end latency
# percentiles (kept in the per-run JSON), plus duration and completion counts.

launch_x4(){ # tag model alias [extra vllm args...]
  local tag="$1" model="$2" alias="$3"; shift 3
  kill_all
  cat > /workspace/bench/l_lib.sh <<L
#!/usr/bin/env bash
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_OFFLINE=1
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=\$i vllm serve $model --served-model-name $alias \\
    --host 0.0.0.0 --port \$((8000+i)) \\
    --kv-cache-dtype fp8 --max-model-len 40960 --max-num-seqs 512 --max-num-batched-tokens 8192 \\
    --gpu-memory-utilization 0.96 \\
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \\
    --no-enable-flashinfer-autotune --enable-prefix-caching --trust-remote-code \\
    --disable-uvicorn-access-log $* \\
    > /workspace/results/smoke/${tag}_p\$((8000+i)).log 2>&1 &
  sleep 2
done
wait
L
  chmod +x /workspace/bench/l_lib.sh
  echo "[$(date +%H:%M:%S)] launch $tag :: $*"
  tmux new-session -d -s srv "bash /workspace/bench/l_lib.sh"
  local t=0 ok=0
  while [ "$t" -lt 720 ]; do
    ok=1
    for p in 8000 8001 8002 8003; do
      curl -fsS -m 3 "http://127.0.0.1:$p/health" >/dev/null 2>&1 || ok=0
    done
    [ "$ok" = 1 ] && break
    sleep 10; t=$((t+10))
  done
  if [ "$ok" != 1 ]; then
    echo "[$(date +%H:%M:%S)] $tag FAILED to become healthy"
    grep -iE "error|invalid|does not support" "/workspace/results/smoke/${tag}_p8000.log" 2>/dev/null \
      | grep -vE "import_utils|deep_ep" | head -3 | cut -c1-200
    return 1
  fi
  echo "[$(date +%H:%M:%S)] $tag healthy ${t}s | $(grep -m1 -oE "GPU KV cache size: [0-9,]+ tokens" "/workspace/results/smoke/${tag}_p8000.log")"
  return 0
}

point(){ # tag label in out prefix c_per_port [alias] [model]
  local tag=$1 label=$2 in=$3 out=$4 pre=$5 c=$6
  local alias=${7:-gptoss} model=${8:-/workspace/models/gpt-oss-120b}
  local np=$(( c * 8 )) tot=$(( c * 4 ))
  mkdir -p "/workspace/results/probe/$tag"
  local i p
  for i in 0 1 2 3; do
    p=$((8000+i))
    vllm bench serve --backend openai --base-url "http://127.0.0.1:$p" --endpoint /v1/completions \
      --model "$alias" --tokenizer "$model" --trust-remote-code \
      --dataset-name random --random-input-len "$in" --random-output-len "$out" \
      --random-prefix-len "$pre" --random-range-ratio 0 \
      --request-rate inf --max-concurrency "$c" --num-prompts "$np" --ignore-eos \
      --seed $((9000 + c + in)) \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --disable-tqdm \
      --save-result --result-dir "/workspace/results/probe/$tag" \
      --result-filename "${tag}__${label}__c${tot}__p${p}.json" \
      > "/workspace/results/probe/$tag/${label}_c${tot}_p${p}.log" 2>&1 &
  done
  wait
  python3 /workspace/bench/agg.py "/workspace/results/probe/$tag" "${tag}__${label}__c${tot}__p" "$label" "$tot" "$tag"
}
