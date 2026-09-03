#!/usr/bin/env bash
# The whole fleet, nothing skipped. Nish: "All the models have different complementary
# strengths and weaknesses. We are building multi-model systems." So every downloaded
# near-frontier model gets the same treatment: chat-mode corruption tripwire first, then the
# three research shapes at steady state, recording which kernel the server actually chose.
#   replica tier : one model per card, four independent replicas
#   TP tier      : models that need 2 or 4 cards
B=/workspace/bench; R=/workspace/results; P=$R/probe; S=$R/smoke; MD=/workspace/models
mkdir -p "$P" "$S"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
source /workspace/bench/hardkill.sh
CLEAN="env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH"
COMMON="--kv-cache-dtype fp8 --max-model-len 40960 --max-num-seqs 512 --max-num-batched-tokens 8192 \
 --gpu-memory-utilization 0.94 --compilation-config {\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"} \
 --no-enable-flashinfer-autotune --enable-prefix-caching --trust-remote-code --disable-uvicorn-access-log"
ENV="export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_OFFLINE=1 MAX_JOBS=6 NVCC_THREADS=2"

wait_ports(){ # tag ports...
  local tag=$1; shift; local t=0 ok=0
  while [ "$t" -lt 1800 ]; do
    ok=1; for p in "$@"; do curl -fsS -m 3 "http://127.0.0.1:$p/health" >/dev/null 2>&1 || ok=0; done
    [ "$ok" = 1 ] && break
    grep -qiE "ValueError|not supported|Traceback|Unknown|invalid" "$S/${tag}_p8000.log" 2>/dev/null && { sleep 25; curl -fsS -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1 || break; }
    sleep 10; t=$((t+10))
  done
  if [ "$ok" != 1 ]; then
    log "  $tag FAILED after ${t}s"
    grep -iE "ValueError|not supported|no kernel|out of memory|Error|Unknown|invalid" "$S/${tag}_p8000.log" 2>/dev/null \
      | grep -vE "import_utils|deep_ep|WARNING|min_frames|max_frames" | head -3 | cut -c1-200
    return 1
  fi
  log "  $tag healthy ${t}s | $(grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$S/${tag}_p8000.log")"
  grep -m2 -oE "Using [A-Za-z0-9_' ]+ (MoE|Mxfp4|NVFP4)[A-Za-z ]*backend[^,]*|Using [A-Za-z0-9]+ for [A-Z0-9]+ GEMM|Selected [A-Za-z0-9]+Kernel" "$S/${tag}_p8000.log" | sed 's/^/    kernel: /'
  return 0
}

serve_x4(){ # tag dir [extra...]   four replicas, ports 8000-8003
  local tag="$1" dir="$2"; shift 2; kill_all
  cat > "$B/l_fl.sh" <<L
#!/usr/bin/env bash
$ENV
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=\$i vllm serve $dir --served-model-name m --host 0.0.0.0 --port \$((8000+i)) $COMMON $* \\
    > $S/${tag}_p\$((8000+i)).log 2>&1 &
  sleep 2
done
wait
L
  chmod +x "$B/l_fl.sh"; log "  launch $tag :: $(basename "$dir") $*"
  tmux new-session -d -s srv "bash $B/l_fl.sh"
  wait_ports "$tag" 8000 8001 8002 8003
}
serve_tp(){ # tag dir tp replicas [extra...]   tp-way sharded, `replicas` copies on consecutive card groups
  local tag="$1" dir="$2" tp=$3 reps=$4; shift 4; kill_all
  cat > "$B/l_fl.sh" <<L
#!/usr/bin/env bash
$ENV
for r in \$(seq 0 $((reps-1))); do
  devs=\$(seq -s, \$((r*$tp)) \$((r*$tp+$tp-1)))
  CUDA_VISIBLE_DEVICES=\$devs vllm serve $dir --served-model-name m --host 0.0.0.0 --port \$((8000+r)) \\
    --tensor-parallel-size $tp --disable-custom-all-reduce $COMMON $* \\
    > $S/${tag}_p\$((8000+r)).log 2>&1 &
  sleep 2
done
wait
L
  chmod +x "$B/l_fl.sh"; log "  launch $tag :: $(basename "$dir") tp$tp x$reps $*"
  tmux new-session -d -s srv "bash $B/l_fl.sh"
  local ports=(); for r in $(seq 0 $((reps-1))); do ports+=($((8000+r))); done
  wait_ports "$tag" "${ports[@]}"
}

pt(){ # tag dir label in out prefix c_per_port nports
  local tag=$1 dir=$2 label=$3 in=$4 out=$5 pre=$6 c=$7 np_=$8
  local np=$(( c*8 )) tot=$(( c*np_ ))
  mkdir -p "$P/$tag"
  for r in $(seq 0 $((np_-1))); do
    local p=$((8000+r))
    $CLEAN vllm bench serve --backend openai --base-url "http://127.0.0.1:$p" --endpoint /v1/completions \
      --model m --tokenizer "$dir" --trust-remote-code \
      --dataset-name random --random-input-len "$in" --random-output-len "$out" \
      --random-prefix-len "$pre" --random-range-ratio 0 \
      --request-rate inf --max-concurrency "$c" --num-prompts "$np" --ignore-eos --seed $((8500+c+in)) \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --disable-tqdm \
      --save-result --result-dir "$P/$tag" --result-filename "${tag}__${label}__c${tot}__p${p}.json" \
      > "$P/$tag/${label}_c${tot}_p${p}.log" 2>&1 &
  done
  wait
  $CLEAN python3 "$B/agg.py" "$P/$tag" "${tag}__${label}__c${tot}__p" "$label" "$tot" "$tag"
}

shapes(){ # tag dir nports
  local tag=$1 dir=$2 n=$3
  $CLEAN python3 "$B/quality20.py" m http://127.0.0.1:8000 "$P/${tag}_quality20.json" --mode chat --max-tokens 1024 2>&1 | tail -1
  pt "$tag" "$dir" router    1024 128 0    $((256/n*1)) "$n"
  pt "$tag" "$dir" promptopt  512 256 3072 $((256/n*1)) "$n"
  pt "$tag" "$dir" judge     4096 512 0    $((128/n*1)) "$n"
}

rep(){ # alias dir [extra...]
  local alias=$1 dir=$2; shift 2
  [ -f "$dir/config.json" ] || { log "SKIP $alias (not downloaded)"; return; }
  log "########## $alias ($(du -sh "$dir" | cut -f1)) ##########"
  serve_x4 "f2_$alias" "$dir" "$@" && shapes "f2_$alias" "$dir" 4
}
tpm(){ # alias dir tp replicas [extra...]
  local alias=$1 dir=$2 tp=$3 reps=$4; shift 4
  [ -f "$dir/config.json" ] || { log "SKIP $alias (not downloaded)"; return; }
  log "########## $alias ($(du -sh "$dir" | cut -f1)) tp$tp x$reps ##########"
  serve_tp "f2_$alias" "$dir" "$tp" "$reps" "$@" && shapes "f2_$alias" "$dir" "$reps"
}

log "===== FLEET, replica tier ====="
rep nemo35   "$MD/Nemotron-3.5-Lightning-30B" --kernel-config.linear_backend b12x
rep qwen36   "$MD/Qwen3.6-35B-A3B-FP8"        --kernel-config.linear_backend b12x
rep muse30   "$MD/Muse-Glimmer-30B"
rep gptoss20 "$MD/gpt-oss-20b" --moe-backend flashinfer_cutlass --quantization-config.moe.activation mxfp8
rep gemma26  "$MD/gemma-4-26B-A4B-it"
rep gemma31  "$MD/gemma-4-31B-it"

log "===== FLEET, TP tier ====="
# MiniMax-M3 MXFP4: the same MoE path that won on gpt-oss, plus its own MTP head
tpm minimaxm3 "$MD/MiniMax-M3-MXFP4" 4 1 --moe-backend flashinfer_cutlass --quantization-config.moe.activation mxfp8
tpm minimaxm3_mtp "$MD/MiniMax-M3-MXFP4" 4 1 --moe-backend flashinfer_cutlass --quantization-config.moe.activation mxfp8 \
  --speculative-config '{"method":"minimax_m3_mtp","num_speculative_tokens":2}'
# Qwen3.8-Flash-Next NVFP4: two TP2 replicas, sm_120 native NVFP4 kernel
tpm qwen38fn  "$MD/Qwen3.8-Flash-Next-NVFP4" 2 2 --kernel-config.linear_backend b12x
log "FLEET2 DONE"
kill_all
