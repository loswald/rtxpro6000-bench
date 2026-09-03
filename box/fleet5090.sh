#!/usr/bin/env bash
# 8x RTX 5090 campaign. Same silicon as the RTX PRO 6000 (sm_120), a third of the memory (32 GB), twice
# the cards, and at Scan the SAME monthly price (GB8-32T £1,999.98 inc VAT vs the 4x PRO 6000 at £1,999.98),
# so every number here is a like-for-like £-comparison with the 4x results.
#
# Layouts on 8 cards: tp=1 -> 8 replicas; tp=2 -> 4 replicas; tp=4 -> 2 replicas (one per socket on the
# dual-socket Vast host: GPUs 0-3 are NUMA 0, 4-7 NUMA 1, so TP4 never crosses the SYS path); tp=8 -> 1
# server (crosses sockets on this host; Scan's GB8-32T is a single EPYC 9354P, so TP8 there is better).
# Workloads and methodology identical to the PRO 6000 campaign: router 1024/128, promptopt 3072-prefix
# 512/256, judge 4096/512; steady state (6x prompts per slot); chat tripwire at a 2k budget; the kernel the
# engine actually selected is recorded; spec-decode acceptance counters are read when speculation is on.
set -u
B=/workspace/bench; R=/workspace/results; P=$R/probe; S=$R/smoke; MD=/workspace/models
mkdir -p "$P" "$S"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
source /workspace/bench/hardkill.sh
NGPU=8

serve(){ # tag dir tp [extra...]
  local tag="$1" dir="$2" tp="$3"; shift 3
  local n=$(( NGPU / tp ))
  kill_all
  {
    echo '#!/usr/bin/env bash'
    echo 'export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 HF_HUB_OFFLINE=1'
    echo 'export FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 CUTE_DSL_ARCH=sm_120a'
    echo 'export VLLM_ENGINE_READY_TIMEOUT_S=3600 MAX_JOBS=4 NVCC_THREADS=2'
    [ -n "${EXTRA_ENV:-}" ] && echo "export $EXTRA_ENV"
    for i in $(seq 0 $((n-1))); do
      local gpus; gpus=$(seq -s, $((i*tp)) $((i*tp+tp-1)))
      # 32 GB cards: keep the sequence budget proportional to what fits; util 0.92 leaves ~2.5 GB for the runtime.
      printf 'CUDA_VISIBLE_DEVICES=%s vllm serve %q --served-model-name m --host 0.0.0.0 --port %d --tensor-parallel-size %d --kv-cache-dtype fp8 --max-model-len 32768 --max-num-seqs %d --max-num-batched-tokens 8192 --gpu-memory-utilization 0.92 --enable-prefix-caching --trust-remote-code --disable-custom-all-reduce --no-enable-flashinfer-autotune --disable-uvicorn-access-log' \
        "$gpus" "$dir" $((8000+i)) "$tp" $(( tp >= 4 ? 256 : 256 ))
      for a in "$@"; do printf ' %q' "$a"; done
      printf ' > %s/%s_p%d.log 2>&1 &\nsleep 2\n' "$S" "$tag" $((8000+i))
    done
    echo 'wait'
  } > "$B/l_59.sh"
  chmod +x "$B/l_59.sh"
  log "  launch $tag :: tp=$tp x$n :: $*"
  tmux new-session -d -s srv "bash $B/l_59.sh"
  local t=0 ok=0
  while [ "$t" -lt 2400 ]; do
    ok=1; for i in $(seq 0 $((n-1))); do curl -fsS -m 3 "http://127.0.0.1:$((8000+i))/health" >/dev/null 2>&1 || ok=0; done
    [ "$ok" = 1 ] && break
    grep -qiE "^(ValueError|RuntimeError|NotImplementedError)|Engine core initialization failed|Address already in use|CUDA out of memory" "$S/${tag}_p8000.log" 2>/dev/null && { sleep 20; break; }
    sleep 15; t=$((t+15))
  done
  if [ "$ok" != 1 ]; then
    log "  $tag FAILED after ${t}s"
    grep -iE "error|not supported|no kernel|Traceback|ValueError|out of memory|Unknown|NotImplemented|does not support" "$S/${tag}_p8000.log" 2>/dev/null \
      | grep -vE "import_utils|deep_ep|min_frames|max_frames|WARNING" | tail -6 | cut -c1-200
    return 1
  fi
  log "  $tag healthy ${t}s | $(grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$S/${tag}_p8000.log") per server x$n"
  grep -m3 -oE "Using [A-Za-z0-9_]+ (attention backend|for NVFP4 GEMM)|Using '[A-Z0-9_]+' [A-Za-z0-9]+ MoE backend|Selected [A-Za-z0-9]+Kernel|Using [A-Za-z0-9]+Kernel for [A-Z0-9 ]+" "$S/${tag}_p8000.log" | sort -u | sed 's/^/    kernel: /'
  return 0
}

pt(){ # tag label in out prefix c_per_server n dir
  local tag=$1 label=$2 in=$3 out=$4 pre=$5 c=$6 n=$7 dir=$8
  local np=$(( c*6 )) tot=$(( c*n ))
  mkdir -p "$P/$tag"
  for i in $(seq 0 $((n-1))); do
    vllm bench serve --backend openai --base-url "http://127.0.0.1:$((8000+i))" --endpoint /v1/completions \
      --model m --tokenizer "$dir" --trust-remote-code \
      --dataset-name random --random-input-len "$in" --random-output-len "$out" --random-prefix-len "$pre" --random-range-ratio 0 \
      --request-rate inf --max-concurrency "$c" --num-prompts "$np" --ignore-eos --seed $((5090+c+in+i)) \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --disable-tqdm \
      --save-result --result-dir "$P/$tag" --result-filename "${tag}__${label}__c${tot}__p$((8000+i)).json" \
      > "$P/$tag/${label}_c${tot}_p$((8000+i)).log" 2>&1 &
  done
  wait
  python3 "$B/agg.py" "$P/$tag" "${tag}__${label}__c${tot}__p" "$label" "$tot" "$tag"
}

run(){ # tag dir tp [extra...]
  local tag="$1" dir="$2" tp="$3"; shift 3
  local n=$(( NGPU / tp ))
  [ -f "$dir/config.json" ] || { log "SKIP $tag (no weights)"; return 1; }
  log "########## $tag  ($(du -sh "$dir" | cut -f1), tp=$tp x$n) ##########"
  serve "$tag" "$dir" "$tp" "$@" || return 1
  python3 "$B/quality20.py" m http://127.0.0.1:8000 "$P/${tag}_quality20.json" --mode chat --max-tokens 2048 2>&1 | tail -1
  # node-level concurrency matched to the PRO 6000 runs: 256 and 1024 for router, 1024 promptopt, 512 judge
  pt "$tag" router    1024 128 0    $(( 256 / n ))  "$n" "$dir"
  pt "$tag" router    1024 128 0    $(( 1024 / n )) "$n" "$dir"
  pt "$tag" promptopt  512 256 3072 $(( 1024 / n )) "$n" "$dir"
  pt "$tag" judge     4096 512 0    $(( 512 / n ))  "$n" "$dir"
  curl -fsS -m 5 http://127.0.0.1:8000/metrics 2>/dev/null | grep -E "spec_decode.*(accepted|draft)" | head -3 | sed 's/^/    spec: /'
  kill_all
}

log "===== 8x RTX 5090 : replica tier (one model per 32 GB card, eight replicas) ====="
# Qwen3.8-27B NVFP4 (AA 52): the headline comparison. The PRO 6000 x4 did 5,161 out tok/s at C1024 with b12x.
run q27_nvfp4_b12x   "$MD/Qwen27B-NVFP4-RTX5090" 1 --kernel-config.linear_backend b12x
run q27_nvfp4_auto   "$MD/Qwen27B-NVFP4-RTX5090" 1
[ -f "$MD/Qwen27B-DSpark-NVFP4/config.json" ] && \
run q27_nvfp4_dspark "$MD/Qwen27B-NVFP4-RTX5090" 1 --kernel-config.linear_backend b12x \
    --speculative-config "{\"method\":\"draft_model\",\"model\":\"$MD/Qwen27B-DSpark-NVFP4\",\"num_speculative_tokens\":7,\"draft_tensor_parallel_size\":1}"
run gptoss20         "$MD/gpt-oss-20b"           1 --moe-backend flashinfer_cutlass --quantization-config.moe.activation mxfp8
run nemo35           "$MD/Nemotron-3.5-Lightning-30B" 1 --kernel-config.linear_backend b12x --moe-backend flashinfer_cutlass
run muse30_nvfp4     "$MD/Muse-Glimmer-30B-NVFP4" 1 --kernel-config.linear_backend b12x --moe-backend flashinfer_cutlass
run gemma26_nvfp4    "$MD/gemma-4-26B-A4B-NVFP4"  1 --kernel-config.linear_backend b12x --moe-backend flashinfer_cutlass
run qwen36_nvfp4     "$MD/Qwen3.6-35B-A3B-NVFP4"  1 --kernel-config.linear_backend b12x --moe-backend flashinfer_cutlass

log "===== 8x RTX 5090 : TP tier ====="
# FP8 Qwen3.8-27B does not fit a 32 GB card: TP2 x4. Direct read of what the memory ceiling costs vs NVFP4 x8.
run q27_fp8_tp2      "$MD/Qwen3.8-27B-FP8"       2 --kernel-config.linear_backend b12x
# gpt-oss-120b (61 GB): TP4 x2, one replica per socket. PRO 6000 x4 replicas did 10,913 router out tok/s.
run gptoss120_tp4    "$MD/gpt-oss-120b"          4 --moe-backend flashinfer_cutlass --quantization-config.moe.activation mxfp8
# Ling-3.0-flash NVFP4 (81 GB, AA 38, 5B active): TP4 x2.
run ling3f_tp4       "$MD/Ling-3.0-flash-NVFP4"  4 --kernel-config.linear_backend b12x --moe-backend flashinfer_cutlass
# Qwen3.8-Flash-Next NVFP4 (126 GB, AA 56): TP8, ~16 GB weights per card. Crosses sockets on this host.
# The default GDN decode kernel deadlocks at ~32 concurrency on RTX PRO 6000 reports; Triton is the documented fix.
EXTRA_ENV="VLLM_GDN_DECODE_KERNEL=triton" run qwen38fn_tp8     "$MD/Qwen3.8-Flash-Next-NVFP4" 8 --kernel-config.linear_backend b12x
EXTRA_ENV="VLLM_GDN_DECODE_KERNEL=triton" run qwen38fn_tp8_mtp "$MD/Qwen3.8-Flash-Next-NVFP4" 8 --kernel-config.linear_backend b12x \
    --speculative-config '{"method":"qwen4_exp_mtp","num_speculative_tokens":3}'

log "FLEET5090 DONE"
kill_all
