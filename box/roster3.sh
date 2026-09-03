#!/usr/bin/env bash
# ROSTER ROTATION: the near-frontier open models the 3 Sept audit found missing, each one
# downloaded, measured and (unless KEEP=1) deleted again, because the box has ~1.2 TB and the
# roster is ~1.5 TB. Runs after fleet2 (chained). Self-contained: its own launcher, so it does
# not depend on fleet2.sh's function names.
#
# Layouts: tp=1 -> 4 independent replicas (ports 8000-8003), tp=2 -> 2 replicas, tp=4 -> 1 server.
# Every model gets: chat tripwire (2k budget), router C64+C256 (per node), promptopt, judge,
# spec-decode counters when speculation is on, and the kernel line the engine actually chose.
set -u
B=/workspace/bench; R=/workspace/results; P=$R/probe; S=$R/smoke; MD=/workspace/models
mkdir -p "$P" "$S"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
source /workspace/bench/hardkill.sh
export HF_HUB_ENABLE_HF_TRANSFER=1

fetch(){ # repo dir
  local d="$MD/$2"
  [ -f "$d/config.json" ] && { log "  have $2"; return 0; }
  log "  downloading $1 -> $2"
  hf download "$1" --local-dir "$d" > "/workspace/dl_$2.log" 2>&1 \
    && { log "  done $2 ($(du -sh "$d" | cut -f1))"; return 0; } \
    || { log "  DOWNLOAD FAILED $2: $(tail -1 /workspace/dl_$2.log | cut -c1-120)"; return 1; }
}

serve(){ # tag dir tp [extra...]   -> replicas = 4/tp
  local tag="$1" dir="$2" tp="$3"; shift 3
  local n=$(( 4 / tp ))
  kill_all
  {
    echo '#!/usr/bin/env bash'
    echo 'export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 HF_HUB_OFFLINE=1'
    echo 'export FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 CUTE_DSL_ARCH=sm_120a'
    echo 'export VLLM_ENGINE_READY_TIMEOUT_S=3600 MAX_JOBS=6 NVCC_THREADS=2'
    for i in $(seq 0 $((n-1))); do
      local gpus; gpus=$(seq -s, $((i*tp)) $((i*tp+tp-1)))
      printf 'CUDA_VISIBLE_DEVICES=%s vllm serve %q --served-model-name m --host 0.0.0.0 --port %d --tensor-parallel-size %d --kv-cache-dtype fp8 --max-model-len 40960 --max-num-seqs %d --max-num-batched-tokens 8192 --gpu-memory-utilization 0.92 --enable-prefix-caching --trust-remote-code --disable-custom-all-reduce --no-enable-flashinfer-autotune --disable-uvicorn-access-log' \
        "$gpus" "$dir" $((8000+i)) "$tp" $(( tp == 4 ? 256 : 512 ))
      for a in "$@"; do printf ' %q' "$a"; done
      printf ' > %s/%s_p%d.log 2>&1 &\nsleep 2\n' "$S" "$tag" $((8000+i))
    done
    echo 'wait'
  } > "$B/l_r3.sh"
  chmod +x "$B/l_r3.sh"
  log "  launch $tag :: tp=$tp x$n :: $*"
  tmux new-session -d -s srv "bash $B/l_r3.sh"
  local t=0 ok=0
  while [ "$t" -lt 2400 ]; do
    ok=1; for i in $(seq 0 $((n-1))); do curl -fsS -m 3 "http://127.0.0.1:$((8000+i))/health" >/dev/null 2>&1 || ok=0; done
    [ "$ok" = 1 ] && break
    grep -qiE "^(ValueError|RuntimeError|NotImplementedError)|Engine core initialization failed|Address already in use" "$S/${tag}_p8000.log" 2>/dev/null && { sleep 20; break; }
    sleep 15; t=$((t+15))
  done
  if [ "$ok" != 1 ]; then
    log "  $tag FAILED after ${t}s"
    grep -iE "error|not supported|no kernel|Traceback|ValueError|out of memory|Unknown|NotImplemented|does not support" "$S/${tag}_p8000.log" 2>/dev/null \
      | grep -vE "import_utils|deep_ep|min_frames|max_frames|WARNING" | tail -6 | cut -c1-200
    return 1
  fi
  log "  $tag healthy ${t}s | $(grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$S/${tag}_p8000.log")"
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
      --request-rate inf --max-concurrency "$c" --num-prompts "$np" --ignore-eos --seed $((9700+c+in+i)) \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --disable-tqdm \
      --save-result --result-dir "$P/$tag" --result-filename "${tag}__${label}__c${tot}__p$((8000+i)).json" \
      > "$P/$tag/${label}_c${tot}_p$((8000+i)).log" 2>&1 &
  done
  wait
  python3 "$B/agg.py" "$P/$tag" "${tag}__${label}__c${tot}__p" "$label" "$tot" "$tag"
}

run(){ # tag dir tp [extra...]
  local tag="$1" dir="$2" tp="$3"; shift 3
  local n=$(( 4 / tp ))
  [ -f "$dir/config.json" ] || { log "SKIP $tag (no weights)"; return 1; }
  log "########## $tag  ($(du -sh "$dir" | cut -f1), tp=$tp x$n) ##########"
  serve "$tag" "$dir" "$tp" "$@" || return 1
  python3 "$B/quality20.py" m http://127.0.0.1:8000 "$P/${tag}_quality20.json" --mode chat --max-tokens 2048 2>&1 | tail -1
  local c64=$(( 64 / n )) c256=$(( 256 / n ))
  pt "$tag" router    1024 128 0    "$c64"  "$n" "$dir"
  pt "$tag" router    1024 128 0    "$c256" "$n" "$dir"
  pt "$tag" promptopt  512 256 3072 "$c256" "$n" "$dir"
  pt "$tag" judge     4096 512 0    $(( 128 / n )) "$n" "$dir"
  curl -fsS -m 5 http://127.0.0.1:8000/metrics 2>/dev/null | grep -E "spec_decode.*(accepted|draft)" | head -3 | sed 's/^/    spec: /'
  kill_all
}

drop(){ # dir
  [ "${KEEP:-0}" = 1 ] && return
  python3 -c "import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=True)" "$MD/$1"
  log "  removed $1 ($(df -h /workspace | awk 'NR==2{print $4}') free)"
}

log "===== ROSTER ROTATION: models the audit found missing, priority order ====="
# Space first. By the time this runs, fleet2 has measured everything below; the box is 1.2 TB
# and the rotation needs up to 260 GB per model. Keep the flagship (GLM-5.3-Flash), the two
# Qwen3.8-27B builds used as controls, and every drafter (tiny). Everything else re-downloads
# in 10-40 minutes if a re-run is ever wanted.
for m in MiniMax-M3-MXFP4 Qwen3.8-Flash-Next-NVFP4 Inkling-Small-NVFP4 gpt-oss-120b gemma-4-31B-it \
         gemma-4-26B-A4B-it Muse-Glimmer-30B Qwen3.6-35B-A3B-FP8 Nemotron-3.5-Lightning-30B gpt-oss-20b \
         Qwen27B-NVFP4-RedHat Qwen27B-NVFP4-unsloth; do
  if grep -q "$m\|${m%%-*}" /workspace/results/summary_full.tsv 2>/dev/null || [ -d "$MD/$m" ]; then
    [ "${KEEP:-0}" = 1 ] || { python3 -c "import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=True)" "$MD/$m"; log "  freed $m"; }
  fi
done
log "  disk after cleanup: $(df -h /workspace | awk 'NR==2{print $4" free"}')"

# 1. Hy3 (Tencent, AA 42, 295B-A21B). RedHat W4A4 experts + FP8 attention, MTP head kept. Lowest engine risk.
if fetch RedHatAI/Hy3-NVFP4-FP8 Hy3-NVFP4-FP8; then
  run hy3     "$MD/Hy3-NVFP4-FP8" 4 --kernel-config.linear_backend b12x --moe-backend flashinfer_cutlass
  run hy3_mtp "$MD/Hy3-NVFP4-FP8" 4 --kernel-config.linear_backend b12x --moe-backend flashinfer_cutlass \
      --speculative-config '{"method":"hy_v3_mtp","num_speculative_tokens":1}'
  drop Hy3-NVFP4-FP8
fi

# 2. Nemotron 3 Super 120B-A12B (AA 26; 84 GB -> one card, 4 replicas) + MTPv2 drafter.
if fetch nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 Nemotron-3-Super-NVFP4; then
  fetch nvidia/Nemotron-3-Super-120B-A12B-BF16-MTPv2 Nemotron-3-Super-MTPv2
  run nemo3s "$MD/Nemotron-3-Super-NVFP4" 1 --kernel-config.linear_backend b12x --moe-backend flashinfer_cutlass --mamba-ssm-cache-dtype float16
  [ -f "$MD/Nemotron-3-Super-MTPv2/config.json" ] && \
  run nemo3s_mtp "$MD/Nemotron-3-Super-NVFP4" 1 --kernel-config.linear_backend b12x --moe-backend flashinfer_cutlass --mamba-ssm-cache-dtype float16 \
      --speculative-config "{\"method\":\"nemotron_h_mtp\",\"model\":\"$MD/Nemotron-3-Super-MTPv2\",\"num_speculative_tokens\":3}"
  drop Nemotron-3-Super-NVFP4; drop Nemotron-3-Super-MTPv2
fi

# 3. Ling-3.0-flash (Ant, AA 38, 5.1B active; 81 GB -> one card). Known gibberish bug #53968 on FP8: the tripwire is the test.
if fetch olka-fi/Ling-3.0-flash-NVFP4 Ling-3.0-flash-NVFP4; then
  run ling3f "$MD/Ling-3.0-flash-NVFP4" 1 --kernel-config.linear_backend b12x --moe-backend flashinfer_cutlass
  run ling3f_mtp "$MD/Ling-3.0-flash-NVFP4" 1 --kernel-config.linear_backend b12x --moe-backend flashinfer_cutlass \
      --speculative-config '{"method":"bailing_hybrid_v3_mtp","num_speculative_tokens":1}'
  drop Ling-3.0-flash-NVFP4
fi

# 4. Laguna-S-2.1 (Poolside agentic coder, 100 GB -> 2x TP2) + official DFlash drafter.
if fetch poolside/Laguna-S-2.1-NVFP4 Laguna-S-2.1-NVFP4; then
  fetch poolside/Laguna-S-2.1-DFlash-NVFP4 Laguna-S-2.1-DFlash
  run laguna "$MD/Laguna-S-2.1-NVFP4" 2 --moe-backend flashinfer_cutlass --max-model-len 65536
  [ -f "$MD/Laguna-S-2.1-DFlash/config.json" ] && \
  run laguna_dflash "$MD/Laguna-S-2.1-NVFP4" 2 --moe-backend flashinfer_cutlass --max-model-len 65536 \
      --speculative-config "{\"method\":\"dflash\",\"model\":\"$MD/Laguna-S-2.1-DFlash\",\"num_speculative_tokens\":7}"
  drop Laguna-S-2.1-NVFP4; drop Laguna-S-2.1-DFlash
fi

# 5. Mistral-Medium-3.5-128B (AA 30, the only dense ~128B open control; 99 GB -> 2x TP2) + official EAGLE.
if fetch nvidia/Mistral-Medium-3.5-128B-NVFP4 Mistral-Medium-3.5-NVFP4; then
  fetch mistralai/Mistral-Medium-3.5-128B-EAGLE Mistral-Medium-3.5-EAGLE
  run mistralm "$MD/Mistral-Medium-3.5-NVFP4" 2 --kernel-config.linear_backend b12x --tokenizer-mode mistral --config-format mistral --load-format mistral
  [ -f "$MD/Mistral-Medium-3.5-EAGLE/config.json" ] && \
  run mistralm_eagle "$MD/Mistral-Medium-3.5-NVFP4" 2 --kernel-config.linear_backend b12x --tokenizer-mode mistral --config-format mistral --load-format mistral \
      --speculative-config "{\"method\":\"eagle\",\"model\":\"$MD/Mistral-Medium-3.5-EAGLE\",\"num_speculative_tokens\":3}"
  drop Mistral-Medium-3.5-NVFP4; drop Mistral-Medium-3.5-EAGLE
fi

# 6. Step-3.7-Flash (StepFun, AA 31, 11B active, 3-layer MTP; 125 GB -> TP4).
if fetch stepfun-ai/Step-3.7-Flash-NVFP4 Step-3.7-Flash-NVFP4; then
  run step37 "$MD/Step-3.7-Flash-NVFP4" 4 --enable-expert-parallel --moe-backend flashinfer_cutlass
  run step37_mtp "$MD/Step-3.7-Flash-NVFP4" 4 --enable-expert-parallel --moe-backend flashinfer_cutlass \
      --speculative-config '{"method":"step3p5_mtp","num_speculative_tokens":3}'
  drop Step-3.7-Flash-NVFP4
fi

# 7. MiMo-V2.5 0703 (Xiaomi, AA 38, 15B active; 184 GB -> TP4), the one candidate already validated on 4x RTX PRO 6000.
if fetch mitomtuna/MiMo-V2.5-0703-NVFP4 MiMo-V2.5-NVFP4; then
  run mimo25 "$MD/MiMo-V2.5-NVFP4" 4 --kernel-config.linear_backend b12x --moe-backend flashinfer_cutlass
  [ -d "$MD/MiMo-V2.5-NVFP4/dflash" ] && \
  run mimo25_dflash "$MD/MiMo-V2.5-NVFP4" 4 --kernel-config.linear_backend b12x --moe-backend flashinfer_cutlass \
      --speculative-config "{\"method\":\"dflash\",\"model\":\"$MD/MiMo-V2.5-NVFP4/dflash\",\"num_speculative_tokens\":7}"
  drop MiMo-V2.5-NVFP4
fi

# 8. Ornith-1.5-397B (vendor claims above GLM-5.3-Flash, no third-party evals; 251 GB -> TP4). Known sm_120 CUDA-graph SIGSEGV #54331.
if fetch littlecedar/Ornith-1.5-397B-NVFP4-MTP-Graft Ornith-1.5-397B-NVFP4; then
  run ornith "$MD/Ornith-1.5-397B-NVFP4" 4 --limit-mm-per-prompt '{"image":0}' --gpu-memory-utilization 0.90 \
    || run ornith_eager "$MD/Ornith-1.5-397B-NVFP4" 4 --limit-mm-per-prompt '{"image":0}' --gpu-memory-utilization 0.90 --enforce-eager
  run ornith_mtp "$MD/Ornith-1.5-397B-NVFP4" 4 --limit-mm-per-prompt '{"image":0}' --gpu-memory-utilization 0.90 \
      --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}'
  drop Ornith-1.5-397B-NVFP4
fi

# 9. Motif-3 (AA 47, 186.9 GB). Only the vendor's vLLM fork loads it; motif_vllm.sh runs that fork from the
#    image tree lifted into /workspace/motifimg. Skipped cleanly if the image pull did not complete.
if [ -d /workspace/motifimg/usr ] && fetch Motif-Technologies/Motif-3-NVFP4 Motif-3-NVFP4; then
  kill_all
  MD="$MD/Motif-3-NVFP4" bash /workspace/bench/motif_vllm.sh 2>&1 | tee -a "$R/motif.log"
  drop Motif-3-NVFP4
else
  log "SKIP motif3 (vendor image tree or weights absent)"
fi

log "ROSTER3 DONE"
kill_all
