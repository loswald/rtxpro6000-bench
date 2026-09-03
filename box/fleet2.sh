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

need_dl(){ # dir -> 0 once the download log says done/have for it (config.json lands before the safetensors do)
  local dn; dn=$(basename "$1") DL=/workspace/results/dl6000.log
  [ -f "$DL" ] || { [ -f "$1/config.json" ]; return; }   # no download log on this box: fall back to presence
  grep -qE "\] (done|have) $dn( |\$)" "$DL" && return 0
  log "  WAIT $dn (still downloading)"
  local w=0
  while ! grep -qE "\] (done|have) $dn( |\$)" "$DL" && [ $w -lt 300 ] && { tmux has-session -t =dl2 2>/dev/null || tmux has-session -t =dl 2>/dev/null; }; do sleep 60; w=$((w+1)); done
  grep -qE "\] (done|have) $dn( |\$)" "$DL"
}

rep(){ # alias dir [extra...]
  local alias=$1 dir=$2; shift 2
  need_dl "$dir" || { log "SKIP $alias (not downloaded)"; return; }
  log "########## $alias ($(du -sh "$dir" | cut -f1)) ##########"
  serve_x4 "f2_$alias" "$dir" "$@" && shapes "f2_$alias" "$dir" 4
}
tpm(){ # alias dir tp replicas [extra...]
  local alias=$1 dir=$2 tp=$3 reps=$4; shift 4
  need_dl "$dir" || { log "SKIP $alias (not downloaded)"; return; }
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


log "===== FLEET, drafter arms (speculation on models already measured) ====="
D=/workspace/models
# MiniMax-M3 + NVIDIA's DSpark drafter (native MTP is unusable: no checkpoint ships the MTP tensors).
if [ -f "$D/MiniMax-M3-DSpark/config.json" ]; then
  tpm minimaxm3_dspark "$MD/MiniMax-M3-MXFP4" 4 1 --moe-backend flashinfer_cutlass --quantization-config.moe.activation mxfp8 \
      --block-size 128 --speculative-config "{\"method\":\"dspark\",\"model\":\"$D/MiniMax-M3-DSpark\",\"num_speculative_tokens\":8}"
else log "SKIP minimaxm3_dspark (drafter absent)"; fi
# gemma-4-26B-A4B + Google's official MTP assistant (0.4 GB).
if [ -f "$D/gemma-4-26B-A4B-it-assistant/config.json" ]; then
  rep gemma26_mtp "$MD/gemma-4-26B-A4B-it" \
      --speculative-config "{\"method\":\"gemma4_mtp\",\"model\":\"$D/gemma-4-26B-A4B-it-assistant\",\"num_speculative_tokens\":3}"
else log "SKIP gemma26_mtp (assistant absent)"; fi
# Nemotron-3.5-Lightning: built-in MTP head (free), then NVIDIA's DSpark drafter.
rep nemo35_mtp "$MD/Nemotron-3.5-Lightning-30B" --kernel-config.linear_backend b12x \
    --speculative-config "{\"method\":\"nemotron_h_mtp\",\"num_speculative_tokens\":3}"
if [ -f "$D/Nemotron-3.5-Lightning-DSpark/config.json" ]; then
  rep nemo35_dspark "$MD/Nemotron-3.5-Lightning-30B" --kernel-config.linear_backend b12x \
      --speculative-config "{\"method\":\"dspark\",\"model\":\"$D/Nemotron-3.5-Lightning-DSpark\",\"num_speculative_tokens\":3}"
else log "SKIP nemo35_dspark (drafter absent)"; fi
# Muse-Glimmer-30B + Meta's official DFlash assistant.
if [ -f "$D/Muse-Glimmer-30B-assistant/config.json" ]; then
  rep muse30_dflash "$MD/Muse-Glimmer-30B" \
      --speculative-config "{\"method\":\"dflash\",\"model\":\"$D/Muse-Glimmer-30B-assistant\",\"num_speculative_tokens\":15}"
else log "SKIP muse30_dflash (assistant absent)"; fi
# Qwen3.8-27B NVFP4 (b12x) + the incoai DFlash2 block-diffusion drafter (vs DSpark measured in nvtier2).
if [ -f "$D/Qwen3.8-27B-DFlash2/config.json" ]; then
  rep q27_dflash2 "$MD/Qwen27B-NVFP4-RTX5090" --kernel-config.linear_backend b12x \
      --speculative-config "{\"method\":\"dflash\",\"model\":\"$D/Qwen3.8-27B-DFlash2\",\"num_speculative_tokens\":7}"
else log "SKIP q27_dflash2 (drafter absent)"; fi

log "===== FLEET, late additions ====="
# Inkling-Small (Thinking Machines, Apache-2.0, 266B MoE, multimodal incl. audio). vLLM 0.28.1 has
# native inkling_mm_model + inkling_mtp support. Runs only if the download finished in time.
if need_dl "$MD/Inkling-Small-NVFP4"; then
  tpm inkling     "$MD/Inkling-Small-NVFP4" 4 1 --kernel-config.linear_backend b12x
  tpm inkling_mtp "$MD/Inkling-Small-NVFP4" 4 1 --kernel-config.linear_backend b12x \
      --speculative-config '{"method":"inkling_mtp","num_speculative_tokens":2}'
else
  log "SKIP inkling (download not finished)"
fi
# GLM-5.3-Flash MTP arm. The first attempt lost its JSON quotes in the launcher; glm_vllm.sh now
# routes the config through SPEC, and ARMS=mtp skips the already-measured base arm.
kill_all
glm_ready(){ # weights downloaded and the vendor image lifted (img6000.log on a fresh box; presence otherwise)
  need_dl "$MD/GLM-5.3-Flash-NVFP4" || return 1
  if [ -f /workspace/results/img6000.log ]; then
    local w=0; while ! grep -q "done /workspace/glmimg" /workspace/results/img6000.log && ! grep -q "FAILED /workspace/glmimg" /workspace/results/img6000.log && [ $w -lt 180 ]; do sleep 60; w=$((w+1)); done
  fi
  [ -d /workspace/glmimg/usr ]
}
if glm_ready; then
  ARMS=${GLM_ARMS:-mtp} bash /workspace/bench/glm_vllm.sh >> /workspace/results/glm_vllm_full.log 2>&1
else log "SKIP glm (weights or vendor image absent)"; fi

log "FLEET2 DONE"
kill_all
