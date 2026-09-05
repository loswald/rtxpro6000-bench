#!/usr/bin/env bash
# GLM-5.3-Flash, the layouts that remove the per-layer all-reduce, at the sequence budgets the vendor build's
# linear-attention cache allows: with TP1 each rank holds the whole Mamba state, so "max_num_seqs (512) exceeds
# available Mamba cache blocks (192)" - DP4 + EP runs at 192 sequences per rank (768 in flight across the box),
# DP2 x TP2 + EP at 384 per rank. Two shapes each, same vendor build and sm_120 port as glm_perf.sh.
set -u
B=/workspace/bench; R=/workspace/results; P=$R/probe; S=$R/smoke
MD=${MD:-/workspace/models/GLM-5.3-Flash-NVFP4}
TOK=/workspace/models/glm53f_tok
mkdir -p "$P" "$S" /workspace/glmvllm
log(){ echo "[$(date +%H:%M:%S)] $*"; }
source "$B/hardkill.sh"
CLEAN="env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH"
VEND=/workspace/glmimg/usr/local/lib/python3.12/dist-packages/vllm
[ -e /workspace/glmvllm/vllm ] || ln -s "$VEND" /workspace/glmvllm/vllm
ulimit -n 65536 2>/dev/null || ulimit -n 8192 2>/dev/null

launch(){ # tag seqs [extra...]
  local tag="$1" seqs="$2"; shift 2
  kill_all
  cat > "$B/l_gp2.sh" <<L
#!/usr/bin/env bash
ulimit -n 65536 2>/dev/null || ulimit -n 8192 2>/dev/null
export PYTHONPATH=/workspace/glmvllm
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_OFFLINE=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600 MAX_JOBS=6 NVCC_THREADS=2
exec python3 -m vllm.entrypoints.openai.api_server \\
  --model $MD --served-model-name m --host 0.0.0.0 --port 8000 \\
  --attention-backend FLASHINFER_MLA_SPARSE_SM90 \\
  --kv-cache-dtype auto --block-size 1024 --max-model-len 40960 --max-num-seqs $seqs \\
  --max-num-batched-tokens 16384 --gpu-memory-utilization 0.90 \\
  --enable-prefix-caching --trust-remote-code --disable-custom-all-reduce \\
  --no-enable-flashinfer-autotune --disable-uvicorn-access-log $*
L
  chmod +x "$B/l_gp2.sh"
  log "  launch $tag: --max-num-seqs $seqs $*"
  tmux new-session -d -s srv "bash $B/l_gp2.sh > $S/${tag}.log 2>&1; echo EXIT=\$? >> $S/${tag}.log"
  local t=0 ok=0
  while [ "$t" -lt 1800 ]; do
    curl -fsS -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    grep -q "^EXIT=" "$S/${tag}.log" 2>/dev/null && break
    tmux has-session -t =srv 2>/dev/null || break
    sleep 15; t=$((t+15))
  done
  [ "$ok" = 1 ] || { log "  $tag FAILED after ${t}s: $(grep -ohE "Worker failed with error '[^']{0,160}|ValueError: [^\"]{0,140}|error: argument [^\"]{0,120}|CUDA out of memory[^\"]{0,40}" "$S/${tag}.log" | sort -u | head -2 | paste -sd'|')"; kill_all; return 1; }
  log "  $tag healthy in ${t}s | $(grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$S/${tag}.log")"
}
pt(){ # tag label in out prefix conc
  local tag=$1 label=$2 in=$3 out=$4 pre=$5 c=$6
  mkdir -p "$P/$tag"
  $CLEAN vllm bench serve --backend openai --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
    --model m --tokenizer "$TOK" --trust-remote-code \
    --dataset-name random --random-input-len "$in" --random-output-len "$out" \
    --random-prefix-len "$pre" --random-range-ratio 0 \
    --request-rate inf --max-concurrency "$c" --num-prompts $((c*4)) --ignore-eos --seed $((9300+c+in)) \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --disable-tqdm \
    --save-result --result-dir "$P/$tag" --result-filename "${tag}__${label}__c${c}__p8000.json" \
    > "$P/$tag/${label}_c${c}.log" 2>&1
  $CLEAN python3 "$B/agg.py" "$P/$tag" "${tag}__${label}__c${c}__p" "$label" "$c" "$tag" 2>&1 | tail -1
}
arm(){ # tag seqs [extra...]
  local tag="$1" seqs="$2"; shift 2
  if launch "$tag" "$seqs" "$@"; then
    pt "$tag" router 1024 128 0 1024
    pt "$tag" promptopt 512 256 3072 1024
    $CLEAN python3 "$B/quality20.py" m http://127.0.0.1:8000 "$P/${tag}_quality20.json" --mode chat --max-tokens 1024 2>&1 | tail -1
  fi
}
log "===== GLM-5.3-Flash: the MoE kernel, then the layouts without the per-layer all-reduce ====="
# The vendor build's default ("auto") routes this checkpoint's experts through a TRITON FP8 MoE kernel. Every other
# backend it lists is tried at TP4 / 512 sequences; one that this model rejects fails in a minute and the sweep moves on.
# DP layouts first: DeepSeek's DP4+EP arm on the other box just posted +48% over TP4 at the same shape, and
# they are the arms the step-time ceiling analysis says can move. The MoE-kernel arms follow, each guarded so
# that the 403-item quality run on the fastest layout keeps at least 105 minutes before the 22:00 UTC deadline.
CUT=$(date -d "2026-09-05 20:15:00" +%s)
guard(){ [ "$(date +%s)" -lt "$CUT" ] || { log "  skip $1: past 20:15 UTC, the quality run on the fastest layout takes priority"; return 1; }; }
arm glm53f_dp4ep4_s192     192 --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel
arm glm53f_dp2tp2ep2_s384  384 --tensor-parallel-size 2 --data-parallel-size 2 --enable-expert-parallel
guard moetrtllm   && arm glm53f_s512_moetrtllm     512 --tensor-parallel-size 4 --moe-backend flashinfer_trtllm
guard moecutlass  && arm glm53f_s512_moecutlass    512 --tensor-parallel-size 4 --moe-backend cutlass
guard moeficutlass && arm glm53f_s512_moeficutlass 512 --tensor-parallel-size 4 --moe-backend flashinfer_cutlass
guard moemarlin   && arm glm53f_s512_moemarlin     512 --tensor-parallel-size 4 --moe-backend marlin
log "GLMPERF2 DONE"
kill_all
