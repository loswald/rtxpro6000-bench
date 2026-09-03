#!/usr/bin/env bash
# Campaign 4: the interconnect A/B, with the 0.28.0 flag spellings. Budget-aware.
R=/workspace/results; P=$R/probe; S=$R/smoke; B=/workspace/bench; MD=/workspace/models
mkdir -p $P $S
log(){ echo "[$(date +%H:%M:%S)] $*"; }
kill_all(){ tmux kill-session -t =probe 2>/dev/null; tmux kill-session -t =srv 2>/dev/null
  for pid in $(pgrep -f "vllm serv[e]"); do kill "$pid" 2>/dev/null; done; sleep 8
  for pid in $(pgrep -f "vllm serv[e]"); do kill -9 "$pid" 2>/dev/null; done; sleep 3; }
serve(){ # tag launcher ports limit
  local tag="$1" launcher="$2" ports="$3" limit="$4" t=0 ok
  kill_all; log "launch $tag"
  tmux new-session -d -s srv "bash $launcher > $S/$tag.log 2>&1; echo EXIT=\$? >> $S/$tag.log"
  while [ "$t" -lt "$limit" ]; do
    ok=1; for p in $ports; do curl -fsS -m 3 "http://127.0.0.1:$p/health" >/dev/null 2>&1 || ok=0; done
    [ "$ok" = 1 ] && { log "$tag healthy in ${t}s"; return 0; }
    if grep -q "^EXIT=" "$S/$tag.log" 2>/dev/null; then
      log "$tag DIED after ${t}s"; grep -iE "error|invalid choice|not supported|Traceback" "$S/$tag.log" | grep -vE "import_utils|deep_ep" | tail -3 | cut -c1-200; return 1; fi
    sleep 10; t=$((t+10))
  done
  log "$tag TIMED OUT after ${limit}s"; return 1; }

cat > $B/l_ds4_tp4.sh <<'L1'
#!/usr/bin/env bash
export VLLM_USE_DEEP_GEMM=0 FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0
export NCCL_IB_DISABLE=1 NCCL_MIN_NCHANNELS=8 NCCL_DEBUG=WARN
export VLLM_DSV4_OPROJ_SM120_FALLBACK=1 CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1
exec vllm serve /workspace/models/DeepSeek-V4-Flash-0731 --served-model-name ds4flash --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 --tokenizer-mode deepseek_v4 --block-size 256 --moe-backend flashinfer_b12x \
  --attention_config.use_fp4_indexer_cache False --kv-cache-dtype fp8 \
  --max-model-len 40960 --max-num-seqs 256 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.92 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --no-enable-flashinfer-autotune --kernel-config.linear_backend b12x --disable-custom-all-reduce \
  --enable-prefix-caching --trust-remote-code
L1
cat > $B/l_ds4_tp4_auto.sh <<'L2'
#!/usr/bin/env bash
export VLLM_USE_DEEP_GEMM=0 FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0
export NCCL_IB_DISABLE=1 NCCL_MIN_NCHANNELS=8 NCCL_DEBUG=WARN
export VLLM_DSV4_OPROJ_SM120_FALLBACK=1 CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1
exec vllm serve /workspace/models/DeepSeek-V4-Flash-0731 --served-model-name ds4flash --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 --enable-expert-parallel --tokenizer-mode deepseek_v4 --block-size 256 --moe-backend auto \
  --attention_config.use_fp4_indexer_cache False --kv-cache-dtype fp8 \
  --max-model-len 40960 --max-num-seqs 256 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.92 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --no-enable-flashinfer-autotune --kernel-config.linear_backend b12x --disable-custom-all-reduce \
  --enable-prefix-caching --trust-remote-code
L2
chmod +x $B/l_ds4_*.sh

log "B: DeepSeek TP4 (flashinfer_b12x MoE) -- the interconnect A/B"
if serve gen5_ds4_tp4 $B/l_ds4_tp4.sh 8000 900; then
  python3 $B/quality20.py ds4flash http://127.0.0.1:8000 $P/gen5_ds4_tp4_quality20.json
  bash $B/probe_v2.sh gen5_ds4_tp4 ds4flash $MD/DeepSeek-V4-Flash-0731 deepseek_v4 full > $P/gen5_ds4_tp4.log 2>&1
else
  log "falling back to TP4+EP4 with auto MoE"
  if serve gen5_ds4_tp4ep4 $B/l_ds4_tp4_auto.sh 8000 900; then
    python3 $B/quality20.py ds4flash http://127.0.0.1:8000 $P/gen5_ds4_tp4ep4_quality20.json
    bash $B/probe_v2.sh gen5_ds4_tp4ep4 ds4flash $MD/DeepSeek-V4-Flash-0731 deepseek_v4 full > $P/gen5_ds4_tp4ep4.log 2>&1
  fi
fi
log "C: SGLang A/B on gpt-oss x4"
kill_all
if grep -q "SGLANG-INSTALL-DONE" /workspace/sglang_setup.log 2>/dev/null; then
  bash $B/sglang_ab.sh >> $R/sglang_ab.log 2>&1
else
  log "sglang install incomplete, skipping"; tail -3 /workspace/sglang_setup.log | cut -c1-140
fi
log "CAMPAIGN4 DONE"
kill_all
