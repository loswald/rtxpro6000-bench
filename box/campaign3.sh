#!/usr/bin/env bash
# Campaign 3 on the Gen5 / ACS-clean box: interconnect A/B against the old host, same configs.
R=/workspace/results; P=$R/probe; S=$R/smoke; B=/workspace/bench; MD=/workspace/models
mkdir -p $P $S
log(){ echo "[$(date +%H:%M:%S)] $*"; }
kill_all(){ tmux kill-session -t =probe 2>/dev/null; tmux kill-session -t =srv 2>/dev/null
  for pid in $(pgrep -f "vllm serv[e]"); do kill "$pid" 2>/dev/null; done; sleep 10
  for pid in $(pgrep -f "vllm serv[e]"); do kill -9 "$pid" 2>/dev/null; done; sleep 4; }
wait_health(){ local ports="$1" limit="$2" t=0 ok
  while [ "$t" -lt "$limit" ]; do ok=1
    for p in $ports; do curl -fsS -m 3 "http://127.0.0.1:$p/health" >/dev/null 2>&1 || ok=0; done
    [ "$ok" = 1 ] && return 0; sleep 10; t=$((t+10)); done; return 1; }
serve(){ local tag="$1" launcher="$2" ports="$3" limit="$4"
  kill_all; log "launch $tag"
  tmux new-session -d -s srv "bash $launcher > $S/$tag.log 2>&1; echo EXIT=\$? >> $S/$tag.log"
  if wait_health "$ports" "$limit"; then log "$tag healthy"; return 0; fi
  log "$tag FAILED"; grep -iE "error|not supported|invalid|unrecognized" "$S/$tag.log" 2>/dev/null | grep -vE "import_utils|deep_ep" | tail -3 | cut -c1-200; return 1; }

cat > $B/l_gptoss.sh <<'L1'
#!/usr/bin/env bash
exec bash /workspace/bench/launch_x4.sh /workspace/models/gpt-oss-120b gptoss
L1
cat > $B/l_qwen27b.sh <<'L2'
#!/usr/bin/env bash
exec bash /workspace/bench/launch_x4.sh /workspace/models/Qwen3.8-27B-FP8 qwen27b \
  --kv-cache-dtype fp8 --kernel-config.linear_backend b12x
L2
cat > $B/l_ds4_tp4.sh <<'L3'
#!/usr/bin/env bash
export VLLM_USE_DEEP_GEMM=0 FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0
export NCCL_IB_DISABLE=1 NCCL_MIN_NCHANNELS=8 NCCL_DEBUG=WARN
export VLLM_DSV4_OPROJ_SM120_FALLBACK=1 CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1
exec vllm serve /workspace/models/DeepSeek-V4-Flash-0731 --served-model-name ds4flash --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 --tokenizer-mode deepseek_v4 --block-size 256 --moe-backend b12x \
  --attention_config.use_fp4_indexer_cache False --kv-cache-dtype fp8 \
  --max-model-len 40960 --max-num-seqs 256 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.92 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --no-enable-flashinfer-autotune --kernel-config.linear_backend b12x --disable-custom-all-reduce \
  --enable-prefix-caching --trust-remote-code
L3
cat > $B/l_ds4_dp4.sh <<'L4'
#!/usr/bin/env bash
export VLLM_USE_DEEP_GEMM=0 FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0
export NCCL_IB_DISABLE=1 NCCL_MIN_NCHANNELS=8 NCCL_DEBUG=WARN
export VLLM_DSV4_OPROJ_SM120_FALLBACK=1 CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1
exec vllm serve /workspace/models/DeepSeek-V4-Flash-0731 --served-model-name ds4flash --host 0.0.0.0 --port 8000 \
  --data-parallel-size 4 --tensor-parallel-size 1 --enable-expert-parallel \
  --tokenizer-mode deepseek_v4 --block-size 256 --moe-backend auto \
  --attention_config.use_fp4_indexer_cache False --kv-cache-dtype fp8 \
  --max-model-len 40960 --max-num-seqs 256 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.92 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --no-enable-flashinfer-autotune --kernel-config.linear_backend b12x --disable-custom-all-reduce \
  --enable-prefix-caching --trust-remote-code
L4
chmod +x $B/l_*.sh

log "A: gpt-oss-120b x4 (card+host parity vs old box)"
if serve gen5_gptoss_x4 $B/l_gptoss.sh "8000 8001 8002 8003" 600; then
  python3 $B/quality20.py gptoss http://127.0.0.1:8000 $P/gen5_gptoss_x4_quality20.json
  bash $B/probe4.sh gen5_gptoss_x4 gptoss $MD/gpt-oss-120b auto full > $P/gen5_gptoss_x4.log 2>&1
fi
log "B: DeepSeek-V4-Flash TP4 (THE interconnect A/B)"
if serve gen5_ds4_tp4 $B/l_ds4_tp4.sh 8000 900; then
  python3 $B/quality20.py ds4flash http://127.0.0.1:8000 $P/gen5_ds4_tp4_quality20.json
  bash $B/probe_v2.sh gen5_ds4_tp4 ds4flash $MD/DeepSeek-V4-Flash-0731 deepseek_v4 full > $P/gen5_ds4_tp4.log 2>&1
fi
log "C: DeepSeek-V4-Flash DP4+EP4"
if serve gen5_ds4_dp4 $B/l_ds4_dp4.sh 8000 900; then
  bash $B/probe_v2.sh gen5_ds4_dp4 ds4flash $MD/DeepSeek-V4-Flash-0731 deepseek_v4 quick > $P/gen5_ds4_dp4.log 2>&1
fi
log "D: Qwen3.8-27B-FP8 x4"
if serve gen5_qwen27b_x4 $B/l_qwen27b.sh "8000 8001 8002 8003" 900; then
  bash $B/probe4.sh gen5_qwen27b_x4 qwen27b $MD/Qwen3.8-27B-FP8 auto quick > $P/gen5_qwen27b_x4.log 2>&1
fi
log "CAMPAIGN3 DONE"
kill_all
