#!/usr/bin/env bash
# Round 3: everything re-tested AT SATURATION, where the knobs can actually bind.
# Uses the same steady-state client as sat.sh (num_prompts = 8x concurrency).
B=/workspace/bench; R=/workspace/results; P=$R/probe; S=$R/smoke; MD=/workspace/models
mkdir -p $P $S
log(){ echo "[$(date +%H:%M:%S)] $*"; }
kill_all(){ tmux kill-session -t =srv 2>/dev/null
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do kill -9 "$pid" 2>/dev/null; done
  sleep 8; }

launch_x4(){ # tag, model, alias, then extra args
  local tag="$1" model="$2" alias="$3"; shift 3
  kill_all
  cat > $B/l_r3.sh <<L
#!/usr/bin/env bash
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_OFFLINE=1
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=\$i vllm serve $model --served-model-name $alias \
    --host 0.0.0.0 --port \$((8000+i)) \
    --kv-cache-dtype fp8 --max-model-len 40960 --max-num-seqs 512 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.96 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --no-enable-flashinfer-autotune --enable-prefix-caching --trust-remote-code \
    --disable-uvicorn-access-log $* \
    > /workspace/results/smoke/${tag}_p\$((8000+i)).log 2>&1 &
  sleep 2
done
wait
L
  chmod +x $B/l_r3.sh
  log "launch $tag :: $*"
  tmux new-session -d -s srv "bash $B/l_r3.sh"
  local t=0 ok=0
  while [ $t -lt 720 ]; do
    ok=1; for p in 8000 8001 8002 8003; do curl -fsS -m 3 "http://127.0.0.1:$p/health" >/dev/null 2>&1 || ok=0; done
    [ "$ok" = 1 ] && break; sleep 10; t=$((t+10))
  done
  [ "$ok" = 1 ] || { log "$tag FAILED"; grep -iE "error|invalid|does not support" $S/${tag}_p8000.log 2>/dev/null | grep -vE "import_utils|deep_ep" | head -2 | cut -c1-200; return 1; }
  log "$tag healthy ${t}s | $(grep -m1 -oE "Using '[A-Z0-9_]+' Mxfp4 MoE backend" $S/${tag}_p8000.log) | $(grep -m1 -oE "GPU KV cache size: [0-9,]+ tokens" $S/${tag}_p8000.log)"
  return 0
}

point(){ # tag label in out prefix c_per_port alias model
  local tag=$1 label=$2 in=$3 out=$4 pre=$5 c=$6 alias=${7:-gptoss} model=${8:-$MD/gpt-oss-120b}
  local np=$(( c * 8 )) tot=$(( c * 4 ))
  mkdir -p $P/$tag
  for i in 0 1 2 3; do
    p=$((8000+i))
    vllm bench serve --backend openai --base-url "http://127.0.0.1:$p" --endpoint /v1/completions \
      --model "$alias" --tokenizer "$model" --trust-remote-code \
      --dataset-name random --random-input-len "$in" --random-output-len "$out" \
      --random-prefix-len "$pre" --random-range-ratio 0 \
      --request-rate inf --max-concurrency "$c" --num-prompts "$np" --ignore-eos --seed $((9000+c+in)) \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 --disable-tqdm \
      --save-result --result-dir $P/$tag --result-filename "${tag}__${label}__c${tot}__p${p}.json" \
      > $P/$tag/${label}_c${tot}_p${p}.log 2>&1 &
  done
  wait
  python3 - "$P/$tag" "${tag}__${label}__c${tot}__p" "$label" "$tot" <<'PY'
import glob, json, sys
d, pref, label, tot = sys.argv[1:]
fs = sorted(glob.glob(f"{d}/{pref}*.json"))
o=i=r=0.0; dur=0.0; ttft=0.0; comp=0
for f in fs:
    try: j=json.load(open(f))
    except Exception: continue
    c=j.get("completed",0) or 0; du=j.get("duration") or 1
    o+=j.get("output_throughput",0) or 0; r+=j.get("request_throughput",0) or 0
    i+=(j.get("total_input_tokens",0) or 0)/du; dur=max(dur,du); comp+=c
    ttft+=(j.get("mean_ttft_ms",0) or 0)*c
print(f"  {label:10s} C{tot:<4} {dur:6.1f}s {r*60:7.0f} req/min {o:7.0f} out tok/s {i:8.0f} in tok/s ttft {ttft/max(comp,1):6.0f} ms" if comp else f"  {label} C{tot} FAILED")
PY
}

# --- the three arms that could still move gpt-oss, all at saturation ---
if launch_x4 r3_ficutlass_mxfp8 $MD/gpt-oss-120b gptoss --moe-backend flashinfer_cutlass --quantization-config.moe.activation mxfp8; then
  point r3_ficutlass_mxfp8 router 1024 128 0 256; point r3_ficutlass_mxfp8 judge 4096 512 0 128
fi
if launch_x4 r3_apiserver4 $MD/gpt-oss-120b gptoss --moe-backend marlin --api-server-count 4; then
  point r3_apiserver4 router 1024 128 0 256; point r3_apiserver4 short 256 64 0 256
fi
if launch_x4 r3_mnbt16k $MD/gpt-oss-120b gptoss --moe-backend marlin --max-num-batched-tokens 16384; then
  point r3_mnbt16k router 1024 128 0 256; point r3_mnbt16k judge 4096 512 0 128
fi
# --- control: re-run the baseline in-session so the deltas above are readable ---
if launch_x4 r3_control $MD/gpt-oss-120b gptoss --moe-backend marlin; then
  point r3_control router 1024 128 0 256; point r3_control judge 4096 512 0 128; point r3_control short 256 64 0 256
fi
# --- the other two models, at saturation, for a fair comparison ---
if launch_x4 r3_qwen27b $MD/Qwen3.8-27B-FP8 qwen27b --kernel-config.linear_backend b12x; then
  point r3_qwen27b router 1024 128 0 256 qwen27b $MD/Qwen3.8-27B-FP8
  point r3_qwen27b judge 4096 512 0 128 qwen27b $MD/Qwen3.8-27B-FP8
  python3 $B/quality20.py qwen27b http://127.0.0.1:8000 $P/r3_qwen27b_quality20.json 2>&1 | tail -1
fi
log "ROUND3 DONE"
kill_all
