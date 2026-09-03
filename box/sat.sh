#!/usr/bin/env bash
# R1 SATURATION LADDER. Our previous "C256" was 64 requests/engine with 64 prompts = ONE WAVE.
# This raises seats (fp8 KV + util 0.96 + seqs 512) and runs num_prompts = 8x concurrency
# so every point measures steady state, not a single burst.
B=/workspace/bench; R=/workspace/results; P=$R/probe; S=$R/smoke; MD=/workspace/models
mkdir -p $P $S
log(){ echo "[$(date +%H:%M:%S)] $*"; }
kill_all(){ tmux kill-session -t =srv 2>/dev/null
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do kill -9 "$pid" 2>/dev/null; done
  sleep 8; }

launch(){ # $1 = tag, rest = extra args
  local tag="$1"; shift
  kill_all
  cat > $B/l_sat.sh <<L
#!/usr/bin/env bash
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_OFFLINE=1
mkdir -p /workspace/results/smoke
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=\$i vllm serve /workspace/models/gpt-oss-120b --served-model-name gptoss \
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
  chmod +x $B/l_sat.sh
  log "launch $tag :: $*"
  tmux new-session -d -s srv "bash $B/l_sat.sh"
  local t=0 ok=0
  while [ $t -lt 720 ]; do
    ok=1; for p in 8000 8001 8002 8003; do curl -fsS -m 3 "http://127.0.0.1:$p/health" >/dev/null 2>&1 || ok=0; done
    [ "$ok" = 1 ] && break; sleep 10; t=$((t+10))
  done
  [ "$ok" = 1 ] || { log "$tag FAILED"; grep -iE "error|invalid|not supported" $S/${tag}_p8000.log 2>/dev/null | grep -vE "import_utils|deep_ep" | head -3 | cut -c1-200; return 1; }
  log "$tag healthy in ${t}s"
  grep -m1 "Mxfp4 MoE backend" $S/${tag}_p8000.log | sed 's/.*\] /  kernel: /' | cut -c1-90
  grep -m1 "GPU KV cache size" $S/${tag}_p8000.log | sed 's/.*\] /  /' | cut -c1-110
  return 0
}

point(){ # tag label in out prefix c_per_port
  local tag=$1 label=$2 in=$3 out=$4 pre=$5 c=$6
  local np=$(( c * 8 )) tot=$(( c * 4 ))
  mkdir -p $P/$tag
  for i in 0 1 2 3; do
    p=$((8000+i))
    vllm bench serve --backend openai --base-url "http://127.0.0.1:$p" --endpoint /v1/completions \
      --model gptoss --tokenizer $MD/gpt-oss-120b --trust-remote-code \
      --dataset-name random --random-input-len "$in" --random-output-len "$out" \
      --random-prefix-len "$pre" --random-range-ratio 0 \
      --request-rate inf --max-concurrency "$c" --num-prompts "$np" --ignore-eos --seed $((9000+c+in)) \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 --disable-tqdm \
      --save-result --result-dir $P/$tag --result-filename "${tag}__${label}__c${tot}__p${p}.json" \
      > $P/$tag/${label}_c${tot}_p${p}.log 2>&1 &
  done
  wait
  python3 - "$P/$tag" "${tag}__${label}__c${tot}__p" "$label" "$tot" "$np" <<'PY'
import glob, json, sys
d, pref, label, tot, np_ = sys.argv[1:]
fs = sorted(glob.glob(f"{d}/{pref}*.json"))
o=i=r=0.0; dur=0.0; ttft=0.0; tpot=0.0; comp=0
for f in fs:
    try: j=json.load(open(f))
    except Exception: continue
    c=j.get("completed",0) or 0; du=j.get("duration") or 1
    o+=j.get("output_throughput",0) or 0; r+=j.get("request_throughput",0) or 0
    i+=(j.get("total_input_tokens",0) or 0)/du; dur=max(dur,du); comp+=c
    ttft+=(j.get("mean_ttft_ms",0) or 0)*c; tpot+=(j.get("mean_tpot_ms",0) or 0)*c
if comp:
    print(f"  {label:10s} C{tot:<4} np{np_:<5} {dur:6.1f}s  {r*60:7.0f} req/min  {o:7.0f} out tok/s  {i:8.0f} in tok/s  ttft {ttft/comp:7.0f} ms  tpot {tpot/comp:5.1f} ms")
else:
    print(f"  {label:10s} C{tot:<4} FAILED ({len(fs)} files)")
PY
}

TAG=sat_marlin
if launch $TAG --moe-backend marlin; then
  log "SATURATION LADDER (np = 8x concurrency, so these are steady-state numbers)"
  point $TAG router    1024 128  0    64
  point $TAG router    1024 128  0   128
  point $TAG router    1024 128  0   256
  point $TAG short      256  64  0   128
  point $TAG short      256  64  0   256
  point $TAG promptopt  512 256 3072 128
  point $TAG promptopt  512 256 3072 256
  point $TAG judge     4096 512  0    64
  point $TAG judge     4096 512  0   128
  point $TAG rollout   8192 2048 0    32
  point $TAG rollout   8192 2048 0    64
  python3 $B/quality20.py gptoss http://127.0.0.1:8000 $P/${TAG}_quality20.json 2>&1 | tail -1
fi
log "SATURATION DONE"
kill_all
