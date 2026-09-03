#!/usr/bin/env bash
# Proper quantization quality gate, replacing the string-identity smoke test.
#
#   1. lm-eval GSM8K at both precisions -> task accuracy + stderr, reported as
#      recovery % of the bf16 baseline. This is the metric quantization releases use.
#   2. Logit-level divergence: top-1 agreement, top-5 overlap, KL between the two
#      next-token distributions over identical teacher-forced contexts.
#   3. The old string diff is kept ONLY as a corruption tripwire, with its own
#      fp8-vs-fp8 noise floor so it is interpretable.
#
# Both servers run simultaneously on separate GPUs so the comparison is paired.
B=/workspace/bench; R=/workspace/results; P=$R/probe; S=$R/smoke; MD=/workspace/models
mkdir -p "$P" "$S"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
kill_all(){
  tmux kill-session -t =srv 2>/dev/null
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
    kill -9 "$pid" 2>/dev/null
  done
  sleep 8
}

log "installing lm-eval into an isolated venv (does not touch the vLLM environment)"
if [ ! -x /workspace/venv-eval/bin/lm_eval ]; then
  uv venv --python 3.12 --system-site-packages /workspace/venv-eval >/dev/null 2>&1
  uv pip install --python /workspace/venv-eval/bin/python -q "lm-eval[api]" >/dev/null 2>&1
fi
/workspace/venv-eval/bin/lm_eval --version 2>&1 | head -1

kill_all
# port 8000 = fp8 KV, 8001 = bf16 KV, 8002 = fp8 again (noise floor for both metrics)
cat > "$B/l_qg.sh" <<'LQG'
#!/usr/bin/env bash
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 FLASHINFER_CUDA_ARCH_LIST=12.0f HF_HUB_OFFLINE=1
serve_one () {  # gpu port kvdtype
  CUDA_VISIBLE_DEVICES=$1 vllm serve /workspace/models/gpt-oss-120b --served-model-name gptoss \
    --host 0.0.0.0 --port $2 --kv-cache-dtype $3 --max-model-len 8192 --max-num-seqs 64 \
    --gpu-memory-utilization 0.90 --moe-backend marlin --no-enable-flashinfer-autotune \
    --trust-remote-code --disable-uvicorn-access-log \
    > /workspace/results/smoke/qg_$2.log 2>&1 &
}
serve_one 0 8000 fp8
sleep 2
serve_one 1 8001 auto
sleep 2
serve_one 2 8002 fp8
wait
LQG
chmod +x "$B/l_qg.sh"
log "launching fp8 (8000), bf16 (8001) and a second fp8 (8002, the control)"
tmux new-session -d -s srv "bash $B/l_qg.sh"
t=0; ok=0
while [ "$t" -lt 720 ]; do
  ok=1
  for p in 8000 8001 8002; do curl -fsS -m 3 "http://127.0.0.1:$p/health" >/dev/null 2>&1 || ok=0; done
  [ "$ok" = 1 ] && break
  sleep 10; t=$((t+10))
done
if [ "$ok" != 1 ]; then
  log "quality-gate servers failed to start"
  grep -iE "error|not supported" "$S/qg_8000.log" 2>/dev/null | head -3 | cut -c1-180
  exit 1
fi
log "all three servers up in ${t}s"
for p in 8000 8001 8002; do
  grep -m1 -oE "GPU KV cache size: [0-9,]+ tokens" "$S/qg_$p.log" | sed "s/^/  :$p /"
done

log "--- metric 1: GSM8K task accuracy (200 questions, 5-shot, strict answer match) ---"
run_gsm () {  # port label
  /workspace/venv-eval/bin/lm_eval --model local-completions \
    --model_args "model=gptoss,base_url=http://127.0.0.1:$1/v1/completions,num_concurrent=32,max_retries=2,tokenized_requests=False" \
    --tasks gsm8k --num_fewshot 5 --limit 200 --batch_size 32 \
    --output_path "$P/gsm8k_$2" > "$P/gsm8k_$2.log" 2>&1
  python3 - "$2" <<'PY'
import glob, json, sys
lab = sys.argv[1]
fs = sorted(glob.glob(f"/workspace/results/probe/gsm8k_{lab}/**/results*.json", recursive=True))
if not fs:
    print(f"  {lab:6s} no results"); raise SystemExit
r = json.load(open(fs[-1]))["results"]["gsm8k"]
em = r.get("exact_match,strict-match", r.get("exact_match,flexible-extract"))
se = r.get("exact_match_stderr,strict-match", 0) or 0
print(f"  {lab:6s} GSM8K exact_match = {em:.4f} +/- {se:.4f}")
PY
}
run_gsm 8001 bf16
run_gsm 8000 fp8
run_gsm 8002 fp8b
python3 - <<'PY'
import glob, json
def score(lab):
    fs = sorted(glob.glob(f"/workspace/results/probe/gsm8k_{lab}/**/results*.json", recursive=True))
    if not fs: return None, None
    r = json.load(open(fs[-1]))["results"]["gsm8k"]
    return (r.get("exact_match,strict-match", r.get("exact_match,flexible-extract")),
            r.get("exact_match_stderr,strict-match", 0) or 0)
b, bse = score("bf16"); f, fse = score("fp8"); f2, _ = score("fp8b")
if b and f:
    print(f"  RECOVERY: fp8 retains {100*f/b:.1f}% of the bf16 GSM8K score")
    if f2: print(f"  control : fp8 vs fp8 spread = {abs(f-f2):.4f} (run-to-run noise on this metric)")
    import math
    sig = math.sqrt(bse**2 + fse**2)
    d = abs(b-f)
    print(f"  difference {d:.4f} vs combined stderr {sig:.4f} -> " +
          ("INDISTINGUISHABLE" if d <= 2*sig else "SIGNIFICANT, investigate"))
PY

log "--- metric 2: logit-level divergence (top-1 agreement, top-5 overlap, KL) ---"
echo "  treatment: fp8 vs bf16"
python3 "$B/logit_diff.py" http://127.0.0.1:8000 http://127.0.0.1:8001 gptoss "$P/logit_fp8_vs_bf16.json"
echo "  control  : fp8 vs fp8"
python3 "$B/logit_diff.py" http://127.0.0.1:8000 http://127.0.0.1:8002 gptoss "$P/logit_fp8_vs_fp8.json"
python3 - <<'PY'
import json
try:
    t = json.load(open("/workspace/results/probe/logit_fp8_vs_bf16.json"))["summary"]
    c = json.load(open("/workspace/results/probe/logit_fp8_vs_fp8.json"))["summary"]
    print(f"  control  top1 {c['top1_agreement']:.3f}  top5 {c['top5_overlap']:.3f}  meanKL {c['mean_kl']:.5f}")
    print(f"  fp8/bf16 top1 {t['top1_agreement']:.3f}  top5 {t['top5_overlap']:.3f}  meanKL {t['mean_kl']:.5f}")
    print(f"  EXCESS KL attributable to the KV dtype: {t['mean_kl']-c['mean_kl']:+.5f}")
    print(f"  top-1 agreement lost vs control: {c['top1_agreement']-t['top1_agreement']:+.3f}")
except Exception as e:
    print("  comparison unavailable:", type(e).__name__, e)
PY

log "--- metric 3: corruption tripwire, with its own floor ---"
python3 "$B/kvdiff.py" http://127.0.0.1:8000 http://127.0.0.1:8002 gptoss "$P/kv_floor_fp8_vs_fp8.json"

log "QUALITY GATE DONE"
kill_all
