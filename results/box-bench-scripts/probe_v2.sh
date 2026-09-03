#!/usr/bin/env bash
# probe.sh <tag> <served_model_alias> <tokenizer_path> [tokenizer_mode] [set: full|quick|spec]
TAG=$1; MODEL=$2; TOK=$3; TOKMODE=${4:-auto}; SET=${5:-full}; PORT=${PORT:-8000}
OUT=/workspace/results/probe/$TAG; mkdir -p "$OUT"
TSV=$OUT/summary.tsv
[ -f "$TSV" ] || printf "tag\tshape\tin\tout\tprefix\tC\tprompts\tdur_s\treq_s\tout_tps\ttotal_tps\tin_tps\tttft_ms\tp99_ttft\ttpot_ms\tp99_tpot\tcompleted\n" > "$TSV"
run() { # shape in out prefix C prompts
  SEED=$(( 1000 + $5 + ${#1} * 37 + $2 ))
  local shape=$1 in=$2 out=$3 pre=$4 C=$5 np=$6 f="${TAG}__${shape}__c${C}.json"
  echo "[$(date +%H:%M:%S)] $shape in=$in out=$out prefix=$pre C=$C prompts=$np"
  nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,memory.used --format=csv,noheader > "$OUT/${f%.json}.gpu_before.csv"
  vllm bench serve --backend openai --endpoint /v1/completions --model "$MODEL" --tokenizer "$TOK" --tokenizer-mode "$TOKMODE" --trust-remote-code \
    --dataset-name random --random-input-len "$in" --random-output-len "$out" --random-range-ratio 0 --random-prefix-len "$pre" \
    --request-rate inf --max-concurrency "$C" --num-prompts "$np" --ignore-eos --seed $SEED \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 --save-result --result-dir "$OUT" --result-filename "$f" --disable-tqdm \
    --metadata "tag=$TAG" "shape=$shape" "prefix_len=$pre" --base-url "http://127.0.0.1:$PORT" > "$OUT/${f%.json}.log" 2>&1
  python3 - "$OUT/$f" "$TAG" "$shape" "$in" "$out" "$pre" "$C" "$np" >> "$TSV" <<"PY"
import json, sys
f, tag, shape, i, o, pre, C, np_ = sys.argv[1:]
try:
    d = json.load(open(f))
except Exception as e:
    print("\t".join([tag, shape, i, o, pre, C, np_, "FAILED", str(e)[:40]])); sys.exit()
dur = d.get("duration") or 1
row = [tag, shape, i, o, pre, C, np_, round(dur, 1), round(d.get("request_throughput", 0), 2), round(d.get("output_throughput", 0)),
       round(d.get("total_token_throughput", 0)), round((d.get("total_input_tokens", 0) or 0) / dur), round(d.get("mean_ttft_ms", 0)),
       round(d.get("p99_ttft_ms", 0)), round(d.get("mean_tpot_ms", 0), 1), round(d.get("p99_tpot_ms", 0), 1), d.get("completed")]
print("\t".join(str(x) for x in row))
PY
  tail -1 "$TSV"
}
# WARMUP: JIT/cudagraph shapes settle before measurements (results discarded)
echo "[$(date +%H:%M:%S)] warm-up"
vllm bench serve --backend openai --endpoint /v1/completions --model "$MODEL" --tokenizer "$TOK" --tokenizer-mode "$TOKMODE" --trust-remote-code --dataset-name random --random-input-len 1024 --random-output-len 64 --random-range-ratio 0 --request-rate inf --max-concurrency 32 --num-prompts 64 --ignore-eos --seed 7 --disable-tqdm --base-url "http://127.0.0.1:$PORT" > "$OUT/warmup.log" 2>&1
vllm bench serve --backend openai --endpoint /v1/completions --model "$MODEL" --tokenizer "$TOK" --tokenizer-mode "$TOKMODE" --trust-remote-code --dataset-name random --random-input-len 256 --random-output-len 32 --random-range-ratio 0 --request-rate inf --max-concurrency 256 --num-prompts 256 --ignore-eos --seed 8 --disable-tqdm --base-url "http://127.0.0.1:$PORT" >> "$OUT/warmup.log" 2>&1
case "$SET" in
  full)
    run short 256 64 0 16 64;        run short 256 64 0 64 128;        run short 256 64 0 256 256
    run router 1024 128 0 16 64;     run router 1024 128 0 64 128;     run router 1024 128 0 256 256
    run promptopt 512 256 3072 16 64; run promptopt 512 256 3072 64 128; run promptopt 512 256 3072 256 256
    run judge 4096 512 0 16 64;      run judge 4096 512 0 64 128;      run judge 4096 512 0 128 128
    run rollout 8192 2048 0 16 32;   run rollout 8192 2048 0 64 64 ;;
  quick)
    run router 1024 128 0 16 64; run router 1024 128 0 64 128; run judge 4096 512 0 64 128; run rollout 8192 2048 0 32 32 ;;
  spec)
    run router 1024 128 0 16 64; run router 1024 128 0 64 128; run judge 4096 512 0 16 64; run judge 4096 512 0 64 128; run rollout 8192 2048 0 16 32 ;;
esac
echo "[$(date +%H:%M:%S)] PROBE DONE $TAG"
