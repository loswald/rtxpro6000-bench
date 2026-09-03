#!/usr/bin/env bash
# probe4.sh <tag> <alias> <tokenizer_path> [tokenizer_mode] [set] -- drives 4 replicas (ports 8000-8003), C and prompts split across ports, rows aggregated
TAG=$1; MODEL=$2; TOK=$3; TOKMODE=${4:-auto}; SET=${5:-full}; PORTS="8000 8001 8002 8003"
OUT=/workspace/results/probe/$TAG; mkdir -p "$OUT"; TSV=$OUT/summary.tsv
[ -f "$TSV" ] || printf "tag\tshape\tin\tout\tprefix\tC\tprompts\tdur_s\treq_s\tout_tps\ttotal_tps\tin_tps\tttft_ms\tp99_ttft\ttpot_ms\tp99_tpot\tcompleted\n" > "$TSV"
bench() { # port C np seed file in out pre
  vllm bench serve --backend openai --endpoint /v1/completions --model "$MODEL" --tokenizer "$TOK" --tokenizer-mode "$TOKMODE" --trust-remote-code \
    --dataset-name random --random-input-len "$6" --random-output-len "$7" --random-range-ratio 0 --random-prefix-len "$8" \
    --request-rate inf --max-concurrency "$2" --num-prompts "$3" --ignore-eos --seed "$4" \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 --save-result --result-dir "$OUT" --result-filename "$5" --disable-tqdm \
    --metadata "tag=$TAG" "port=$1" --base-url "http://127.0.0.1:$1" > "$OUT/${5%.json}.log" 2>&1
}
run() { # shape in out prefix C prompts
  local shape=$1 in=$2 out=$3 pre=$4 C=$5 np=$6
  local pc=$(( C / 4 )); [ "$pc" -lt 1 ] && pc=1
  local pn=$(( np / 4 )); [ "$pn" -lt 8 ] && pn=8
  echo "[$(date +%H:%M:%S)] $shape in=$in out=$out prefix=$pre C=$C (x4 ports @ $pc) prompts=$np"
  local pids=()
  for port in $PORTS; do bench "$port" "$pc" "$pn" $((1234 + port)) "${TAG}__${shape}__c${C}__p${port}.json" "$in" "$out" "$pre" & pids+=($!); done
  wait "${pids[@]}"
  python3 - "$OUT" "${TAG}__${shape}__c${C}__p" "$TAG" "$shape" "$in" "$out" "$pre" "$C" "$np" >> "$TSV" <<"PY"
import glob, json, sys
d, prefix, tag, shape, i, o, pre, C, np_ = sys.argv[1:]
files = sorted(glob.glob(d + "/" + prefix + "*.json"))
rq = ot = tt = it = 0.0; comp = 0; dur = 0.0; ttft = p99t = tpot = p99p = 0.0
for f in files:
    try: j = json.load(open(f))
    except Exception: continue
    c = j.get("completed", 0) or 0; du = j.get("duration") or 1
    rq += j.get("request_throughput", 0) or 0; ot += j.get("output_throughput", 0) or 0; tt += j.get("total_token_throughput", 0) or 0
    it += (j.get("total_input_tokens", 0) or 0) / du; comp += c; dur = max(dur, du)
    ttft += (j.get("mean_ttft_ms", 0) or 0) * c; tpot += (j.get("mean_tpot_ms", 0) or 0) * c
    p99t = max(p99t, j.get("p99_ttft_ms", 0) or 0); p99p = max(p99p, j.get("p99_tpot_ms", 0) or 0)
if comp == 0:
    print("\t".join([tag, shape, i, o, pre, C, np_, "FAILED", str(len(files)) + " files"])); sys.exit()
row = [tag, shape, i, o, pre, C, np_, round(dur, 1), round(rq, 2), round(ot), round(tt), round(it), round(ttft / comp), round(p99t), round(tpot / comp, 1), round(p99p, 1), comp]
print("\t".join(str(x) for x in row))
PY
  tail -1 "$TSV"
}
echo "[$(date +%H:%M:%S)] warm-up (all ports)"
for port in $PORTS; do bench "$port" 16 32 7 "warmup_p${port}.json" 1024 64 0 & done; wait
for port in $PORTS; do bench "$port" 64 64 8 "warmup2_p${port}.json" 256 32 0 & done; wait
find "$OUT" -maxdepth 1 -name "warmup*.json" -delete
case "$SET" in
  full)
    run short 256 64 0 16 64;        run short 256 64 0 64 128;        run short 256 64 0 256 256
    run router 1024 128 0 16 64;     run router 1024 128 0 64 128;     run router 1024 128 0 256 256
    run promptopt 512 256 3072 16 64; run promptopt 512 256 3072 64 128; run promptopt 512 256 3072 256 256
    run judge 4096 512 0 16 64;      run judge 4096 512 0 64 128;      run judge 4096 512 0 128 128
    run rollout 8192 2048 0 16 32;   run rollout 8192 2048 0 64 64 ;;
  tune)
    run router 1024 128 0 256 256; run promptopt 512 256 3072 256 256; run judge 4096 512 0 64 128 ;;
  quick)
    run router 1024 128 0 64 128; run router 1024 128 0 256 256; run judge 4096 512 0 64 128; run rollout 8192 2048 0 32 32 ;;
esac
echo "[$(date +%H:%M:%S)] PROBE4 DONE $TAG"
