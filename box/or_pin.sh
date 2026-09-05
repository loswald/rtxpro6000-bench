#!/usr/bin/env bash
# The same 403 items against OpenRouter with the provider PINNED, so quality can be read per price tier:
# DeepSeek-V4-Flash-0731 from DeepSeek itself ($0.22/$0.66), from the cheapest FP8 host (Baidu, $0.05/$0.10) and from
# the FP4 host at the list price the economics used (Relace, $0.065/$0.18); GLM-5.3-Flash from Z.AI (FP8, the list
# price) and from DeepInfra (FP4, same price). Runs beside or_eval.sh; the key file is the same.
set -u
B=/workspace/bench; R=/workspace/results; OUT=$R/eval_or; KEYFILE=${KEYFILE:-/workspace/.openrouter_key}
[ -s "$KEYFILE" ] || { echo "no key at $KEYFILE"; exit 1; }
EVAL_API_KEY=$(tr -d '\r\n' < "$KEYFILE"); export EVAL_API_KEY
mkdir -p "$OUT"; log(){ echo "[$(date +%H:%M:%S)] $*"; }
CAPS="math=32768,code=20480,knowledge=20480,ifeval=16384,tools=8192,longctx=6144"
# tag | slug | provider name as OpenRouter spells it | sampling extra-body from lists/profiles.tsv (merged with the pin)
RUNS="or_ds_deepseek|deepseek/deepseek-v4-flash-0731|DeepSeek|--temperature 1.0 --top-p 0.95
or_ds_baidu|deepseek/deepseek-v4-flash-0731|Baidu|--temperature 1.0 --top-p 0.95
or_ds_relace|deepseek/deepseek-v4-flash-0731|Relace|--temperature 1.0 --top-p 0.95
or_glm_zai|z-ai/glm-5.3-flash|Z.AI|--temperature 0.95 --top-p 0.95 --extra-body {\"min_p\":0.0}
or_glm_deepinfra|z-ai/glm-5.3-flash|DeepInfra|--temperature 0.95 --top-p 0.95 --extra-body {\"min_p\":0.0}
or_q27_darkbloom|qwen/qwen3.8-27b|Darkbloom|--temperature 1.0 --top-p 0.95 --extra-body {\"top_k\":20,\"min_p\":0.0,\"presence_penalty\":0.0}
or_q27_parasail|qwen/qwen3.8-27b|Parasail|--temperature 1.0 --top-p 0.95 --extra-body {\"top_k\":20,\"min_p\":0.0,\"presence_penalty\":0.0}
or_ds_deepinfra|deepseek/deepseek-v4-flash-0731|DeepInfra|--temperature 1.0 --top-p 0.95"
want="${*:-or_ds_deepseek or_glm_zai or_ds_baidu or_ds_relace or_glm_deepinfra}"
for w in $want; do
  line=$(printf '%s\n' "$RUNS" | grep "^$w|") || { log "unknown $w"; continue; }
  IFS='|' read -r tag slug prov samp <<< "$line"
  if [ -f "$OUT/$tag.json" ] && python3 -c "import json,sys; d=json.load(open('$OUT/$tag.json')); sys.exit(0 if d.get('partial') is False else 1)" 2>/dev/null; then log "$tag done"; continue; fi
  # merge the sampling extra-body (if any) with the provider pin into one JSON object
  base=$(printf '%s' "$samp" | sed -n 's/.*--extra-body \({[^}]*}\).*/\1/p'); [ -z "$base" ] && base='{}'
  samp_nobody=$(printf '%s' "$samp" | sed 's/--extra-body {[^}]*}//')
  body=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); d['provider']={'order':[sys.argv[2]],'allow_fallbacks':False}; print(json.dumps(d,separators=(',',':')))" "$base" "$prov")
  log "$tag: $slug pinned to $prov"
  # shellcheck disable=SC2086
  env -u PYTHONHOME -u PYTHONPATH python3 "$B/evalsuite/run_eval.py" --tag "$tag" --base-urls https://openrouter.ai/api --model "$slug" \
    --out "$OUT" --concurrency "${OR_CONC:-24}" --time-budget "${OR_BUDGET:-2700}" --reasoning --request-timeout 3600 \
    --max-tokens 32768 --max-tokens-family "$CAPS" $samp_nobody --extra-body "$body" > "$OUT/$tag.log" 2>&1
  python3 -c "import json; d=json.load(open('$OUT/$tag.json')); a=d['aggregate']; print('  ', '$tag', 'n', a.get('n_scored'), 'acc', a.get('acc_micro'), 'errors', a.get('n_error'), 'partial', d.get('partial'))" 2>/dev/null || log "  $tag: no result file"
done
log "ORPIN DONE"
